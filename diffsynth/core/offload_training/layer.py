"""
Layer offloading for training — hook-based CPU offload.

CPU copy strategy: all non-trainable params have a permanent CPU copy.
Offloading = deleting the GPU copy (no PCIe transfer).
Loading = copying from the CPU copy to GPU.

Hook lifecycle per module:

No checkpointing:
  forward_pre(load from CPU copy → GPU) → forward() → forward_hook(delete GPU copy)
  backward_pre(load from CPU copy → GPU) → backward() → backward_hook(delete GPU copy)

With checkpointing (use_reentrant=False):
  First forward:
    forward_pre(load → GPU) → forward() → forward_hook(delete GPU copy, mark in_recompute)
  Recomputing forward (during backward):
    forward_pre(load → GPU) → forward() → forward_hook(keep GPU — in_recompute)
  Backward:
    backward_pre(load → GPU) → backward() → backward_hook(delete GPU copy)

_in_recompute tracks modules that have completed their first forward and need
weights kept on GPU during recomputing. It is cleared by reset_in_recompute()
after backward() returns, so the next training step starts fresh.
"""
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


def has_parameters(module: nn.Module) -> bool:
    return len(list(module.parameters())) > 0


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


class LayerOffloadManager:

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        param_size_threshold: int = None,
        verbose: bool = True,
    ):
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self._param_size_threshold = param_size_threshold * 1024 * 1024 if param_size_threshold is not None else None
        self._in_recompute: set = set()
        self._cpu_copies: dict[int, torch.Tensor] = {}  # id(param) -> cpu tensor
        self.verbose = verbose

        trainable_ids = set()
        if not optimize_on_cpu:
            for param in model.parameters():
                if param.requires_grad:
                    trainable_ids.add(id(param))

        if param_size_threshold is not None:
            units = self._find_units_recursive(model)
        else:
            units = [m for m in model.modules() if is_leaf_module(m)]

        units = [m for m in units if has_parameters(m)]
        for module in units:
            self._register_single_module_hooks(module, trainable_ids)

    def _should_force_recurse(self, module: nn.Module) -> bool:
        if is_leaf_module(module):
            return False
        if (
            count_parameters(module) > self._param_size_threshold
            or ('forward' not in type(module).__dict__)
            or (hasattr(module, 'encode') and hasattr(module, 'decode'))
        ):
            return True
        return False

    def _find_units_recursive(self, module: nn.Module) -> list:
        if self._should_force_recurse(module):
            units = []
            for child in module.children():
                units.extend(self._find_units_recursive(child))
            return units
        return [module]

    def _register_single_module_hooks(self, module: nn.Module, trainable_ids: set):
        # Find leaf children with parameters (for backward hook delegation)
        leaf_children = []
        for child in module.modules():
            if child is module:
                continue
            if not has_parameters(child):
                continue
            if not is_leaf_module(child):
                continue
            leaf_children.append(child)

        def forward_pre_hook(mod, args):
            for param in mod.parameters():
                pid = id(param)
                if pid in trainable_ids:
                    continue
                if pid in self._cpu_copies and param.device.type != self.target_device.type:
                    param.data = self._cpu_copies[pid].to(self.target_device, non_blocking=True)

        def forward_hook(mod, args, output):
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            for param in mod.parameters():
                pid = id(param)
                if pid in trainable_ids:
                    continue
                if pid in self._cpu_copies:
                    del param.data

        def backward_pre_hook(mod, grad_output):
            for param in mod.parameters():
                pid = id(param)
                if pid in trainable_ids:
                    continue
                if pid in self._cpu_copies and param.device.type != self.target_device.type:
                    param.data = self._cpu_copies[pid].to(self.target_device, non_blocking=True)

        def backward_hook(mod, grad_input, grad_output):
            if len(leaf_children) > 0:
                return
            for param in mod.parameters():
                pid = id(param)
                if pid in trainable_ids:
                    continue
                if pid in self._cpu_copies:
                    del param.data

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)

        for child in leaf_children:
            def child_backward_pre_hook(mod, grad_output):
                for param in mod.parameters():
                    pid = id(param)
                    if pid in trainable_ids:
                        continue
                    if pid in self._cpu_copies and param.device.type != self.target_device.type:
                        param.data = self._cpu_copies[pid].to(self.target_device, non_blocking=True)

            def child_backward_hook(mod, grad_input, grad_output):
                for param in mod.parameters():
                    pid = id(param)
                    if pid in trainable_ids:
                        continue
                    if pid in self._cpu_copies:
                        del param.data

            child.register_full_backward_pre_hook(child_backward_pre_hook)
            child.register_full_backward_hook(child_backward_hook)

    def reset_in_recompute(self):
        self._in_recompute.clear()


def setup_layer_offload(
    model: nn.Module,
    target_device: torch.device,
    optimize_on_cpu: bool = False,
    param_size_threshold: int = None,
):
    # Move all params to CPU first, so the manager can create CPU copies
    for _, child in model.named_children():
        for param in child.parameters():
            param.data = param.data.to('cpu')

    manager = LayerOffloadManager(model, target_device, optimize_on_cpu, param_size_threshold)
    if not optimize_on_cpu:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(target_device)
    return manager


def move_gradients_to_cpu(model: nn.Module):
    for param in model.parameters():
        if param.grad is not None:
            if param.device.type != 'cpu':
                param.data = param.data.to('cpu', non_blocking=True)
            param.grad = param.grad.to('cpu')
