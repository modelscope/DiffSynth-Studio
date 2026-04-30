"""
Layer offloading for training — hook-based CPU offload.

At any time, only one module's weights reside on GPU. The rest stay in CPU memory.

Hook lifecycle per module:

No checkpointing:
  forward_pre(load→GPU) → forward() → forward_hook(offload→CPU, mark in_recompute)
  backward_pre(load→GPU) → backward() → backward_hook(offload→CPU)

With checkpointing (use_reentrant=False):
  First forward:
    forward_pre(load→GPU) → forward() → forward_hook(offload→CPU, mark in_recompute)
  Recomputing forward (during backward):
    forward_pre(load→GPU) → forward() → forward_hook(in_recompute=True → keep GPU)
  Backward:
    backward_pre(load→GPU) → backward() → backward_hook(offload→CPU)

_in_recompute tracks modules that have completed their first forward and need
weights kept on GPU during recomputing. It is cleared by reset_in_recompute()
after backward() returns, so the next training step starts fresh.
"""
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


class LayerOffloadManager:

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
    ):
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        # Track which modules are in the middle of recomputation to avoid offloading them
        self._in_recompute: set = set()
        # Track which parameters are on GPU to avoid unnecessary transfers. If optimize_on_cpu is True, all params stay on CPU and we skip GPU transfers.
        self._params_on_gpu: set = set()
        if not optimize_on_cpu:
            for param in model.parameters():
                if param.requires_grad:
                    self._params_on_gpu.add(id(param))
        for module in model.modules():
            if len(list(module.children())) == 0 and len(list(module.parameters(recurse=False))) > 0:
                self._register_single_module_hooks(module)

    def _register_single_module_hooks(self, module: nn.Module):
        def forward_pre_hook(mod, args):
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device, non_blocking=True)

        def forward_hook(mod, args, output):
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        def backward_pre_hook(mod, grad_output):
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device, non_blocking=True)

        def backward_hook(mod, grad_input, grad_output):
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)

    def reset_in_recompute(self):
        self._in_recompute.clear()


def setup_layer_offload(
    model: nn.Module,
    target_device: torch.device,
    optimize_on_cpu: bool = False,
):
    manager = LayerOffloadManager(model, target_device, optimize_on_cpu)
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
