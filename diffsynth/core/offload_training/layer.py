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
from collections import defaultdict


class LayerOffloadManager:
    """
    Manages per-module CPU offload for training.

    Registers forward/backward hooks on modules at a configurable granularity.
    Modules whose total parameter count is below `param_size_threshold` are
    registered as a single offload unit; larger modules are recursively split
    into their children.

    optimize_on_cpu behavior:
    - True: ALL params are offloaded. Gradients stay on GPU during backward.
      Caller must move grads to CPU AFTER accelerator.backward() returns.
    - False: Trainable params stay on GPU. Non-trainable params do offload.
      Optimizer runs on GPU automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        param_size_threshold: int = 500,  # 500 MB
    ):
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self._param_size_threshold = param_size_threshold * 1024 * 1024  # MB → bytes
        self._in_recompute: set = set()
        self._trainable_params: set = set()
        if not optimize_on_cpu:
            for param in model.parameters():
                if param.requires_grad:
                    self._trainable_params.add(id(param))
        # Store model reference for reset
        self._model = model
        self._register_hooks(model)

    def _count_params(self, module: nn.Module) -> int:
        """Total parameters in a module and all its children."""
        return sum(p.numel() for p in module.parameters())

    def _should_force_recurse(self, module: nn.Module) -> bool:
        """Force recursion in two cases:
        1. Module doesn't define its own forward (e.g. nn.ModuleList, pipeline wrappers).
        2. Module has forward() but pipeline calls encode()/decode() directly,
           bypassing forward(). Common for VAEs."""
        if 'forward' not in type(module).__dict__:
            return True
        if hasattr(module, 'encode') and hasattr(module, 'decode'):
            return True
        return False

    def _find_offload_units(self, model: nn.Module) -> list:
        """Recursively find modules to register hooks on.

        Always recurses into top-level children to avoid registering the
        pipeline wrapper itself as a unit (which would conflict with child hooks).
        For child modules: if total params <= threshold, register hook on it
        as a whole (fewer, larger transfers). If > threshold or lacks forward(),
        recurse into children.
        """
        units = []
        for child in model.children():
            units.extend(self._find_units_recursive(child))
        return units

    def _find_units_recursive(self, module: nn.Module) -> list:
        total_params = self._count_params(module)
        if self._should_force_recurse(module) or total_params > self._param_size_threshold:
            units = []
            for child in module.children():
                units.extend(self._find_units_recursive(child))
            return units
        return [module]

    def _register_hooks(self, model: nn.Module):
        for module in self._find_offload_units(model):
            self._register_single_module_hooks(module)

    def _should_move_params(self, module: nn.Module, grad_input=None) -> bool:
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    return True
        for param in module.parameters(recurse=True):
            if param.requires_grad:
                return True
        return False

    def _register_single_module_hooks(self, module: nn.Module):
        module_type = type(module).__name__
        def forward_pre_hook(mod, args):
            for param in mod.parameters(recurse=True):
                if id(param) not in self._trainable_params and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device)

        def forward_hook(mod, args, output):
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            for param in mod.parameters(recurse=True):
                if id(param) not in self._trainable_params and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)

        # Always register backward hooks
        def backward_pre_hook(mod, grad_input):
            if not self._should_move_params(mod, grad_input):
                return
            for param in mod.parameters(recurse=True):
                if id(param) not in self._trainable_params and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device)

        def backward_hook(mod, grad_input, grad_output):
            if not self._should_move_params(mod, grad_input):
                return
            for param in mod.parameters(recurse=True):
                if id(param) not in self._trainable_params and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)

    def reset_in_recompute(self):
        """Call after accelerator.backward() returns to prepare for next training step.

        During gradient checkpointing recomputation, forward_hook returns early
        (_in_recompute is set), keeping weights on GPU. After backward returns,
        we must explicitly move all non-trainable params back to CPU.
        """
        self._in_recompute.clear()
        # Move all non-trainable params back to CPU
        for child in self._model.children():
            for param in child.parameters():
                if id(param) not in self._trainable_params and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)


def setup_layer_offload(
    model: nn.Module,
    target_device: torch.device,
    optimize_on_cpu: bool = False,
    param_size_threshold: int = 500,  # 500 MB
):
    """
    Full setup for layer offload training — moves params to CPU and enables offload.

    Call this AFTER accelerator.prepare(model, ...).

    Args:
        model: The prepared model (can be DDP-wrapped). Must have named_children().
        target_device: The GPU device for computation.
        optimize_on_cpu: If True, all params are offloaded, optimizer runs on CPU.
                        If False, trainable params stay on GPU, optimizer runs on GPU.
        param_size_threshold: Modules with total params below this (in MB) are registered
                              as a single offload unit. Larger modules are recursively
                              split into their children. Default: 500 MB.

    Returns:
        A single LayerOffloadManager instance. Call reset_in_recompute() on it
        after accelerator.backward() to prepare for the next training step.
    """
    # Move params back to CPU
    for child in model.named_children():
        for param in child[1].parameters():
            param.data = param.data.to('cpu', non_blocking=True)

    # Register offload hooks on all leaf modules
    manager = LayerOffloadManager(model, target_device, optimize_on_cpu, param_size_threshold)
    if not optimize_on_cpu:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(target_device)
    return manager


def move_gradients_to_cpu(model: nn.Module):
    """
    Move all parameters and their gradients to CPU. Call this AFTER accelerator.backward() returns.
    Only needed when optimize_on_cpu=True.
    """
    for param in model.parameters():
        if param.grad is not None:
            # Move param data to CPU first if still on GPU (e.g. left by backward_hook race),
            # then move gradient to CPU. PyTorch requires grad and param on same device.
            if param.device.type != 'cpu':
                param.data = param.data.to('cpu', non_blocking=True)
            param.grad = param.grad.to('cpu')
