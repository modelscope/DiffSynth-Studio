"""
Layer offloading for training — hook-based CPU offload.

Two offload strategies:
1. Non-trainable params: permanent CPU copy kept at all times.
   Offload = reassign param.data to CPU copy (GPU tensor freed, no PCIe transfer).
   Load = copy from CPU copy to GPU.

2. Trainable params: no permanent CPU copy (they change during training).
   Offload = param.data.to('cpu') (actual PCIe transfer).
   Load = param.data.to(target_device).

Hook lifecycle per module:

No checkpointing:
  forward_pre(load→GPU) → forward() → forward_hook(offload)
  backward_pre(load→GPU) → backward() → backward_hook(offload)

With checkpointing (use_reentrant=False):
  First forward:
    forward_pre(load→GPU) → forward() → forward_hook(offload, mark in_recompute)
  Recomputing forward (during backward):
    forward_pre(load→GPU) → forward() → forward_hook(in_recompute=True → keep GPU)
  Backward:
    backward_pre(load→GPU) → backward() → backward_hook(offload)
"""
import torch
import torch.nn as nn
import warnings
from .layer import UnitWiseHookManager, is_leaf_module, count_parameters
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


class OffloadTrainingManager:

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        param_size_threshold: int = None,  # in MB
    ):
        self.model = model
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        param_size_threshold = param_size_threshold * 1024 * 1024 if param_size_threshold is not None else None
        self._register_units(model, target_device, optimize_on_cpu, param_size_threshold)

    def _register_units(self, model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False, param_size_threshold: int = None):
        units = self._find_units_recursive(model, param_size_threshold)
        self.units = [UnitWiseHookManager(unit, target_device, optimize_on_cpu) for unit in units]

    def _find_units_recursive(self, module: nn.Module, param_size_threshold: int = None) -> list:
        if param_size_threshold is None:
            return [m for m in module.modules() if is_leaf_module(m)]
        if self._should_force_recurse(module, param_size_threshold):
            units = []
            for child in module.children():
                units.extend(self._find_units_recursive(child, param_size_threshold))
            return units
        return [module]

    def _should_force_recurse(self, module: nn.Module, param_size_threshold: int = None) -> bool:
        if is_leaf_module(module):
            return False
        if (
            count_parameters(module) > param_size_threshold
            or ('forward' not in type(module).__dict__)
            or (hasattr(module, 'encode') and hasattr(module, 'decode'))
        ):
            return True
        return False

    # run after backward() and before optimizer.step()
    def after_backward(self):
        for unit in self.units:
            unit.after_backward()

