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
from .offloader import StaticParamOffloader, TrainableParamOffloader, AlwaysOnGPUParamOffloader, BufferOffloader
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


def has_parameters(module: nn.Module) -> bool:
    return len(list(module.parameters())) > 0


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


class UnitWiseParamManager:
    def __init__(self, model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False, params: list = None, buffers: list = None):
        self.model = model
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self.param_offloaders = {}
        params = params or model.parameters()
        for param in params:
            if not param.requires_grad:
                self.param_offloaders[id(param)] = StaticParamOffloader(param, target_device)
                print(f"Registering StaticParamOffloader for param with shape {param.shape} and {param.numel()} elements (non-trainable)")
            else:
                if optimize_on_cpu:
                    self.param_offloaders[id(param)] = TrainableParamOffloader(param, target_device)
                    print(f"Registering TrainableParamOffloader for param with shape {param.shape} and {param.numel()} elements (trainable, optimize_on_cpu=True)")
                else:
                    AlwaysOnGPUParamOffloader(param, target_device)
                    print(f"Registering AlwaysOnGPUParamOffloader for param with shape {param.shape} and {param.numel()} elements (trainable, optimize_on_cpu=False)")
        if buffers is not None and len(buffers) > 0:
            for mod, buf_name, buf in buffers:
                self.param_offloaders[id(buf)] = BufferOffloader(mod, buf_name, buf, target_device)
                print(f"Registering BufferOffloader for buffer with shape {buf.shape} and {buf.numel()} elements in module {type(mod).__name__}")

    def move_gradients_to_cpu(self):
        for offloader in self.param_offloaders.values():
            offloader.offload_grad()

    def onload_module(self, module: nn.Module):
        for param in module.parameters(recurse=False):
            if id(param) in self.param_offloaders:
                self.param_offloaders[id(param)].onload()
        for name, buf in module.named_buffers(recurse=False):
            if id(buf) in self.param_offloaders:
                self.param_offloaders[id(buf)].onload()

    def offload_module(self, module: nn.Module):
        for param in module.parameters(recurse=False):
            if id(param) in self.param_offloaders:
                self.param_offloaders[id(param)].offload()
        for name, buf in module.named_buffers(recurse=False):
            if id(buf) in self.param_offloaders:
                self.param_offloaders[id(buf)].offload()

class UnitWiseHookManager:
    def __init__(
        self,
        model: torch.nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        verbose: bool = False,
        params: list = None,
        buffers: list = None,
    ):
        self._verbose = verbose
        print(f"Initializing UnitWiseHookManager for module {type(model).__name__} on device {target_device} with optimize_on_cpu={optimize_on_cpu}")
        self.param_manager = UnitWiseParamManager(model, target_device, optimize_on_cpu, params=params, buffers=buffers)
        self._in_recompute: set = set()
        self._register_hooks_for_unit(model)

    def _register_hooks_for_unit(self, module: nn.Module):
        def forward_pre_hook(mod, args):
            if self._verbose:
                print(f"Forward Pre Hook: Loading {type(mod).__name__} to {self.param_manager.target_device}")
            self.param_manager.onload_module(mod)

        def forward_hook(mod, args, output):
            if self._verbose:
                print(f"Forward Hook: Offloading {type(mod).__name__} to CPU") if not mod in self._in_recompute else print(f"Forward Hook: Keeping {type(mod).__name__} on GPU for recompute")
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            self.param_manager.offload_module(mod)

        def backward_pre_hook(mod, grad_output):
            if self._verbose:
                print(f"Backward Pre Hook: Loading {type(mod).__name__} to {self.param_manager.target_device} for backward") if not mod in self._in_recompute else print(f"Backward Pre Hook: Keeping {type(mod).__name__} on GPU for backward")
            self.param_manager.onload_module(mod)

        def backward_hook(mod, grad_input, grad_output):
            if self._verbose:
                print(f"Backward Hook: Offloading {type(mod).__name__} to CPU after backward")
            self.param_manager.offload_module(mod)

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        if is_leaf_module(module):
            module.register_full_backward_hook(backward_hook)
        else:
            # Parent module backward_hook fires before child backward completes.
            # Register on leaf children instead.
            sub_modules = [m for m in module.modules() if is_leaf_module(m) and has_parameters(m)]
            for sub_mod in sub_modules:
                sub_mod.register_full_backward_hook(backward_hook)

    def reset_in_recompute(self):
        self._in_recompute.clear()

    def after_backward(self):
        self.reset_in_recompute()
        self.param_manager.move_gradients_to_cpu()

    @property
    def managed_param_ids(self):
        return set(self.param_manager.param_offloaders.keys())

class OffloadTrainingManager:

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        param_size_threshold: int = None,  # in MB
        verbose: bool = False,
    ):
        self.model = model
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self.hierarchy = dict()
        param_size_threshold = param_size_threshold * 1024 * 1024 if param_size_threshold is not None else None
        self._register_units(model, target_device, optimize_on_cpu, param_size_threshold)

    def _register_units(self, model: nn.Module, target_device: torch.device, optimize_on_cpu: bool = False, param_size_threshold: int = None):
        units = self._find_units_recursive(model, param_size_threshold)
        for unit in units:
            print(f"Registering offload hooks for unit: {self._get_module_hierarchy(unit)}")
        self.units = [UnitWiseHookManager(unit, self.target_device, self.optimize_on_cpu) for unit in units]

        managed_param_ids = set().union(*[unit.managed_param_ids for unit in self.units])
        orphan_params, orphan_buffers = self._find_orphan_params_and_buffers(model, managed_param_ids)
        for orphan_module in set(orphan_params.keys()) | set(orphan_buffers.keys()):
            params = orphan_params.get(orphan_module, [])
            buffers = orphan_buffers.get(orphan_module, [])
            print(f"Registering orphan hooks for module: {self._get_module_hierarchy(orphan_module)} with {len(params)} orphan params and {len(buffers)} orphan buffers")
            self.units.append(UnitWiseHookManager(orphan_module, target_device, optimize_on_cpu, params=params, buffers=buffers))

    def _find_orphan_params_and_buffers(self, model: nn.Module, managed_param_ids: set):
        orphan_params_by_module = {}
        for _, mod in model.named_modules():
            for param in mod.parameters(recurse=False):
                if id(param) not in managed_param_ids:
                    orphan_params_by_module.setdefault(mod, []).append(param)

        # Collect orphan buffers grouped by owner module
        orphan_buffers_by_module = {}
        for _, mod in model.named_modules():
            for name, buf in mod.named_buffers(recurse=False):
                orphan_buffers_by_module.setdefault(mod, []).append((mod, name, buf))
        
        return orphan_params_by_module, orphan_buffers_by_module

    def init_module_hierarchy(self, module: nn.Module) -> str:
        for name, mod in self.model.named_modules():
            if mod is module:
                self.hierarchy[id(mod)] = name

    def _get_module_hierarchy(self, module: nn.Module) -> str:
        if id(module) not in self.hierarchy:
            self.init_module_hierarchy(module)
        return self.hierarchy.get(id(module), "UnknownModule")

    def _find_units_recursive(self, module: nn.Module, param_size_threshold: int = None) -> list:
        if param_size_threshold is None:
            return [m for m in module.modules() if is_leaf_module(m) and has_parameters(m)]
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
