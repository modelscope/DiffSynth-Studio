"""
Layer offloading for training — hook-based CPU offload.

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
from .memory_buffer import PinnedArenaPool, BaseBufferPool
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed with respect to module outputs")


def has_parameters(module: nn.Module) -> bool:
    return len(list(module.parameters())) > 0

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


class UnitWiseParamManager:
    def __init__(self, model: nn.Module, target_device: torch.device, enable_optimizer_cpu_offload: bool = False, params: list = None, buffers: list = None, memory_buffer: BaseBufferPool = None):
        self.model = model
        self.target_device = target_device
        self.param_offloaders = {}
        for param in (model.parameters() if params is None else params):
            if not param.requires_grad:
                self.param_offloaders[id(param)] = StaticParamOffloader(param, target_device, memory_buffer=memory_buffer)
            else:
                if enable_optimizer_cpu_offload:
                    self.param_offloaders[id(param)] = TrainableParamOffloader(param, target_device)
                else:
                    self.param_offloaders[id(param)] = AlwaysOnGPUParamOffloader(param, target_device)
        if buffers is not None and len(buffers) > 0:
            for mod, buf_name, buf in buffers:
                self.param_offloaders[id(buf)] = BufferOffloader(mod, buf_name, buf, target_device, memory_buffer=memory_buffer)

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
    def __init__(self, model: nn.Module, target_device: torch.device, enable_optimizer_cpu_offload: bool = False,
                 params: list = None, buffers: list = None, memory_buffer: BaseBufferPool = None):
        self.param_manager = UnitWiseParamManager(model, target_device, enable_optimizer_cpu_offload, params=params, buffers=buffers, memory_buffer=memory_buffer)
        self._in_recompute: set = set()
        self._register_hooks(model)

    def _register_hooks(self, module: nn.Module):
        def forward_pre_hook(mod, args):
            self.param_manager.onload_module(mod)

        def forward_hook(mod, args, output):
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            self.param_manager.offload_module(mod)

        def backward_pre_hook(mod, grad_output):
            self.param_manager.onload_module(mod)

        def backward_hook(mod, grad_input, grad_output):
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

    def after_backward(self):
        self._in_recompute.clear()
        self.param_manager.move_gradients_to_cpu()

    @property
    def managed_param_ids(self):
        return set(self.param_manager.param_offloaders.keys())


class OffloadTrainingManager:
    def __init__(self, model: nn.Module, target_device: torch.device, enable_optimizer_cpu_offload: bool = False, cpu_offload_split_threshold: int = None):
        self.model = model
        self.target_device = target_device
        self.enable_optimizer_cpu_offload = enable_optimizer_cpu_offload
        cpu_offload_split_threshold = cpu_offload_split_threshold * 1024 * 1024 if cpu_offload_split_threshold is not None else None
        self._register_units(model, target_device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)

    def _register_units(self, model: nn.Module, target_device: torch.device, enable_optimizer_cpu_offload: bool, cpu_offload_split_threshold: int = None):
        self.memory_buffer = PinnedArenaPool.from_model(model)
        units = self._find_units_recursive(model, cpu_offload_split_threshold)
        self.units = [UnitWiseHookManager(u, target_device, enable_optimizer_cpu_offload, memory_buffer=self.memory_buffer) for u in units]

        managed_param_ids = set().union(*[unit.managed_param_ids for unit in self.units])
        orphan_params, orphan_buffers = self._find_orphan_params_and_buffers(model, managed_param_ids)
        for orphan_module in set(orphan_params.keys()) | set(orphan_buffers.keys()):
            params = orphan_params.get(orphan_module, [])
            buffers = orphan_buffers.get(orphan_module, [])
            self.units.append(UnitWiseHookManager(orphan_module, target_device, enable_optimizer_cpu_offload, params=params, buffers=buffers, memory_buffer=self.memory_buffer))

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

    def _find_units_recursive(self, module: nn.Module, cpu_offload_split_threshold: int = None) -> list:
        if cpu_offload_split_threshold is None:
            return [m for m in module.modules() if is_leaf_module(m) and has_parameters(m)]
        if self._should_force_recurse(module, cpu_offload_split_threshold):
            units = []
            for child in module.children():
                units.extend(self._find_units_recursive(child, cpu_offload_split_threshold))
            return units
        return [module]

    def _should_force_recurse(self, module: nn.Module, cpu_offload_split_threshold: int = None) -> bool:
        if is_leaf_module(module):
            return False
        if (
            count_parameters(module) > cpu_offload_split_threshold
            or ('forward' not in type(module).__dict__)
            or (hasattr(module, 'encode') and hasattr(module, 'decode'))
        ):
            return True
        return False

    # run after backward() and before optimizer.step()
    def after_backward(self):
        for unit in self.units:
            unit.after_backward()
        torch.cuda.synchronize()
