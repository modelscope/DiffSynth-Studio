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

def has_parameters(module: nn.Module) -> bool:
    return len(list(module.parameters())) > 0

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0

class OffloadTrainingManager:

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        optimize_on_cpu: bool = False,
        param_size_threshold: int = None, # in MB
        verbose: bool = False,
    ):
        self.target_device = target_device
        self.optimize_on_cpu = optimize_on_cpu
        self._param_size_threshold = param_size_threshold * 1024 * 1024 if param_size_threshold is not None else None
        self._in_recompute: set = set()
        self._params_on_gpu: set = set()
        self.verbose = verbose
        self.model = model
        self._name_map = {id(m): n for n, m in model.named_modules()}
        if not optimize_on_cpu:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(target_device)
                    self._params_on_gpu.add(id(param))
        if param_size_threshold is not None:
            units = self._find_units_recursive(model)
        else:
            units = [m for m in model.modules() if is_leaf_module(m)]

        units = [m for m in units if has_parameters(m)]
        for module in units:
            self._register_single_module_hooks(module)
            name = self._name_map.get(id(module), "?")
            # print(f"Registered hooks for module {name} ({type(module).__name__}, {count_parameters(module)} parameters)")

    def _should_force_recurse(self, module: nn.Module) -> bool:
        if is_leaf_module(module):
            return False
        if (
            count_parameters(module) > self._param_size_threshold
            or ('forward' not in type(module).__dict__)
            or (hasattr(module, 'encode') and hasattr(module, 'decode') # special handling for vae modules TODO: may find a better implementation
            )
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

    def _register_single_module_hooks(self, module: nn.Module):
        def forward_pre_hook(mod, args):
            if self.verbose:
                print(f"Forward Pre Hook: Loading {type(mod).__name__} to {self.target_device}")
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device, non_blocking=True)

        def forward_hook(mod, args, output):
            if self.verbose:
                print(f"Forward Hook: Offloading {type(mod).__name__} to CPU") if not mod in self._in_recompute else print(f"Forward Hook: Keeping {type(mod).__name__} on GPU for recompute")
            if mod in self._in_recompute:
                return
            self._in_recompute.add(mod)
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        def backward_pre_hook(mod, grad_output):
            if self.verbose:
                print(f"Backward Pre Hook: Loading {type(mod).__name__} to {self.target_device} for backward") if not mod in self._in_recompute else print(f"Backward Pre Hook: Keeping {type(mod).__name__} on GPU for backward")
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != self.target_device.type:
                    param.data = param.data.to(self.target_device, non_blocking=True)

        def backward_hook(mod, grad_input, grad_output):
            if self.verbose:
                print(f"Backward Hook: Offloading {type(mod).__name__} to CPU after backward")
            for param in mod.parameters():
                if id(param) not in self._params_on_gpu and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        if is_leaf_module(module):        
            module.register_full_backward_hook(backward_hook)
        else:
            # backward modules must register on leaves to ensure hooks are called after gradients are computed for the entire module
            sub_modules = [m for m in module.modules() if is_leaf_module(m) and has_parameters(m)]
            for sub_mod in sub_modules:
                sub_mod.register_full_backward_hook(backward_hook)

    def reset_in_recompute(self):
        self._in_recompute.clear()

    def move_gradients_to_cpu(model: nn.Module):
        for param in model.parameters():
            if param.grad is not None:
                if param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)
                param.grad = param.grad.to('cpu')

    # run after backward() and before optimizer.step()
    def after_backward(self,):
        self.reset_in_recompute()
        if self.optimize_on_cpu:
            self.move_gradients_to_cpu(self.model)
