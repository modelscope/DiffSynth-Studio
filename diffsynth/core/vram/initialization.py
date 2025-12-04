import torch
from contextlib import contextmanager


@contextmanager
def skip_model_initialization(device=torch.device("meta")):

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    old_register_parameter = torch.nn.Module.register_parameter
    torch.nn.Module.register_parameter = register_empty_parameter
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
