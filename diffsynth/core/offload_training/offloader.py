import torch
from .memory_buffer import PinnedArenaPool


class OffloaderMixin:
    def onload(self):
        pass

    def offload(self):
        pass

    def offload_grad(self):
        pass


class BaseParamOffloader(OffloaderMixin):
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device):
        self.param = param
        self.target_device = target_device

class StaticParamOffloader(BaseParamOffloader):
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device, pinned_pool: PinnedArenaPool = None):
        super().__init__(param, target_device)
        cpu_data = param.data.cpu().detach().contiguous()
        if pinned_pool is not None:
            self.cpu_copy = pinned_pool.allocate_like(cpu_data)
        else:
            self.cpu_copy = cpu_data.pin_memory()
        param.data = self.cpu_copy

    def onload(self):
        self.param.data = self.cpu_copy.to(self.target_device, non_blocking=True)

    def offload(self):
        self.param.data = self.cpu_copy


class TrainableParamOffloader(BaseParamOffloader):
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device):
        super().__init__(param, target_device)
        assert param.requires_grad, "TrainableParamOffloader can only be used with trainable parameters"

    def onload(self):
        self.param.data = self.param.data.to(self.target_device, non_blocking=True)

    def offload(self):
        self.param.data = self.param.data.to('cpu', non_blocking=True)

    def offload_grad(self):
        if self.param.grad is not None:
            self.param.grad = self.param.grad.to('cpu', non_blocking=True)
        else:
            print(f"Warning: Attempting to offload grad for param with shape {self.param.shape} but grad is None")


class AlwaysOnGPUParamOffloader(BaseParamOffloader):
    def __init__(self, param, target_device):
        super().__init__(param, target_device)
        self.param.data = self.param.data.to(self.target_device)


class BufferOffloader(OffloaderMixin):
    def __init__(self, module: torch.nn.Module, buf_name: str, buf: torch.Tensor, target_device: torch.device, pinned_pool: PinnedArenaPool = None):
        self.module = module
        self.buf_name = buf_name
        self.target_device = target_device
        cpu_data = buf.data.cpu().contiguous()
        if pinned_pool is not None:
            self.cpu_copy = pinned_pool.allocate_like(cpu_data)
        else:
            self.cpu_copy = cpu_data.pin_memory()

    def onload(self):
        self.module._buffers[self.buf_name] = self.cpu_copy.to(self.target_device, non_blocking=True)

    def offload(self):
        self.module._buffers[self.buf_name] = self.cpu_copy
