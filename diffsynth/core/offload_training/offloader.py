import torch
from .memory_buffer import BaseBufferPool


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
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device, memory_buffer: BaseBufferPool = None):
        super().__init__(param, target_device)
        cpu_data = param.data.cpu().detach().contiguous()
        self.cpu_copy = memory_buffer.allocate_like(cpu_data) if memory_buffer is not None else cpu_data.pin_memory()
        self._placeholder = torch.empty(0, device=target_device, dtype=param.dtype)
        param.data = self._placeholder

    def onload(self):
        self.param.data = self.cpu_copy.to(self.target_device, non_blocking=True)

    def offload(self):
        self.param.data = self._placeholder


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


class AlwaysOnGPUParamOffloader(BaseParamOffloader):
    def __init__(self, param, target_device):
        super().__init__(param, target_device)
        self.param.data = self.param.data.to(self.target_device)


class BufferOffloader(OffloaderMixin):
    def __init__(self, module: torch.nn.Module, buf_name: str, buf: torch.Tensor, target_device: torch.device, memory_buffer: BaseBufferPool = None):
        self.module = module
        self.buf_name = buf_name
        self.target_device = target_device
        cpu_data = buf.data.cpu().contiguous()
        self.cpu_copy = memory_buffer.allocate_like(cpu_data) if memory_buffer is not None else cpu_data.pin_memory()

    def onload(self):
        self.module._buffers[self.buf_name] = self.cpu_copy.to(self.target_device, non_blocking=True)

    def offload(self):
        self.module._buffers[self.buf_name] = self.cpu_copy
