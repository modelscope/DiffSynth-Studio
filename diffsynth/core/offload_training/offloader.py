import torch


class BaseParamOffloader:
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device):
        self.param = param
        self.target_device = target_device

    def onload(self):
        pass

    def offload(self):
        pass

    def offload_grad(self):
        pass


class StaticParamOffloader(BaseParamOffloader):
    def __init__(self, param: torch.nn.Parameter, target_device: torch.device):
        super().__init__(param, target_device)
        self.cpu_copy = param.data.cpu().pin_memory()

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
        self.param.grad = self.param.grad.to('cpu', non_blocking=True)


class AlwaysOnGPUParamOffloader(BaseParamOffloader):
    def __init__(self, param, target_device):
        super().__init__(param, target_device)
        self.param.data = self.param.data.to(self.target_device)
