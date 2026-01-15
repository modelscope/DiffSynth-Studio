import torch
from contextlib import contextmanager


def npu_autocast_patch_wrapper(func):
    @contextmanager
    def wrapper(*args, **kwargs):
        flag = False
        if "npu" in args or ("device_type" in kwargs and kwargs["device_type"] == "npu"):
            if torch.float32 in args or ("dtype" in kwargs and kwargs["dtype"] == torch.float32):
                flag = True
        with func(*args, **kwargs) as ctx:
            if flag:
                torch.npu.set_autocast_enabled(True)
            yield ctx
    return wrapper


def npu_autocast_patch():
    torch.amp.autocast = npu_autocast_patch_wrapper(torch.amp.autocast)
    torch.autocast = npu_autocast_patch_wrapper(torch.autocast)
