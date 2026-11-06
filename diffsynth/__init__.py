import torch

if getattr(torch.version, "musa", None) is not None:
    import torchada

from .core import *
