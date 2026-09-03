import torch
from .type_defs import VRAM_CONFIG


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_DEVICES = ["cpu", "cuda"]
_DTYPE_NAMES = list(_DTYPES)


class VRAMConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        defaults = [("offload_device", "cpu"), ("offload_dtype", "bfloat16"),
                    ("onload_device", "cpu"), ("onload_dtype", "bfloat16"),
                    ("preparing_device", "cuda"), ("preparing_dtype", "bfloat16"),
                    ("computation_device", "cuda"), ("computation_dtype", "bfloat16")]
        for name, default in defaults:
            required[name] = (_DEVICES if name.endswith("device") else _DTYPE_NAMES, {"default": default})
        return {"required": required}

    RETURN_TYPES = (VRAM_CONFIG,)
    RETURN_NAMES = ("vram_config",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, **kwargs):
        return ({name: (_DTYPES[value] if name.endswith("dtype") else value)
                 for name, value in kwargs.items()},)
