import torch
from .type_defs import VRAM_LIMIT


class VRAMLimitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "buffer_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10, "step": 0.5}),
        }}

    RETURN_TYPES = (VRAM_LIMIT,)
    RETURN_NAMES = ("vram_limit",)
    FUNCTION = "execute"
    CATEGORY = "DiffSynth/config"

    def execute(self, device="cuda", buffer_size=0.5):
        try:
            if not torch.cuda.is_available():
                return (float("inf"),)
            total = torch.cuda.mem_get_info(device)[1] / (1024 ** 3)
            return (max(total - float(buffer_size), 0.0),)
        except Exception:
            return (float("inf"),)
