from .controlnet_input import ControlNetInput
from .annotator import Annotator

_LAZY = {"video_to_lineart": "video_lineart", "video_to_blur": "video_blur", "caption_windows": "video_caption"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)


__all__ = ["ControlNetInput", "Annotator", *_LAZY]
