from abc import ABC
import torch


QUANT_BACKENDS = {}


class QuantBackend(ABC):
    """
    Adapter between the framework and a quantization library (bnb / torchao / custom).
    Subclasses are registered in `QUANT_BACKENDS` ({name -> class}) and instantiated by
    `QuantizeConfig` with the method's backend config. A backend operates on single
    Linears; model-level traversal and replacement is done by the `QuantizeConfig` methods.

    The quantized Linear produced by a backend must satisfy:
    (a) It is an `nn.Linear` drop-in: `forward(x)` internally performs dequant + matmul.
    (b) `.to(...)` moves devices but never re-types the packed weight / quant state:
        a dtype cast (`.to(dtype)` / `.half()` / `.float()`) must leave their storage
        format and values intact. Non-float packed storage (bnb uint8) satisfies this
        structurally; tensor subclasses intercept the cast (torchao); quantized Linears
        built from plain float parameters/buffers must guard the module's `_apply`
        themselves (see `Fp8Linear` in `diffsynth.models.ideogram4_dit`).
    (c) `state_dict()` / `load_state_dict(assign=True)` round-trips (optionally via
        `flatten_state_dict` / `unflatten_state_dict`).
    (d) (Training branch only) `forward` is differentiable w.r.t. its input.
    """

    name: str = ""

    def __init__(self, config=None):
        self.config = config

    def capabilities(self) -> dict:
        return {
            "is_serializable": False,
            "is_trainable": False,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def validate_environment(self):
        return

    def create_quantized_linear(self, linear: torch.nn.Linear, compute_dtype: torch.dtype, device=None) -> torch.nn.Module:
        raise NotImplementedError(
            f"Backend `{self.name}` cannot quantize an fp model online. Use a method whose "
            "backend supports it, or load an already quantized checkpoint."
        )

    def create_quantized_linear_shell(self, linear: torch.nn.Linear, compute_dtype: torch.dtype) -> torch.nn.Module:
        raise NotImplementedError(
            f"Backend `{self.name}` cannot load pre-quantized checkpoints. Use "
            "`load_prequantized=False` to quantize an fp model online instead."
        )

    def dequantize_to_linear(self, module: torch.nn.Module, compute_dtype: torch.dtype) -> torch.nn.Linear:
        raise NotImplementedError(
            f"Backend `{self.name}` cannot dequantize back to `nn.Linear`, so "
            '`mode="dequant_once"` is unavailable.'
        )

    def quantized_linear_classes(self) -> tuple:
        return ()

    def is_quantized_linear(self, module) -> bool:
        classes = self.quantized_linear_classes()
        return len(classes) > 0 and isinstance(module, classes)

    def flatten_state_dict(self, state_dict: dict):
        return state_dict, {}

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        return state_dict


def register_quant_backend(name):
    def decorator(cls):
        cls.name = name
        QUANT_BACKENDS[name] = cls
        return cls
    return decorator
