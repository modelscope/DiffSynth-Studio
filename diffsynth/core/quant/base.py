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
        format and values intact.
    (c) `state_dict()` / `load_state_dict(assign=True)` round-trips (optionally via
        `flatten_state_dict` / `unflatten_state_dict`).
    (d) (Training branch only) `forward` is differentiable w.r.t. its input, so
        gradients can pass through frozen quantized layers to reach LoRA branches.
        Declared statically by `capabilities()["is_differentiable"]` and verifiable
        at runtime by calling `check_differentiable` on a quantized Linear the
        backend produced.
    """

    name: str = ""

    def __init__(self, config=None):
        self.config = config

    def capabilities(self) -> dict:
        return {
            "is_serializable": False,
            "is_differentiable": False,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def validate_environment(self):
        return

    def create_quantized_linear(self, linear: torch.nn.Linear, compute_device=None, model_device=None) -> torch.nn.Module:
        raise NotImplementedError(
            f"Backend `{self.name}` cannot quantize an fp model online. Use a method whose "
            "backend supports it, or load an already quantized checkpoint."
        )

    def create_quantized_linear_shell(self, linear: torch.nn.Linear, compute_dtype: torch.dtype) -> torch.nn.Module:
        raise NotImplementedError(
            f"Backend `{self.name}` cannot load pre-quantized checkpoints. Use "
            "`load_prequantized=False` to quantize an fp model online instead."
        )

    def dequantize_to_linear(self, module: torch.nn.Module, compute_dtype: torch.dtype, compute_device=None, model_device=None) -> torch.nn.Linear:
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


def check_differentiable(module: torch.nn.Module, example_input: torch.Tensor = None, verbose: bool = True) -> bool:
    """
    Check whether gradients pass through `module` w.r.t. its input: run a real
    backward pass from the output (`torch.autograd.grad`) and verify a finite
    gradient arrives at the input. This is what LoRA training requires from
    frozen (e.g. quantized) layers. The module is cast to bfloat16 in place and
    probed with a bfloat16 input; if `example_input` is None, a random one is
    built for modules exposing `in_features`.

    Example (probing a torchao-quantized Linear):

        import torch
        from diffsynth.core.quant import check_differentiable
        from torchao.quantization import quantize_, Int8WeightOnlyConfig

        linear = torch.nn.Linear(1024, 1024, dtype=torch.bfloat16, device="cuda")
        quantize_(linear, Int8WeightOnlyConfig(version=2))
        check_differentiable(linear)
    """
    def report(result, detail):
        if verbose:
            print(f"check_differentiable ({type(module).__name__}): {'OK' if result else 'FAIL'} -- {detail}")
        return result

    try:
        module = module.to(torch.bfloat16)
        if example_input is None:
            if not hasattr(module, "in_features"):
                raise ValueError("`example_input` is required for modules without `in_features`.")
            device = next((t.device for t in list(module.parameters()) + list(module.buffers())), torch.device("cpu"))
            example_input = torch.randn(4, module.in_features, device=device)
        x = example_input.detach().to(torch.bfloat16).requires_grad_(True)
        y = module(x)
        if not y.requires_grad:
            return report(False, "the output does not require grad, so no autograd graph was recorded")
        input_grad = torch.autograd.grad(y, x, grad_outputs=torch.randn_like(y), allow_unused=True)[0]
    except Exception as error:
        return report(False, f"{type(error).__name__}: {error}")
    if input_grad is None:
        return report(False, "backward finished but no gradient reached the input")
    if not torch.isfinite(input_grad.float()).all():
        return report(False, "the input gradient contains non-finite values")
    return report(True, "gradients pass through the module to its input")
