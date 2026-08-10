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
        """The Linear classes this backend produces. MUST be subclass of `torch.nn.Linear`."""
        raise NotImplementedError(
            f"Backend `{self.name}` must declare the Linear classes it produces."
        )

    def checkpoint_key_patterns(self) -> tuple:
        """Checkpoint entries one quantized Linear needs, relative to its dotted name."""
        return ("weight", "weight.", "bias")

    def is_quantized_linear(self, module) -> bool:
        return isinstance(module, self.quantized_linear_classes())

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


def resolve_checkpoint_keys(patterns, layer_name: str, available_keys) -> list:
    """Expand a backend's `checkpoint_key_patterns` into the absolute keys of one layer.

    `available_keys` is anything supporting `in` and iteration (a dict or a `DiskMap`).
    Exact patterns are probed with `in`, so a whole-file key index costs O(1) per pattern;
    only patterns ending in "." require scanning the index.
    """
    prefix = f"{layer_name}." if layer_name else ""
    nested_prefixes = tuple(prefix + pattern for pattern in patterns if pattern.endswith("."))
    keys = [prefix + pattern for pattern in patterns
            if not pattern.endswith(".") and prefix + pattern in available_keys]
    if nested_prefixes:
        keys += [key for key in available_keys if key.startswith(nested_prefixes)]
    return list(dict.fromkeys(keys))


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


def check_backend_contract(backend, in_features: int = 512, out_features: int = 512,
                           compute_dtype: torch.dtype = torch.bfloat16,
                           compute_device: str = "cuda", verbose: bool = True) -> bool:
    """
    Verify a backend satisfies the quantized-Linear contract: it declares its classes,
    both factory methods return instances of them, and every declared class subclasses
    `torch.nn.Linear` so LoRA target detection and VRAM management can see it. The
    checkpoint key patterns are checked against the keys the backend actually writes,
    since a pattern list that misses a scale makes disk offload load corrupt layers
    without raising.

    Example:

        from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS, check_backend_contract
        spec = QUANT_METHODS["bitsandbytes_nf4"]
        check_backend_contract(QUANT_BACKENDS[spec.backend](spec.config_factory({})))
    """
    failures = []

    def check(condition, detail):
        if not condition:
            failures.append(detail)
        if verbose:
            print(f"  [{'PASS' if condition else 'FAIL'}] {detail}")
        return condition

    if verbose:
        print(f"check_backend_contract ({backend.name}):")
    try:
        classes = backend.quantized_linear_classes()
    except NotImplementedError as error:
        if verbose:
            print(f"  [FAIL] quantized_linear_classes() is not implemented: {error}")
        return False
    if not check(len(classes) > 0, f"quantized_linear_classes() is non-empty: {[cls.__name__ for cls in classes]}"):
        return False
    for cls in classes:
        check(issubclass(cls, torch.nn.Linear), f"{cls.__name__} subclasses torch.nn.Linear")

    plain = torch.nn.Linear(in_features, out_features, bias=True, dtype=compute_dtype, device=compute_device)
    plain.requires_grad_(False)
    check(not backend.is_quantized_linear(plain), "a plain nn.Linear is not reported as quantized")

    try:
        shell = backend.create_quantized_linear_shell(plain, compute_dtype)
    except NotImplementedError:
        shell = None
        if verbose:
            print("  [SKIP] create_quantized_linear_shell() is unsupported by this backend")
    if shell is not None:
        check(isinstance(shell, classes), f"create_quantized_linear_shell() returns a declared class, got {type(shell).__name__}")
        check(backend.is_quantized_linear(shell), "the shell is recognized before load_state_dict (disk offload routing)")

    try:
        quantized = backend.create_quantized_linear(plain, compute_device=compute_device)
    except NotImplementedError:
        quantized = None
        if verbose:
            print("  [SKIP] create_quantized_linear() is unsupported by this backend")
    if quantized is not None:
        check(isinstance(quantized, classes), f"create_quantized_linear() returns a declared class, got {type(quantized).__name__}")

    patterns = backend.checkpoint_key_patterns()
    check(len(patterns) > 0, f"checkpoint_key_patterns() is non-empty: {list(patterns)}")
    saved = quantized if quantized is not None else shell
    if saved is None:
        if verbose:
            print("  [SKIP] neither factory method is supported, so the stored keys cannot be checked")
    else:
        state_dict = {f"proj.{key}": value for key, value in saved.state_dict().items()}
        if backend.capabilities().get("is_serializable", False):
            try:
                state_dict = backend.flatten_state_dict(state_dict)[0]
            except Exception as error:
                if verbose:
                    print(f"  [SKIP] flatten_state_dict() failed, falling back to the raw state dict keys: {type(error).__name__}: {error}")
        uncovered = sorted(set(state_dict) - set(resolve_checkpoint_keys(patterns, "proj", state_dict)))
        check(len(uncovered) == 0, f"checkpoint_key_patterns() covers every stored key; uncovered: {uncovered}")

    if verbose:
        print(f"  => {'OK' if not failures else str(len(failures)) + ' FAILED'}")
    return not failures
