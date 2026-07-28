from abc import ABC
import torch


# Global registry of quantization backends: {name -> QuantBackend instance}.
# The framework only depends on this registry and the abstract interface below.
# It never imports bitsandbytes / torchao directly.
QUANT_BACKENDS = {}


class DtypeGuardedLinear(torch.nn.Linear):
    """
    An `nn.Linear`-typed base for a custom backend's quantized Linear built from plain
    parameters / buffers. Subclasses register their own packed tensors; dtype casts
    (`to(dtype)` / `half()` / `float()`) never re-type the tensors listed in
    `dtype_guarded_tensor_names`, while device moves -- and every other tensor,
    e.g. the bias -- behave as in a regular module.

    Inheriting `nn.Linear` keeps contract guarantee (a) literal (`isinstance` checks see
    a Linear), matching bnb (subclass) and torchao (original class). One optional way to
    satisfy guarantee (b); backends whose packed storage is already immune (non-float
    dtypes, tensor subclasses) do not need it. See `Fp8Linear` in
    `diffsynth.models.ideogram4_dit` for a usage example.
    """

    # Names of the attributes holding packed weights / quant state.
    dtype_guarded_tensor_names: tuple = ()

    def __init__(self, in_features: int, out_features: int):
        # Deliberately skip `nn.Linear.__init__`: it would allocate and randomly
        # initialize an fp weight that the subclass replaces with packed tensors anyway.
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

    def _apply(self, fn, recurse=True):
        # All dtype/device conversions funnel through `_apply`; wrap `fn` so that a
        # conversion which would change a protected tensor's dtype is redone as a
        # device-only move.
        protected = {id(tensor) for name in self.dtype_guarded_tensor_names
                     if (tensor := getattr(self, name, None)) is not None}

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)


class QuantBackend(ABC):
    """
    Adapter between the framework and a quantization library (bnb / torchao / custom).

    A backend only provides single-Linear-granularity construction / conversion.
    The traversal & replacement over a whole model is owned by the framework passes
    in `diffsynth.core.quant.api`.

    The quantized Linear produced by a backend must satisfy these black-box guarantees:
    (a) It is an `nn.Linear` drop-in: `forward(x)` internally performs dequant + matmul.
    (b) `.to(...)` moves devices but never re-types the packed weight / quant state:
        a dtype cast (`.to(dtype)` / `.half()` / `.float()`) must leave their storage
        format and values intact. Non-float packed storage (bnb uint8) satisfies this
        structurally; tensor subclasses intercept the cast (torchao); quantized Linears
        built from plain parameters/buffers may inherit `DtypeGuardedLinear` to get it.
    (c) `state_dict()` / `load_state_dict(assign=True)` round-trips (optionally via
        `flatten_state_dict` / `unflatten_state_dict`).
    (d) (Training branch only) `forward` is differentiable w.r.t. its input.
    """

    name: str = ""

    def capabilities(self, config) -> dict:
        # Report what this backend can do under the given scheme, so the framework can refuse
        # an unsupported request up front instead of failing mid-run. The flags depend on the
        # concrete `config`, and are checked before saving, training and compiling.
        return {
            "is_serializable": True,
            "is_trainable": False,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def validate_environment(self, config):
        # Check that whatever this scheme needs -- the backend library, its version, the GPU
        # architecture -- is actually available, and raise an actionable error if not. Called
        # before any weight is touched, so an unusable setup is reported immediately rather
        # than after the cost of loading or quantizing has been paid.
        return

    def quantize_linear_from_fp(self, linear: torch.nn.Linear, config, compute_dtype: torch.dtype, device=None) -> torch.nn.Module:
        # Quantize one Linear whose fp weights are already loaded, and return the module
        # that replaces it. This is the online quantization path: values are transformed,
        # not just re-shaped. `device` is where the quantized layer should end up; the fp
        # source may still live on CPU, so only one layer's fp copy transits through the GPU
        # at a time.
        raise NotImplementedError(
            f"Backend `{self.name}` cannot quantize an fp model online. Use a preset whose "
            "backend supports it, or load an already quantized checkpoint."
        )

    def create_quantized_linear_for_load(self, in_features: int, out_features: int, bias: bool, config) -> torch.nn.Module:
        # Build an empty quantized Linear whose state dict keys and shapes match a
        # pre-quantized checkpoint, and return it. Called before `load_state_dict` when
        # reading such a checkpoint: this pass only puts the right structure in place, and
        # `load_state_dict(assign=True)` then fills in the packed weights and quant state.
        raise NotImplementedError(
            f"Backend `{self.name}` cannot load pre-quantized checkpoints. Use "
            "`load_prequantized=False` to quantize an fp model online instead."
        )

    def dequantize_to_linear(self, module: torch.nn.Module, compute_dtype: torch.dtype) -> torch.nn.Linear:
        # Reconstruct a plain fp `nn.Linear` from a quantized one, so the model runs at full
        # precision afterwards while carrying the quantization error it already went through.
        # This backs `mode="dequant_once"`.
        raise NotImplementedError(
            f"Backend `{self.name}` cannot dequantize back to `nn.Linear`, so "
            '`mode="dequant_once"` is unavailable.'
        )

    def quantized_linear_classes(self) -> tuple:
        # Report the classes of this backend's quantized Linear, which is how the framework
        # recognizes an already quantized layer (VRAM wrapping, dequantization, saving).
        # Backends whose quantized module keeps the `nn.Linear` class (e.g. torchao) cannot
        # be identified this way and override `is_quantized_linear` instead.
        return ()

    def is_quantized_linear(self, module) -> bool:
        # Decide whether a module is one of this backend's quantized Linears.
        classes = self.quantized_linear_classes()
        return len(classes) > 0 and isinstance(module, classes)

    # ---- Serialization adapters (default no-op; only tensor-subclass backends need these) ----
    def flatten_state_dict(self, state_dict: dict):
        # Turn a state dict that holds composite quantized tensors into plain tensors plus
        # the metadata needed to rebuild them, so it can be written to safetensors.
        return state_dict, {}

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        # Inverse of `flatten_state_dict`: rebuild the composite quantized tensors from the
        # plain tensors and their metadata.
        return state_dict


def register_quant_backend(name):
    # Built-in and user-defined backends are registered through the same path.
    def decorator(cls):
        cls.name = name
        QUANT_BACKENDS[name] = cls()
        return cls
    return decorator
