from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Optional
import torch
from .base import QUANT_BACKENDS


# Global registry of quantization methods: {method_name -> QuantMethodSpec}.
# A method binds a backend and its concrete quantization config, so users never
# combine `backend` and `scheme` manually (which could produce invalid pairs).
QUANT_METHODS = {}


@dataclass
class QuantMethodSpec:
    backend: str                            # key into QUANT_BACKENDS
    config_factory: Callable[[dict], Any]   # backend_config_kwargs -> backend-specific config
    label: str = ""


def register_quant_method(name, backend, config_factory, label=""):
    QUANT_METHODS[name] = QuantMethodSpec(backend=backend, config_factory=config_factory, label=label)


def describe_quant_method(name):
    """
    Print one method's registration info: its backend, what it does, and what its
    `backend_config_kwargs` feed. The backend config is built with the default kwargs
    and introspected, so the printout shows its concrete type and per-field defaults;
    a dataclass config (e.g. torchao's) accepts any of its constructor kwargs.
    """
    if name not in QUANT_METHODS:
        raise ValueError(f"Unknown quantization method: {name}. Available methods:\n{_available_methods()}")
    spec = QUANT_METHODS[name]
    lines = [f"method: {name}", f"backend: {spec.backend}", f"detail: {spec.label}"]
    try:
        config = spec.config_factory({})
    except Exception as error:
        lines.append(f"backend config: could not be built with default kwargs: {error}")
    else:
        if is_dataclass(config):
            cls = type(config)
            lines.append(f"backend config: {cls.__module__}.{cls.__qualname__}")
            lines.append("backend_config_kwargs: any constructor kwarg of that class; defaults:")
            width = max(len(f.name) for f in fields(config))
            lines += [f"  {f.name:<{width}} = {getattr(config, f.name)!r}" for f in fields(config)]
        elif isinstance(config, dict) and config:
            lines.append("backend_config_kwargs: merged into the config dict below (defaults shown):")
            width = max(len(key) for key in config)
            lines += [f"  {key:<{width}} = {value!r}" for key, value in config.items()]
        else:
            lines.append("backend_config_kwargs: (none)")
    print("\n".join(lines))


def _available_methods():
    # Boxed, column-aligned list of the registered methods (same style as the
    # model downloader tips): name, its backend and its human-readable detail.
    methods = sorted(QUANT_METHODS.items())
    if len(methods) == 0:
        return "  (no method registered)"
    name_width = max(len(name) for name, _ in methods)
    backend_width = max(len(spec.backend) for _, spec in methods)
    lines = [f"method: {name:<{name_width}}  backend: {spec.backend:<{backend_width}}  detail: {spec.label}"
             for name, spec in methods]
    width = max(len(line) for line in lines)
    body = "\n".join(f"│ {line:<{width}} │" for line in lines)
    return f"┌{'─' * (width + 2)}┐\n{body}\n└{'─' * (width + 2)}┘"


def _name_matches(full_name, patterns):
    if patterns is None:
        return False
    if full_name in patterns:
        return True
    return any(full_name.endswith(f".{pattern}") for pattern in patterns)


def _should_quantize(full_name, module, quantize_config):
    if not isinstance(module, torch.nn.Linear):
        return False
    if quantize_config.target_modules is not None and not _name_matches(full_name, quantize_config.target_modules):
        return False
    if _name_matches(full_name, quantize_config.exclude_modules):
        return False
    return True


def _replace_target_linears(model, quantize_config, transform):
    # Snapshot via `list(...)` so replacements do not disturb the iteration.
    # Returns the full names of the replaced Linears.
    replaced = []
    for full_name, module in list(model.named_modules()):
        if full_name == "" or not _should_quantize(full_name, module, quantize_config):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf_name, transform(module))
        replaced.append(full_name)
    return replaced


def _dequantize_linears(model, backend, compute_dtype):
    # Returns the full names of the restored Linears.
    restored = []
    for full_name, module in list(model.named_modules()):
        # Only the backend can tell: torchao keeps `nn.Linear`, bnb subclasses it.
        if full_name == "" or not backend.is_quantized_linear(module):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf_name, backend.dequantize_to_linear(module, compute_dtype))
        restored.append(full_name)
    return restored


@dataclass
class QuantizeConfig:
    """
    User-facing quantization config, attached to `ModelConfig(quantize=...)`.

    method: name from QUANT_METHODS, determines backend + scheme + backend config.
        Resolved at construction: `self.backend` holds a ready-to-use `QuantBackend`
        instance that owns its backend-specific config.
    mode: "dynamic" keeps the backend-native quantized Linear, which dequantizes its weight
          on every forward; "dequant_once" instead dequantizes to a plain fp `nn.Linear` as
          soon as the weights are quantized or loaded, so the model runs at full precision
          while carrying the quantization error once.
    target_modules / exclude_modules: list of module names, matched the way peft LoRA's
          `target_modules` list is -- exact dotted name, or dot-boundary suffix
          (e.g. "img_mod.1" matches "transformer_blocks.0.img_mod.1").
    backend_config_kwargs: advanced users may pass extra kwargs of the method's backend
          config (e.g. nf4 blocksize); `describe_quant_method(name)` prints what a
          method accepts.
    load_prequantized: the checkpoint already holds quantized weights, so load them
          directly instead of quantizing fp weights online. The backend supplies the
          quantized Linear that matches the checkpoint's layout; a checkpoint with a
          different layout is supported by registering a small custom backend, as done for
          `diffsynth.models.ideogram4_dit`.
    """
    method: str = None
    mode: str = "dynamic"
    target_modules: Optional[list] = None
    exclude_modules: Optional[list] = None
    backend_config_kwargs: dict = field(default_factory=dict)
    load_prequantized: bool = False

    def __post_init__(self):
        if self.method is None:
            raise ValueError(f"`QuantizeConfig.method` is required. Available methods:\n{_available_methods()}")
        if self.mode not in ("dynamic", "dequant_once"):
            raise ValueError(f"`QuantizeConfig.mode` should be `dynamic` or `dequant_once`, but got `{self.mode}`.")
        for field_name in ("target_modules", "exclude_modules"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, list):
                raise ValueError(f"`QuantizeConfig.{field_name}` should be a list of module names, but got `{type(value).__name__}`.")
        self.backend = self._build_backend()

    def _build_backend(self):
        if self.method not in QUANT_METHODS:
            raise ValueError(f"Unknown quantization method: {self.method}. Available methods:\n{_available_methods()}")
        spec = QUANT_METHODS[self.method]
        if spec.backend not in QUANT_BACKENDS:
            raise ValueError(f"Quantization backend `{spec.backend}` (required by method `{self.method}`) is not registered.")
        return QUANT_BACKENDS[spec.backend](spec.config_factory(dict(self.backend_config_kwargs)))

    # The methods below are the only quantization entry points for external code
    # (loader, user scripts); `self.backend` stays an internal detail.

    def quantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16, device=None):
        """
        Quantize the target `nn.Linear` layers of an fp model in place (call AFTER `load_state_dict`).

        model: the fp model to quantize.
        compute_dtype: computation dtype of the quantized layers.
        device: target device of the quantized layers; the fp model may stay on CPU so that
            each layer transits through the device one by one. `None` quantizes in place.
        """
        self.backend.validate_environment()

        def quantize(linear):
            return self.backend.create_quantized_linear(linear, compute_dtype, device=device)

        replaced = _replace_target_linears(model, self, quantize)
        print(f"{len(replaced)} nn.Linear layers quantized (method: {self.method}).")
        return model

    def dequantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Replace every quantized Linear in the model by a plain fp `nn.Linear`.

        model: a model holding quantized Linears (from online quantization or a checkpoint).
        compute_dtype: dtype of the restored fp weights.
        """
        _dequantize_linears(model, self.backend, compute_dtype)
        return model

    def prepare_for_prequantized_load(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Replace the target `nn.Linear` layers by empty quantized Linears matching a
        pre-quantized checkpoint (call BEFORE `load_state_dict(assign=True)`).

        model: the freshly constructed fp model.
        compute_dtype: computation dtype of the quantized layers.
        """
        self.backend.validate_environment()

        def build_shell(linear):
            return self.backend.create_quantized_linear_shell(linear, compute_dtype)

        replaced = _replace_target_linears(model, self, build_shell)
        print(f"{len(replaced)} nn.Linear layers replaced for loading the pre-quantized checkpoint (method: {self.method}).")
        return model

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        """
        Rebuild the backend's composite quantized tensors from a flat (safetensors)
        state dict, ready for `load_state_dict(assign=True)` after `prepare_for_prequantized_load`.
        """
        return self.backend.unflatten_state_dict(state_dict, metadata)

    def flatten_state_dict(self, state_dict: dict):
        """
        Inverse of `unflatten_state_dict`: flatten a quantized model's state dict into
        plain tensors + string metadata in the backend's own serialization layout, ready
        for `safetensors.torch.save_file(tensors, path, metadata=metadata)` or any custom
        writer (sharding, other containers).

        state_dict: the quantized model's state dict (`model.state_dict()`).
        Returns (state_dict, metadata), both safetensors-ready.
        """
        tensors, metadata = self.backend.flatten_state_dict(state_dict)
        tensors = {key: value.contiguous() for key, value in tensors.items()}
        # safetensors headers hold strings only.
        metadata = {"format": "pt", **{key: value if isinstance(value, str) else str(value) for key, value in metadata.items()}}
        return tensors, metadata
