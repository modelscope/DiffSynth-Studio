from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Optional
import torch
from .base import QUANT_BACKENDS


QUANT_METHODS = {}


@dataclass
class QuantMethodSpec:
    backend: str
    config_factory: Callable[[dict], Any]
    label: str = ""


def register_quant_method(name, backend, config_factory, label=""):
    QUANT_METHODS[name] = QuantMethodSpec(backend=backend, config_factory=config_factory, label=label)


def describe_quant_method(name):
    """Print a method's backend, label, and the accepted `backend_config_kwargs` with defaults."""
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


@dataclass
class QuantizeConfig:
    """
    Quantization config and entry points for any `nn.Module`.

    Online (dynamic) quantization -- quantize an fp model whose weights are loaded:

        cfg = QuantizeConfig(method="bitsandbytes_nf4")
        model.load_state_dict(fp_state_dict)
        cfg.quantize_model(model, compute_dtype=torch.bfloat16, device="cuda")

    Loading a pre-quantized checkpoint -- swap in quantized shells before loading:

        cfg = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True)
        cfg.prepare_for_prequantized_load(model, compute_dtype=torch.bfloat16)
        state_dict = cfg.unflatten_state_dict(state_dict, metadata)
        model.load_state_dict(state_dict, assign=True)

    Dequantize-once -- after either flow above, back to plain fp Linears that carry
    the quantization error:

        cfg.dequantize_model(model, compute_dtype=torch.bfloat16)

    Fields:
        method: name from QUANT_METHODS, determines backend + scheme + backend config.
        mode: "dynamic" keeps the backend-native quantized Linear, which dequantizes
            its weight on every forward; "dequant_once" dequantizes to a plain fp
            `nn.Linear` right after the weights are quantized or loaded.
        target_modules / exclude_modules: list of module names; a layer is matched if
            its full dotted name equals an entry, or ends with "." + entry
            (e.g. "img_mod.1" matches "transformer_blocks.0.img_mod.1").
        backend_config_kwargs: passed through to the method's backend config factory
            (e.g. nf4 blocksize); `describe_quant_method(name)` prints what a method
            accepts.
        load_prequantized: the checkpoint already holds quantized weights; load them
            directly instead of quantizing fp weights online.
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

    def quantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16, device=None):
        """
        Quantize the targeted `nn.Linear` layers of an fp model in place. Call AFTER
        `load_state_dict`. `device` is the target device of the quantized layers
        (`None` quantizes in place); the fp model may stay on another device.
        """
        self.backend.validate_environment()

        def quantize(linear):
            return self.backend.create_quantized_linear(linear, compute_dtype, device=device)

        replaced = self._replace_target_linears(model, quantize)
        print(f"{len(replaced)} nn.Linear layers quantized (method: {self.method}).")
        return model

    def dequantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """Replace every quantized Linear in the model by a plain fp `nn.Linear`."""
        self._dequantize_linears(model, compute_dtype)
        return model

    def prepare_for_prequantized_load(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Replace the targeted `nn.Linear` layers with empty quantized Linears matching a
        pre-quantized checkpoint. Call BEFORE `load_state_dict(assign=True)`.
        """
        self.backend.validate_environment()

        def build_shell(linear):
            return self.backend.create_quantized_linear_shell(linear, compute_dtype)

        replaced = self._replace_target_linears(model, build_shell)
        print(f"{len(replaced)} nn.Linear layers replaced for loading the pre-quantized checkpoint (method: {self.method}).")
        return model

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        """Rebuild composite quantized tensors from a flat (safetensors) state dict, for `load_state_dict(assign=True)`."""
        return self.backend.unflatten_state_dict(state_dict, metadata)

    def flatten_state_dict(self, state_dict: dict):
        """
        Inverse of `unflatten_state_dict`. Returns (state_dict, metadata), ready for
        `safetensors.torch.save_file(tensors, path, metadata=metadata)`.
        """
        if not self.backend.capabilities().get("is_serializable", False):
            raise NotImplementedError(
                f"Backend `{self.backend.name}` (method `{self.method}`) does not declare "
                'serialization support (`capabilities()["is_serializable"]`), so its quantized '
                "state dict cannot be flattened for saving."
            )
        tensors, metadata = self.backend.flatten_state_dict(state_dict)
        tensors = {key: value.contiguous() for key, value in tensors.items()}
        metadata = {"format": "pt", **{key: value if isinstance(value, str) else str(value) for key, value in metadata.items()}}
        return tensors, metadata

    @staticmethod
    def _name_matches(full_name, patterns):
        if patterns is None:
            return False
        if full_name in patterns:
            return True
        return any(full_name.endswith(f".{pattern}") for pattern in patterns)

    def _should_quantize(self, full_name, module):
        if not isinstance(module, torch.nn.Linear):
            return False
        if self.target_modules is not None and not self._name_matches(full_name, self.target_modules):
            return False
        if self._name_matches(full_name, self.exclude_modules):
            return False
        return True

    def _replace_target_linears(self, model, transform):
        replaced = []
        for full_name, module in list(model.named_modules()):
            if full_name == "" or not self._should_quantize(full_name, module):
                continue
            parent_name, _, leaf_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf_name, transform(module))
            replaced.append(full_name)
        return replaced

    def _dequantize_linears(self, model, compute_dtype):
        restored = []
        for full_name, module in list(model.named_modules()):
            if full_name == "" or not self.backend.is_quantized_linear(module):
                continue
            parent_name, _, leaf_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf_name, self.backend.dequantize_to_linear(module, compute_dtype))
            restored.append(full_name)
        return restored
