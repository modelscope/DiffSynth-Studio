from dataclasses import dataclass, field, fields
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
    from . import backends
    backends.load_all_backends()
    if name not in QUANT_METHODS:
        raise ValueError(f"Unknown quantization method: {name}. Available methods:\n{_available_methods()}")
    spec = QUANT_METHODS[name]
    lines = [f"method: {name}", f"backend: {spec.backend}", f"detail: {spec.label}"]
    try:
        config = spec.config_factory({})
    except Exception as error:
        lines.append(f"backend config: could not be built with default kwargs: {error}")
        print("\n".join(lines))
        return
    cls = type(config)
    lines.append(f"backend config: {cls.__module__}.{cls.__qualname__}")
    all_fields = fields(config)
    user_fields = [f for f in all_fields if f.init]
    pinned_fields = [f for f in all_fields if not f.init]
    if user_fields:
        lines.append("backend_config_kwargs (user-tunable):")
        width = max(len(f.name) for f in user_fields)
        lines += [f"  {f.name:<{width}} = {getattr(config, f.name)!r}" for f in user_fields]
    if pinned_fields:
        lines.append("pinned by method (not overridable):")
        lines += [f"  {f.name} = {getattr(config, f.name)!r}" for f in pinned_fields]
    print("\n".join(lines))


def _available_methods():
    from . import backends
    backends.load_all_backends()
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


def _format_error_report(reports):
    if not reports:
        return "Quantization error report: no targeted nn.Linear layers found."
    methods = sorted({report["method"] for report in reports})
    lines = [f"Quantization error report ({' + '.join(methods)}, {len(reports)} layers, sorted by relative_error, worst first):"]
    mixed = len(methods) > 1
    name_width = max(len(report["name"]) for report in reports)
    header = f"  {'relative_error':>14}  {'max_abs':>12}  {'shape':<16}  {'name':<{name_width}}"
    if mixed:
        header += "  method"
    lines.append(header)
    for report in reports:
        line = (
            f"  {report['relative_error'] * 100:>13.2f}%  {report['max_abs']:>12.3e}"
            f"  {str(report['shape']):<16}  {report['name']:<{name_width}}"
        )
        if mixed:
            line += f"  {report['method']}"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class QuantizeConfig:
    """
    Quantization config and entry points for any `nn.Module`.

    Online (dynamic) quantization -- quantize an fp model whose weights are loaded:

        cfg = QuantizeConfig(method="bitsandbytes_nf4")
        model.load_state_dict(fp_state_dict)
        cfg.quantize_model(model, compute_device="cuda")

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
        from . import backends
        backends.load_backend_for_method(self.method)
        if self.method not in QUANT_METHODS:
            raise ValueError(f"Unknown quantization method: {self.method}. Available methods:\n{_available_methods()}")
        spec = QUANT_METHODS[self.method]
        if spec.backend not in QUANT_BACKENDS:
            raise ValueError(f"Quantization backend `{spec.backend}` (required by method `{self.method}`) is not registered.")
        backend = QUANT_BACKENDS[spec.backend]()
        backend.validate_environment()
        backend.announce_environment()
        backend.config = spec.config_factory(dict(self.backend_config_kwargs))
        return backend

    def quantize_model(self, model: torch.nn.Module, compute_device=None, model_device=None):
        """
        Quantize the targeted `nn.Linear` layers of an fp model in place, keeping each
        layer's existing dtype. Call AFTER `load_state_dict`. Does nothing when
        `load_prequantized` is set, since such a checkpoint is already quantized.

        Parameters:
            model: the fp model whose targeted layers are replaced by quantized ones.
            compute_device: where quantization runs; `None` quantizes each layer in place.
            model_device: where each quantized layer is stored afterwards; `None` leaves it
                on `compute_device`. Combining an fp model on the CPU with
                `compute_device="cuda", model_device="cpu"` streams the work layer by
                layer, so the accelerator only ever holds one layer at a time.
        """
        if self.load_prequantized:
            return model

        def quantize(linear):
            return self.backend.create_quantized_linear(linear, compute_device=compute_device, model_device=model_device)

        replaced = self._replace_target_linears(model, quantize)
        print(f"{len(replaced)} nn.Linear layers quantized (method: {self.method}).")
        return model

    def dequantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16, compute_device=None, model_device=None):
        """
        Replace every quantized Linear in the model by a plain fp `nn.Linear`.
        Does nothing unless `mode` is `"dequant_once"`.

        Parameters:
            model: the quantized model to restore in place.
            compute_dtype: dtype of the restored fp weights.
            compute_device: where dequantization runs; `None` restores each layer in place.
            model_device: where each restored layer is stored afterwards; `None` leaves it
                on `compute_device`.
        """
        if self.mode != "dequant_once":
            return model
        self._dequantize_linears(model, compute_dtype, compute_device, model_device)
        return model

    def is_quantized_linear(self, module) -> bool:
        """Whether `module` is one of this config's backend-native quantized Linears."""
        return self.backend.is_quantized_linear(module)

    def build_quantized_shell(self, module, compute_dtype: torch.dtype, **kwargs):
        """
        Build an empty quantized Linear matching `module`, used to release a layer's
        weights while keeping it routable, and to stage a transient copy on the
        computation device.

        Parameters:
            module: the quantized layer whose shape and bias presence are mirrored.
            compute_dtype: dtype the shell dequantizes to at forward time.
        """
        return self.backend.create_quantized_linear_shell(module, compute_dtype)

    def prepare_for_prequantized_load(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Replace the targeted `nn.Linear` layers with empty quantized Linears matching a
        pre-quantized checkpoint. Call BEFORE `load_state_dict(assign=True)`.

        Parameters:
            model: the freshly constructed model whose targeted layers become shells.
            compute_dtype: dtype the quantized layers dequantize to at forward time.
        """
        def build_shell(linear):
            return self.backend.create_quantized_linear_shell(linear, compute_dtype)

        replaced = self._replace_target_linears(model, build_shell)
        print(f"{len(replaced)} nn.Linear layers replaced for loading the pre-quantized checkpoint (method: {self.method}).")
        return model

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        """
        Rebuild composite quantized tensors from a flat (safetensors) state dict, so the
        result can be given to `load_state_dict(assign=True)`.

        Parameters:
            state_dict: the flat tensors read from the checkpoint.
            metadata: the string-only metadata stored alongside them.
        """
        return self.backend.unflatten_state_dict(state_dict, metadata)

    def flatten_state_dict(self, state_dict: dict):
        """
        Flatten a quantized model's state dict into plain tensors and string-only metadata.
        Returns `(tensors, metadata)`, ready for
        `safetensors.torch.save_file(tensors, path, metadata=metadata)`.

        Parameters:
            state_dict: the quantized model's state dict, holding composite tensors.
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

    def measure_quantization_error(self, model: torch.nn.Module, compute_device="cuda", verbose=True) -> list:
        """
        Measure the weight error this config's quantization introduces.

        Parameters:
            model: the fp model to analyze.
            compute_device: where the quantize-dequantize round trip runs.
            verbose: print the per-layer report, sorted by `relative_error`, worst first.

        Returns:
            A list of per-layer dicts: `name`, `method`, `shape`, `relative_error` and `max_abs` (worst-element absolute error).
        """
        reports = [
            self._measure_layer_error(full_name, module, compute_device)
            for full_name, module in model.named_modules()
            if full_name != "" and self._should_quantize(full_name, module)
        ]
        reports.sort(key=lambda report: report["relative_error"], reverse=True)
        if verbose:
            print(_format_error_report(reports))
        return reports

    def _measure_layer_error(self, full_name, module, compute_device):
        fp_weight = module.weight.detach().to(device=compute_device, dtype=torch.float32)
        linear = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None, dtype=module.weight.dtype, device=compute_device,
        )
        linear.weight.data.copy_(module.weight.detach())
        if module.bias is not None:
            linear.bias.data.copy_(module.bias.detach().to(device=compute_device))
        quantized = self.backend.create_quantized_linear(linear, compute_device=compute_device, model_device=compute_device)
        restored = self.backend.dequantize_to_linear(quantized, compute_dtype=torch.float32, compute_device=compute_device, model_device=compute_device)
        quant_weight = restored.weight.detach()
        error = quant_weight - fp_weight
        report = {
            "name": full_name,
            "method": self.method,
            "shape": tuple(fp_weight.shape),
            "relative_error": (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(fp_weight)).item(),
            "max_abs": error.abs().max().item(),
        }
        del fp_weight, linear, quantized, restored, quant_weight, error
        return report

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
        if self.is_quantized_linear(module):
            return False
        if self.target_modules is not None and not self._name_matches(full_name, self.target_modules):
            return False
        if self._name_matches(full_name, self.exclude_modules):
            return False
        return True

    def _replace_target_linears(self, model, transform):
        replaced = [
            full_name
            for full_name, module in model.named_modules()
            if full_name != "" and self._should_quantize(full_name, module)
        ]
        for full_name in replaced:
            parent_name, _, leaf_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf_name, transform(getattr(parent, leaf_name)))
        return replaced

    def _dequantize_linears(self, model, compute_dtype, compute_device=None, model_device=None):
        restored = [
            full_name
            for full_name, module in model.named_modules()
            if full_name != "" and self.backend.is_quantized_linear(module)
        ]
        for full_name in restored:
            parent_name, _, leaf_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            restored_linear = self.backend.dequantize_to_linear(
                getattr(parent, leaf_name), compute_dtype, compute_device=compute_device, model_device=model_device
            )
            setattr(parent, leaf_name, restored_linear)
        return restored


@dataclass
class MixedQuantizeConfig:
    """
    Combines multiple `QuantizeConfig` passes, one per disjoint layer set, behind the
    same interface as a single `QuantizeConfig`. Mixed quantization assigns different
    methods to different layers, e.g. nf4 for attention/MLP and int8 for the
    quantization-sensitive modulation layers:

        mod_layers = ["img_mod.1", "txt_mod.1", ...]
        cfg = MixedQuantizeConfig(configs=[
            QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
            QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
        ])
        cfg.quantize_model(model, compute_device="cuda")

    The passes run in `configs` order. Their matched layer sets must be pairwise
    disjoint; `quantize_model` and `prepare_for_prequantized_load` verify this on the
    fp model and raise before touching it.

    Fields:
        configs: list of `QuantizeConfig`, one per layer set; all must share the same
            `mode`, and their `load_prequantized` must stay False.
        load_prequantized: set on this wrapper (not on the children) to load a mixed
            pre-quantized checkpoint.
    """
    configs: list = None
    load_prequantized: bool = False

    def __post_init__(self):
        if not self.configs:
            raise ValueError("`MixedQuantizeConfig.configs` requires at least one `QuantizeConfig`.")
        for config in self.configs:
            if not isinstance(config, QuantizeConfig):
                raise ValueError(f"`MixedQuantizeConfig.configs` should contain `QuantizeConfig` instances, but got `{type(config).__name__}`.")
            if config.load_prequantized:
                raise ValueError(
                    "Set `load_prequantized=True` on `MixedQuantizeConfig` itself, not on its "
                    "child configs; the children only describe method + layer set."
                )
        modes = {config.mode for config in self.configs}
        if len(modes) > 1:
            raise ValueError(f"All configs in `MixedQuantizeConfig` should share the same `mode`, but got {sorted(modes)}.")
        self._ownership = {}

    @property
    def method(self):
        return " + ".join(config.method for config in self.configs)

    @property
    def mode(self):
        return self.configs[0].mode

    def quantize_model(self, model: torch.nn.Module, compute_device=None, model_device=None):
        """
        Run every config's quantization pass in order, after verifying their layer sets
        are disjoint. Does nothing when `load_prequantized` is set.

        Parameters:
            model: the fp model whose targeted layers are replaced by quantized ones.
            compute_device: where quantization runs; `None` quantizes each layer in place.
            model_device: where each quantized layer is stored afterwards; `None` leaves it
                on `compute_device`.
        """
        if self.load_prequantized:
            return model
        self._build_ownership(model)
        for config in self.configs:
            config.quantize_model(model, compute_device=compute_device, model_device=model_device)
        return model

    def dequantize_model(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16, compute_device=None, model_device=None):
        """
        Replace every quantized Linear of every config's backend by a plain fp `nn.Linear`.
        Does nothing unless `mode` is `"dequant_once"`.

        Parameters:
            model: the quantized model to restore in place.
            compute_dtype: dtype of the restored fp weights.
            compute_device: where dequantization runs; `None` restores each layer in place.
            model_device: where each restored layer is stored afterwards; `None` leaves it
                on `compute_device`.
        """
        if self.mode != "dequant_once":
            return model
        for config in self.configs:
            config.dequantize_model(model, compute_dtype=compute_dtype, compute_device=compute_device, model_device=model_device)
        return model

    def is_quantized_linear(self, module) -> bool:
        """Whether `module` is a quantized Linear of any config's backend."""
        return any(config.is_quantized_linear(module) for config in self.configs)

    def build_quantized_shell(self, module, compute_dtype: torch.dtype, layer_name: str = None):
        """
        Build an empty quantized Linear for `module` using the config that owns it.

        Parameters:
            module: the quantized layer whose shape and bias presence are mirrored.
            compute_dtype: dtype the shell dequantizes to at forward time.
            layer_name: its dotted name inside the model; required to pick the right config
                when several configs share one backend, since their quantized Linears are
                then the same class.
        """
        return self._owning_config(module, layer_name).build_quantized_shell(module, compute_dtype)

    def _owning_config(self, module, layer_name=None):
        config = self._ownership.get(layer_name)
        if config is not None:
            return config
        for config in self.configs:
            if config.is_quantized_linear(module):
                return config
        raise ValueError(
            f"`{type(module).__name__}` is not a quantized Linear of any config in this "
            f"`MixedQuantizeConfig` (methods: {self.method})."
        )

    def prepare_for_prequantized_load(self, model: torch.nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
        """
        Replace each config's targeted Linears by that backend's shells, after verifying
        the layer sets are disjoint.

        Parameters:
            model: the freshly constructed model whose targeted layers become shells.
            compute_dtype: dtype the quantized layers dequantize to at forward time.
        """
        self._build_ownership(model)
        for config in self.configs:
            config.prepare_for_prequantized_load(model, compute_dtype=compute_dtype)
        return model

    def unflatten_state_dict(self, state_dict: dict, metadata: dict):
        """
        Rebuild composite quantized tensors from a flat (safetensors) state dict.
        """
        if not self._ownership:
            for config in self._distinct_backend_configs():
                state_dict = config.unflatten_state_dict(state_dict, metadata)
            return state_dict
        rebuilt = {}
        for config, keys in self._grouped_by_backend(state_dict):
            rebuilt.update(config.unflatten_state_dict(keys, metadata) if config else keys)
        return rebuilt

    def flatten_state_dict(self, state_dict: dict):
        """
        Flatten the mixed model's state dict into plain tensors and string-only metadata.
        Returns (state_dict, metadata), ready for `safetensors.torch.save_file(tensors,
        path, metadata=metadata)`.
        """
        for config in self.configs:
            if not config.backend.capabilities().get("is_serializable", False):
                raise NotImplementedError(
                    f"Backend `{config.backend.name}` (method `{config.method}`) does not declare "
                    'serialization support (`capabilities()["is_serializable"]`), so the mixed '
                    "state dict cannot be flattened for saving."
                )
        merged_metadata = {}
        if not self._ownership:
            flattened = state_dict
            for config in self._distinct_backend_configs():
                flattened, metadata = config.backend.flatten_state_dict(flattened)
                merged_metadata.update(metadata)
        else:
            flattened = {}
            for config, keys in self._grouped_by_backend(state_dict):
                if config is None:
                    flattened.update(keys)
                    continue
                tensors, metadata = config.backend.flatten_state_dict(keys)
                flattened.update(tensors)
                merged_metadata.update(metadata)
        tensors = {key: value.contiguous() for key, value in flattened.items()}
        metadata = {"format": "pt", **{key: value if isinstance(value, str) else str(value) for key, value in merged_metadata.items()}}
        return tensors, metadata

    def measure_quantization_error(self, model: torch.nn.Module, compute_device="cuda", verbose=True) -> list:
        """
        Measure the weight error of the mixed quantization.
        """
        reports = []
        for config in self.configs:
            reports.extend(config.measure_quantization_error(model, compute_device=compute_device, verbose=False))
        reports.sort(key=lambda report: report["relative_error"], reverse=True)
        if verbose:
            print(_format_error_report(reports))
        return reports

    def _distinct_backend_configs(self):
        seen = set()
        distinct = []
        for config in self.configs:
            if config.backend.name not in seen:
                seen.add(config.backend.name)
                distinct.append(config)
        return distinct

    def _grouped_by_backend(self, state_dict):
        groups = {}
        for key, value in state_dict.items():
            parts = key.split(".")
            config = next((owner for end in range(len(parts) - 1, 0, -1)
                           if (owner := self._ownership.get(".".join(parts[:end]))) is not None), None)
            groups.setdefault(config.backend.name if config else None, (config, {}))[1][key] = value
        return groups.values()

    def _build_ownership(self, model):
        ownership = {}
        for index, config in enumerate(self.configs):
            names = [name for name, module in model.named_modules() if name and config._should_quantize(name, module)]
            overlap = sorted(set(names) & ownership.keys())
            if overlap:
                owner = self.configs.index(ownership[overlap[0]])
                raise ValueError(
                    f"Configs {owner} (`{self.configs[owner].method}`) and {index} (`{config.method}`) "
                    f"in `MixedQuantizeConfig` both match {len(overlap)} layers (e.g. {', '.join(overlap[:5])}). "
                )
            ownership.update({name: config for name in names})
        self._ownership.update(ownership)
