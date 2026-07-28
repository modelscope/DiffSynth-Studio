from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional
from .base import QUANT_BACKENDS


# Global registry of quantization presets: {preset_name -> PresetSpec}.
# A preset binds a backend and its concrete quantization config, so users never
# combine `backend` and `scheme` manually (which could produce invalid pairs).
QUANT_PRESETS = {}


@dataclass
class PresetSpec:
    backend: str                            # key into QUANT_BACKENDS
    config_factory: Callable[[dict], Any]   # overrides -> backend-specific config
    label: str = ""


def register_quant_preset(name, backend, config_factory, label=""):
    QUANT_PRESETS[name] = PresetSpec(backend=backend, config_factory=config_factory, label=label)


@dataclass
class QuantizeConfig:
    """
    User-facing quantization config, attached to `ModelConfig(quantize=...)`.

    preset: name from QUANT_PRESETS, determines backend + scheme + backend config.
    mode: "dynamic" keeps the backend-native quantized Linear, which dequantizes its weight
          on every forward; "dequant_once" instead dequantizes to a plain fp `nn.Linear` as
          soon as the weights are quantized or loaded, so the model runs at full precision
          while carrying the quantization error once.
    target_modules / exclude_modules: list of module names, matched the way peft LoRA's
          `target_modules` list is -- exact dotted name, or dot-boundary suffix
          (e.g. "img_mod.1" matches "transformer_blocks.0.img_mod.1").
    overrides: advanced users may override preset backend kwargs (e.g. nf4 blocksize).
    load_prequantized: the checkpoint already holds quantized weights, so load them
          directly instead of quantizing fp weights online. The backend supplies the
          quantized Linear that matches the checkpoint's layout; a checkpoint with a
          different layout is supported by registering a small custom backend, as done for
          `diffsynth.models.ideogram4_dit`.
    """
    preset: str = None
    mode: str = "dynamic"                   # dynamic | dequant_once
    target_modules: Optional[list] = None
    exclude_modules: Optional[list] = None
    overrides: dict = field(default_factory=dict)
    load_prequantized: bool = False

    def merged_with_defaults(self, defaults: dict):
        """
        Overlay this config on top of a registry-provided base (the `quant_config` entry of a
        published quantized variant in MODEL_CONFIGS). Only the fields the user actually set
        win, so asking for e.g. a different `mode` keeps the published preset and its
        loading behaviour.
        """
        if defaults is None or len(defaults) == 0:
            return self
        blank = QuantizeConfig()
        merged = dict(defaults)
        for f in fields(self):
            value = getattr(self, f.name)
            if value != getattr(blank, f.name):
                merged[f.name] = value
        return QuantizeConfig(**merged)

    def resolve(self):
        if self.preset is None:
            raise ValueError("`QuantizeConfig.preset` is required.")
        if self.preset not in QUANT_PRESETS:
            raise ValueError(f"Unknown quantization preset: {self.preset}. Available presets: {sorted(QUANT_PRESETS)}.")
        for field_name in ("target_modules", "exclude_modules"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, list):
                raise ValueError(f"`QuantizeConfig.{field_name}` should be a list of module names, but got `{type(value).__name__}`.")
        spec = QUANT_PRESETS[self.preset]
        if spec.backend not in QUANT_BACKENDS:
            raise ValueError(f"Quantization backend `{spec.backend}` (required by preset `{self.preset}`) is not registered.")
        if self.mode not in ("dynamic", "dequant_once"):
            raise ValueError(f"`QuantizeConfig.mode` should be `dynamic` or `dequant_once`, but got `{self.mode}`.")
        backend = QUANT_BACKENDS[spec.backend]
        backend_config = spec.config_factory(dict(self.overrides))
        return backend, backend_config
