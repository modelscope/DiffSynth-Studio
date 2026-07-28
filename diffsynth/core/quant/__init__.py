from .base import QuantBackend, QUANT_BACKENDS, register_quant_backend
from .config import QuantizeConfig, PresetSpec, QUANT_PRESETS, register_quant_preset
from .api import quantize_model_weights, dequantize_model_weights, replace_linear_for_quantized_load, save_quantized_model
# Importing `backends` registers the built-in bnb / torchao adapters.
# The adapters import their libraries lazily, so this is safe even when
# bitsandbytes / torchao are not installed.
from . import backends
