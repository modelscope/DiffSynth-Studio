import importlib.util

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_preset


@register_quant_backend("torchao")
class TorchaoQuantBackend(QuantBackend):
    """
    Thin adapter over torchao `quantize_` (weight-only configs).
    The quantized module keeps the `nn.Linear` class; only its weight is swapped
    to a torchao tensor subclass, whose dispatch performs dequant + matmul.
    Therefore identification relies on the weight type, not the module class.
    """

    def validate_environment(self, config):
        try:
            import torchao  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchao is required for torchao quantization presets. "
                'Please install it via `pip install torchao` or `pip install "diffsynth[quant]"`.'
            )

    def capabilities(self, config):
        return {
            "is_serializable": True,    # safetensors flatten requires torchao >= 0.15
            "is_trainable": False,
            "is_compileable": True,
            "requires_calibration": False,
        }

    def is_quantized_linear(self, module) -> bool:
        weight = getattr(module, "weight", None)
        return isinstance(module, torch.nn.Linear) and weight is not None and "torchao" in type(weight).__module__

    def quantize_linear_from_fp(self, linear, config, compute_dtype, device=None):
        from torchao.quantization import quantize_
        linear.requires_grad_(False)
        to_kwargs = {}
        if linear.weight.dtype != compute_dtype:
            to_kwargs["dtype"] = compute_dtype
        if device is not None and linear.weight.device != torch.device(device):
            # Move this single layer to the target device before quantizing, so only
            # one layer's fp copy transits through the GPU at a time.
            to_kwargs["device"] = device
        if to_kwargs:
            linear = linear.to(**to_kwargs)
        # In-place: swaps `linear.weight` to a torchao quantized tensor subclass.
        quantize_(linear, config)
        return linear

    def create_quantized_linear_for_load(self, in_features, out_features, bias, config):
        # The quantization lives in the weight tensor subclass, so the shell is a plain Linear.
        return torch.nn.Linear(in_features, out_features, bias=bias, device="meta")

    def flatten_state_dict(self, state_dict):
        self._require_safetensors_support()
        from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
        flattened = flatten_tensor_state_dict(state_dict)
        if isinstance(flattened, tuple):
            return flattened
        return flattened, {}

    def unflatten_state_dict(self, state_dict, metadata):
        self._require_safetensors_support()
        from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict
        from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao
        if not is_metadata_torchao(metadata):
            raise ValueError(
                "This checkpoint carries no torchao metadata (no `tensor_names` entry in the "
                "safetensors header), so its tensor subclasses cannot be rebuilt. It was most "
                "likely not saved by torchao."
            )
        rebuilt = unflatten_tensor_state_dict(state_dict, metadata)
        if isinstance(rebuilt, tuple):
            rebuilt = rebuilt[0]
        return rebuilt

    def _require_safetensors_support(self):
        from torchao import __version__ as torchao_version
        try:
            from torchao.prototype.safetensors.safetensors_support import (  # noqa: F401
                flatten_tensor_state_dict,
                unflatten_tensor_state_dict,
            )
        except ImportError:
            raise ImportError(
                f"Serializing torchao quantized weights to safetensors needs torchao >= 0.16 "
                f"(found {torchao_version}), which provides "
                "`torchao.prototype.safetensors`."
            )

    def dequantize_to_linear(self, module, compute_dtype):
        weight = module.weight
        fp_weight = weight.dequantize() if hasattr(weight, "dequantize") else weight.data
        fp_weight = fp_weight.to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear


def _int8_weight_only(overrides):
    from torchao.quantization import Int8WeightOnlyConfig
    overrides.setdefault("version", 2)
    return Int8WeightOnlyConfig(**overrides)


def _int4_weight_only(overrides):
    from torchao.quantization import Int4WeightOnlyConfig
    overrides.setdefault("group_size", 128)
    # torchao's default packing format (`plain`, like `preshuffled`) is backed by the
    # external `mslk` kernel library, which also appears to be limited to Hopper or newer.
    packing_format = overrides.get("int4_packing_format", "plain")
    if packing_format in ("plain", "preshuffled") and importlib.util.find_spec("mslk") is None:
        raise ImportError(
            f'The int4 packing format "{packing_format}" requires `mslk` '
            "(https://github.com/meta-pytorch/MSLK). Install it, or select a packing format "
            'shipped with torch, e.g. overrides={"int4_packing_format": "tile_packed_to_4d"} '
            "(runs on any CUDA device, but much slower on large-token workloads)."
        )
    return Int4WeightOnlyConfig(**overrides)


def _fp8_weight_only(overrides):
    from torchao.quantization import Float8WeightOnlyConfig
    return Float8WeightOnlyConfig(**overrides)


register_quant_preset("torchao_int8_w8a16", "torchao", _int8_weight_only, label="int8 weight-only")
register_quant_preset("torchao_int4_w4a16", "torchao", _int4_weight_only, label="int4 groupwise weight-only")
register_quant_preset("torchao_fp8_w8a16", "torchao", _fp8_weight_only, label="fp8 weight-only (storage only, no fp8 matmul hardware required)")
