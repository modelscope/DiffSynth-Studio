import importlib.util

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_method


@register_quant_backend("torchao")
class TorchaoQuantBackend(QuantBackend):
    """Adapter over torchao `quantize_` (weight-only configs); the quantization lives in the weight tensor subclass, not the module class."""

    def validate_environment(self):
        try:
            import torchao  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchao is required for torchao quantization methods. "
                'Please install it via `pip install torchao` or `pip install "diffsynth[quant]"`.'
            )

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": False,
            "is_compileable": True,
            "requires_calibration": False,
        }

    def is_quantized_linear(self, module) -> bool:
        weight = getattr(module, "weight", None)
        return isinstance(module, torch.nn.Linear) and weight is not None and "torchao" in type(weight).__module__

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        from torchao.quantization import quantize_
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        quantize_(linear, self.config)
        if model_device is not None:
            linear = linear.to(device=model_device)
        return linear

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return torch.nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta")

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

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.weight
        fp_weight = (weight.dequantize() if hasattr(weight, "dequantize") else weight.data).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)


def _int8_weight_only(backend_config_kwargs):
    from torchao.quantization import Int8WeightOnlyConfig
    backend_config_kwargs.setdefault("version", 2)
    return Int8WeightOnlyConfig(**backend_config_kwargs)


def _int4_weight_only(backend_config_kwargs):
    from torchao.quantization import Int4WeightOnlyConfig
    packing_format = backend_config_kwargs.get("int4_packing_format", "plain")
    if packing_format in ("plain", "preshuffled") and importlib.util.find_spec("mslk") is None:
        raise ImportError(
            f'The int4 packing format "{packing_format}" requires `mslk` '
            "(https://github.com/meta-pytorch/MSLK, install the build matching your torch version), "
            'or select a packing format shipped with torch, e.g. backend_config_kwargs={"int4_packing_format": "tile_packed_to_4d"} '
            "(runs on any CUDA device, but much slower on large-token workloads)."
        )
    return Int4WeightOnlyConfig(**backend_config_kwargs)


def _fp8_weight_only(backend_config_kwargs):
    from torchao.quantization import Float8WeightOnlyConfig
    return Float8WeightOnlyConfig(**backend_config_kwargs)


register_quant_method("torchao_int8_w8a16", "torchao", _int8_weight_only, label="8bit, int8, weight-only")
register_quant_method("torchao_int4_w4a16", "torchao", _int4_weight_only, label="4bit, int4, weight-only")
register_quant_method("torchao_fp8_w8a16", "torchao", _fp8_weight_only, label="8bit, fp8, weight-only")
