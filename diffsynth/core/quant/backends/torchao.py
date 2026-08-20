import json

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    from torchao.quantization import (
        quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig, Float8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig, Float8DynamicActivationFloat8WeightConfig,
        Float8DynamicActivationInt4WeightConfig,
    )
    from torchao.prototype.mx_formats.inference_workflow import (
        MXDynamicActivationMXWeightConfig, NVFP4DynamicActivationNVFP4WeightConfig, NVFP4WeightOnlyConfig,
    )
    from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict, unflatten_tensor_state_dict
    from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


class TorchaoLinear(torch.nn.Linear):
    """Marker class for torchao-quantized Linears."""


@register_quant_backend("torchao")
class TorchaoQuantBackend(QuantBackend):
    """Adapter over torchao `quantize_` (weight-only configs); the quantization lives in the weight tensor subclass, not the module class."""

    project_url = "https://github.com/pytorch/ao"

    def validate_environment(self):
        if not TORCHAO_AVAILABLE:
            raise ImportError(
                "torchao is required for torchao quantization methods. "
                'Please install it via `pip install torchao` or `pip install "diffsynth[quant]"`.'
            )

    _DIFFERENTIABLE_CONFIGS = frozenset({
        "Int8WeightOnlyConfig",
        "Float8WeightOnlyConfig",
        "NVFP4WeightOnlyConfig",
    })

    def capabilities(self):
        config_name = type(self.config).__name__
        return {
            "is_serializable": True,
            "is_differentiable": config_name in self._DIFFERENTIABLE_CONFIGS,
            "is_compileable": True,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (TorchaoLinear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        quant_linear = TorchaoLinear(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta")
        quant_linear.weight = linear.weight
        quant_linear.bias = linear.bias
        quantize_(quant_linear, self.config)
        if model_device is not None:
            quant_linear = quant_linear.to(device=model_device)
        return quant_linear

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return TorchaoLinear(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta")

    def flatten_state_dict(self, state_dict):
        return flatten_tensor_state_dict(state_dict)

    def unflatten_state_dict(self, state_dict, metadata):
        if not is_metadata_torchao(metadata):
            raise ValueError("This checkpoint carries no torchao metadata.")
        tensor_names = json.loads(metadata["tensor_names"])
        root_names = [name for name in tensor_names if "." not in name]
        if root_names:
            metadata = {**metadata, "tensor_names": json.dumps([name for name in tensor_names if "." in name])}
        rebuilt, _ = unflatten_tensor_state_dict(state_dict, metadata)
        for name in root_names:
            if name in state_dict:
                rebuilt[name] = state_dict[name]
        return rebuilt

    @staticmethod
    def _to_plain_weight(weight, compute_dtype):
        try:
            return weight.dequantize().to(compute_dtype)
        except (NotImplementedError, AttributeError):
            eye = torch.eye(weight.shape[1], dtype=compute_dtype, device=weight.device)
            return torch.nn.functional.linear(eye, weight).t().contiguous()

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = self._to_plain_weight(module.weight, compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)


def _int8_weight_only(backend_config_kwargs):
    backend_config_kwargs.setdefault("version", 2)
    return Int8WeightOnlyConfig(**backend_config_kwargs)


def _int4_weight_only(backend_config_kwargs):
    return Int4WeightOnlyConfig(**backend_config_kwargs)


def _fp8_weight_only(backend_config_kwargs):
    return Float8WeightOnlyConfig(**backend_config_kwargs)


def _int8_dynamic_activation_int8_weight(backend_config_kwargs):
    backend_config_kwargs.setdefault("version", 2)
    return Int8DynamicActivationInt8WeightConfig(**backend_config_kwargs)


def _float8_dynamic_activation_float8_weight(backend_config_kwargs):
    return Float8DynamicActivationFloat8WeightConfig(**backend_config_kwargs)


def _float8_dynamic_activation_int4_weight(backend_config_kwargs):
    return Float8DynamicActivationInt4WeightConfig(**backend_config_kwargs)


def _mx_dynamic_activation_mx_weight(elem_dtype):
    def factory(backend_config_kwargs):
        dtype = getattr(torch, elem_dtype, None)
        if dtype is None:
            raise ImportError(f"This torch build has no `torch.{elem_dtype}`, required by the MX config.")
        kwargs = {"block_size": 32, "activation_dtype": dtype, "weight_dtype": dtype}
        kwargs.update(backend_config_kwargs)
        return MXDynamicActivationMXWeightConfig(**kwargs)
    return factory


def _nvfp4_dynamic_activation_nvfp4_weight(backend_config_kwargs):
    return NVFP4DynamicActivationNVFP4WeightConfig(**backend_config_kwargs)


def _nvfp4_weight_only(backend_config_kwargs):
    return NVFP4WeightOnlyConfig(**backend_config_kwargs)


register_quant_method("torchao_int8_w8a16", "torchao", _int8_weight_only, label="8bit, int8, weight-only")
register_quant_method("torchao_int4_w4a16", "torchao", _int4_weight_only, label="W4A16, int4 weight-only")
register_quant_method("torchao_fp8_w8a16", "torchao", _fp8_weight_only, label="8bit, fp8, weight-only")
register_quant_method("torchao_int8_w8a8", "torchao", _int8_dynamic_activation_int8_weight, label="W8A8, int8 weight + int8 dynamic activation")
register_quant_method("torchao_fp8_w8a8", "torchao", _float8_dynamic_activation_float8_weight, label="W8A8, fp8 weight + fp8 dynamic activation")
register_quant_method("torchao_int4_w4a8", "torchao", _float8_dynamic_activation_int4_weight, label="W4A8, int4 weight + fp8 dynamic activation")
register_quant_method("torchao_mxfp8_w8a8", "torchao", _mx_dynamic_activation_mx_weight("float8_e4m3fn"), label="W8A8, MXFP8 microscaling weight + activation")
register_quant_method("torchao_mxfp4_w4a4", "torchao", _mx_dynamic_activation_mx_weight("float4_e2m1fn_x2"), label="W4A4, MXFP4 microscaling weight + activation")
register_quant_method("torchao_nvfp4_w4a4", "torchao", _nvfp4_dynamic_activation_nvfp4_weight, label="W4A4, NVFP4 weight + activation")
register_quant_method("torchao_nvfp4_w4a16", "torchao", _nvfp4_weight_only, label="W4A16, NVFP4 weight-only")
