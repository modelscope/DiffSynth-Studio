from dataclasses import dataclass

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import entropack as ep

    _REQUIRED_ENTROPACK_API = (
        ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear,
        ep.DFloat11Config, ep.TileANSConfig, ep.LatticeRANSConfig, ep.RawConfig,
    )
    ENTROPACK_AVAILABLE = True
    _ENTROPACK_IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    ep = None
    ENTROPACK_AVAILABLE = False
    _ENTROPACK_IMPORT_ERROR = error


if ENTROPACK_AVAILABLE:

    @dataclass
    class EntroPackDFloat11Config(BackendConfig, ep.DFloat11Config):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackTileANSConfig(BackendConfig, ep.TileANSConfig):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackLossyQuantConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackFP8CodedConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedFP8Linear

    @dataclass
    class EntroPackFP8RawConfig(BackendConfig, ep.RawConfig):
        linear_cls = ep.CompressedFP8Linear

    @dataclass
    class EntroPackINT8CodedConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedINT8Linear

    @dataclass
    class EntroPackINT8RawConfig(BackendConfig, ep.RawConfig):
        linear_cls = ep.CompressedINT8Linear


@register_quant_backend("entropack")
class EntroPackQuantBackend(QuantBackend):
    """Adapter over entropack's `CompressedLinear` family; a method's config is the config its layers get."""

    project_url = "https://github.com/modelscope/entropack"

    def validate_environment(self):
        if not ENTROPACK_AVAILABLE:
            raise ImportError(
                "entropack with the CompressedLinear / CompressedFP8Linear / CompressedINT8Linear "
                "classes is required for this quantization method. Install it via "
                '`pip install "entropack[cuda13]"` (use `cuda12` for a CUDA 12 environment) '
                'or `pip install "diffsynth[quant]"`.'
            ) from _ENTROPACK_IMPORT_ERROR

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear)

    def unflatten_state_dict(self, state_dict, metadata):
        return state_dict

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        quantized = self.config.linear_cls.from_linear(linear, config=self.config)
        return quantized if model_device is None else quantized.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self.config.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            dtype=compute_dtype, config=self.config,
        )

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.dequantize(compute_device).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(
                module.bias.data.to(dtype=compute_dtype, device=weight.device), requires_grad=False,
            )
        return linear if model_device is None else linear.to(device=model_device)


def _df11_config(kwargs):
    return EntroPackDFloat11Config.from_kwargs(kwargs)


def _tile_ans_config(kwargs):
    return EntroPackTileANSConfig.from_kwargs(kwargs)


def _lossy_config(kwargs):
    return EntroPackLossyQuantConfig.from_kwargs(kwargs)


def _fp8_coded_config(kwargs):
    return EntroPackFP8CodedConfig.from_kwargs(kwargs)


def _fp8_raw_config(kwargs):
    return EntroPackFP8RawConfig.from_kwargs(kwargs)


def _int8_coded_config(kwargs):
    return EntroPackINT8CodedConfig.from_kwargs(kwargs)


def _int8_raw_config(kwargs):
    return EntroPackINT8RawConfig.from_kwargs(kwargs)


register_quant_method("entropack_lossless_df11", "entropack", _df11_config, label="lossless, dfloat11")
register_quant_method("entropack_lossless_tile_ans", "entropack", _tile_ans_config, label="lossless, tile_ans")
register_quant_method("entropack_lossy_quant", "entropack", _lossy_config, label="lossy, lattice rate")
register_quant_method("entropack_lossy_quant_fp8", "entropack", _fp8_coded_config, label="W8A8, fp8, lattice-coded")
register_quant_method("entropack_lossy_quant_fp8_raw", "entropack", _fp8_raw_config, label="W8A8, fp8, codes verbatim")
register_quant_method("entropack_lossy_quant_int8", "entropack", _int8_coded_config, label="W8A8, int8, lattice-coded")
register_quant_method("entropack_lossy_quant_int8_raw", "entropack", _int8_raw_config, label="W8A8, int8, codes verbatim")
