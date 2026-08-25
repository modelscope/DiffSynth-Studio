from dataclasses import dataclass, field

import torch
from ..base import QuantBackend, BackendConfig, register_quant_backend
from ..config import register_quant_method

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


def _assign_params_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """`_load_from_state_dict` replacement for shells: assigns tensors as-is, since packed weights do not pass the standard shape check."""
    local_names = set()
    for name, current in list(self._parameters.items()) + list(self._buffers.items()):
        if current is None:
            continue
        local_names.add(name)
        key = prefix + name
        if key in state_dict:
            value = state_dict[key]
            if name in self._parameters:
                self._parameters[name] = value if isinstance(value, torch.nn.Parameter) else torch.nn.Parameter(value, requires_grad=False)
            else:
                self._buffers[name] = value
        elif strict:
            missing_keys.append(key)
    if strict:
        for key in state_dict:
            if key.startswith(prefix) and key[len(prefix):].split(".", 1)[0] not in local_names:
                unexpected_keys.append(key)


@register_quant_backend("bitsandbytes")
class BitsAndBytesQuantBackend(QuantBackend):
    """Adapter over `bitsandbytes.nn.Linear4bit` (NF4 / FP4, weight-only)."""

    project_url = "https://github.com/bitsandbytes-foundation/bitsandbytes"

    def validate_environment(self):
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError(
                "bitsandbytes is required for bitsandbytes quantization methods. "
                'Please install it via `pip install bitsandbytes` or `pip install "diffsynth[quant]"`.'
            )

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (bnb.nn.Linear4bit,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        """The `meta` shell avoids allocating an fp weight; bnb quantizes while `Params4bit` moves to `compute_device`."""
        cfg = self.config
        compute_dtype = linear.weight.dtype
        if compute_device is None:
            compute_device = linear.weight.device

        with torch.device("meta"):
            quant_linear = bnb.nn.Linear4bit(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=cfg.compress_statistics,
                quant_type=cfg.quant_type,
                quant_storage=cfg.quant_storage,
            )
        weight = bnb.nn.Params4bit(
            linear.weight.data.contiguous(),
            requires_grad=False,
            compress_statistics=cfg.compress_statistics,
            blocksize=cfg.blocksize,
            quant_type=cfg.quant_type,
            quant_storage=cfg.quant_storage,
        )
        quant_linear.weight = weight.to(compute_device)
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data.to(device=compute_device), requires_grad=False)
        return quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        with torch.device("meta"):
            shell = bnb.nn.Linear4bit(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=self.config.compress_statistics,
                quant_type=self.config.quant_type,
                quant_storage=self.config.quant_storage,
            )
        shell._load_from_state_dict = _assign_params_from_state_dict.__get__(shell)
        return shell

    def unflatten_state_dict(self, state_dict, metadata):
        rebuilt = {}
        quant_state = {}
        for key, value in state_dict.items():
            weight_key = key.rsplit(".weight.", 1)[0] + ".weight" if ".weight." in key else None
            if weight_key is not None:
                quant_state.setdefault(weight_key, {})[key] = value
            else:
                rebuilt[key] = value
        for weight_key, stats in quant_state.items():
            if weight_key not in rebuilt:
                raise ValueError(f"Found quant state for `{weight_key}` but no packed weight.")
            rebuilt[weight_key] = bnb.nn.Params4bit.from_prequantized(
                data=rebuilt[weight_key], quantized_stats=stats, requires_grad=False, device=rebuilt[weight_key].device,
            )
        return rebuilt

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.weight
        fp_weight = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        if model_device is not None:
            linear = linear.to(device=model_device)
        return linear


@dataclass
class BitsAndBytes4bitConfig(BackendConfig):
    """Shared 4bit knobs; `quant_type` is pinned by the per-method subclass."""

    compress_statistics: bool = True
    blocksize: int = None                                # None -> bnb default (64)
    quant_storage: torch.dtype = torch.uint8


@dataclass
class BitsAndBytesNF4Config(BitsAndBytes4bitConfig):
    quant_type: str = field(init=False, default="nf4")


@dataclass
class BitsAndBytesFP4Config(BitsAndBytes4bitConfig):
    quant_type: str = field(init=False, default="fp4")


register_quant_method("bitsandbytes_nf4", "bitsandbytes", BitsAndBytesNF4Config.from_kwargs, label="4bit, nf4, weight-only")
register_quant_method("bitsandbytes_fp4", "bitsandbytes", BitsAndBytesFP4Config.from_kwargs, label="4bit, fp4, weight-only")
