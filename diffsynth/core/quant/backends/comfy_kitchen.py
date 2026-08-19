import copy
import json
from dataclasses import dataclass, field

import torch
from ..base import QuantBackend, BackendConfig, register_quant_backend
from ..config import register_quant_method

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
    COMFY_KITCHEN_AVAILABLE = True
except ImportError:
    COMFY_KITCHEN_AVAILABLE = False


COMFY_QUANT_KEY = "comfy_quant"
SCALE_KEY = "weight_scale"


class ComfyKitchenLinear(torch.nn.Linear):
    """`nn.Linear` whose weight is a comfy-kitchen `QuantizedTensor`."""

    def __init__(self, in_features, out_features, bias, *, layout, compute_dtype):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        self.layout = layout
        self.compute_dtype = compute_dtype
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        return super().forward(x)

    def __deepcopy__(self, memo):
        clone = type(self)(
            self.in_features, self.out_features, bias=self.bias is not None,
            layout=self.layout, compute_dtype=self.compute_dtype,
        )
        weight = self.weight
        params = getattr(weight, "_params", None)
        if params is None:
            cloned_weight = weight.detach().clone()
        else:
            cloned_weight = QuantizedTensor(weight._qdata.clone(), weight._layout_cls, copy.deepcopy(params))
        clone.weight = torch.nn.Parameter(cloned_weight, requires_grad=False)
        if self.bias is not None:
            clone.bias = torch.nn.Parameter(self.bias.detach().clone(), requires_grad=False)
        memo[id(self)] = clone
        return clone


def _fp8_linear(x_2d, weight, bias, layout, input_scale):
    """Quantize a 2D activation per tensor, then let comfy-kitchen's dispatch pick the fp8 matmul."""
    q_x = QuantizedTensor.from_float(x_2d, layout, scale=input_scale)
    return torch.nn.functional.linear(q_x, weight, bias)


class _FP8LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, layout, input_scale):
        out = _fp8_linear(x.detach().reshape(-1, x.shape[-1]), weight, bias, layout, input_scale)
        ctx.save_for_backward(weight)
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = torch.mm(grad_2d, weight.dequantize().to(grad_2d.dtype)).reshape(ctx.x_shape)
        grad_bias = grad_2d.sum(dim=0) if ctx.needs_input_grad[2] else None
        return grad_x, None, grad_bias, None, None


class ComfyKitchenFP8Linear(ComfyKitchenLinear):

    def __init__(self, in_features, out_features, bias, *, layout, compute_dtype):
        super().__init__(in_features, out_features, bias, layout=layout, compute_dtype=compute_dtype)
        self.register_buffer("input_scale", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        scale = state_dict.pop(prefix + "input_scale", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if scale is not None:
            self.input_scale = scale

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        scale = self.input_scale
        if scale is not None:
            scale = scale.to(dtype=torch.float32, device=x.device)
        if x.requires_grad:
            return _FP8LinearFunction.apply(x, self.weight, self.bias, self.layout, scale)
        out = _fp8_linear(x.reshape(-1, x.shape[-1]), self.weight, self.bias, self.layout, scale)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def __deepcopy__(self, memo):
        clone = super().__deepcopy__(memo)
        if self.input_scale is not None:
            clone.input_scale = self.input_scale.detach().clone()
        return clone


class ComfyQuantFormat:
    """Everything specific to one weight format that ComfyUI writes, one hook per direction."""

    marker_name = None   # the `format` string ComfyUI writes into the marker
    layout_cls = None    # name of the comfy-kitchen layout class
    linear_cls = ComfyKitchenLinear   # the `nn.Linear` subclass that runs this format
    sibling_keys = ()    # per-layer tensors to drop, so `load_state_dict` never sees them

    def params_from_marker(self, marker, qdata, layer_name):
        """`Params` fields beyond `scale`, `orig_dtype` and `orig_shape`, read off the marker."""
        return {}

    def from_float_kwargs(self, in_features, config):
        """Keywords for `QuantizedTensor.from_float` when quantizing an fp weight online."""
        return {}

    def marker_from_params(self, params):
        """Marker fields to write back, so a re-saved checkpoint reloads identically."""
        return {}


class TensorWiseInt8Format(ComfyQuantFormat):

    marker_name = "int8_tensorwise"
    layout_cls = "TensorWiseINT8Layout"
    DEFAULT_GROUPSIZE = 256

    def params_from_marker(self, marker, qdata, layer_name):
        nested = marker.get("params", {})
        convrot = bool(marker.get("convrot", nested.get("convrot", False)))
        groupsize = int(marker.get("convrot_groupsize", nested.get("convrot_groupsize", self.DEFAULT_GROUPSIZE)))
        in_features = qdata.shape[1]
        if convrot and in_features % groupsize:
            raise ValueError(
                f"The marker of `{layer_name}` asks for convrot with groupsize {groupsize}, "
                f"which does not divide in_features={in_features}."
            )
        return {"is_weight": True, "convrot": convrot, "convrot_groupsize": groupsize}

    def from_float_kwargs(self, in_features, config):
        groupsize = config.convrot_groupsize
        kwargs = {"is_weight": True, "per_channel": config.per_channel, "convrot": False, "convrot_groupsize": groupsize}
        if not config.convrot:
            return kwargs
        if groupsize < 4 or groupsize & (groupsize - 1) or (groupsize.bit_length() - 1) % 2:
            raise ValueError(f"convrot_groupsize must be a power of four >= 4, got {groupsize}.")
        while groupsize >= 4 and in_features % groupsize:
            groupsize //= 4
        if groupsize >= 4:
            kwargs.update(convrot=True, convrot_groupsize=groupsize)
        return kwargs

    def marker_from_params(self, params):
        marker = {"convrot": bool(params.convrot)}
        if params.convrot:
            marker["convrot_groupsize"] = int(params.convrot_groupsize)
        return marker


class Float8E4M3Format(ComfyQuantFormat):

    marker_name = "float8_e4m3fn"
    layout_cls = "TensorCoreFP8Layout"
    linear_cls = ComfyKitchenFP8Linear


FORMATS = {fmt.marker_name: fmt for fmt in (TensorWiseInt8Format(), Float8E4M3Format())}


def _parse_marker(marker_tensor, layer_name):
    try:
        marker = json.loads(bytes(marker_tensor.to(torch.uint8).tolist()).decode())
    except Exception as error:
        raise ValueError(f"`{layer_name}.{COMFY_QUANT_KEY}` is not valid JSON: {error}")
    marker_name = marker.get("format")
    if marker_name is None:
        raise ValueError(f"`{layer_name}.{COMFY_QUANT_KEY}` has no `format` field.")
    if marker_name not in FORMATS:
        raise ValueError(f"`{layer_name}` uses format `{marker_name}`, not in {sorted(FORMATS)}.")
    return FORMATS[marker_name], marker


@register_quant_backend("comfy_kitchen")
class ComfyKitchenQuantBackend(QuantBackend):
    """Adapter over comfy-kitchen's `QuantizedTensor`, the quantized-weight format ComfyUI writes."""

    project_url = "https://github.com/Comfy-Org/comfy-kitchen"

    def validate_environment(self):
        if not COMFY_KITCHEN_AVAILABLE:
            raise ImportError(
                "comfy-kitchen is required for comfy_kitchen quantization methods. "
                'Please install it via `pip install comfy-kitchen` or `pip install "diffsynth[quant]"`.'
            )
        if torch.version.cuda is None or tuple(int(v) for v in torch.version.cuda.split(".")[:2]) < (13, 0):
            ck.registry.disable("cuda")

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return tuple(dict.fromkeys(fmt.linear_cls for fmt in FORMATS.values()))

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        quant_format = self._active_format()
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        weight = QuantizedTensor.from_float(
            linear.weight.data, quant_format.layout_cls,
            **quant_format.from_float_kwargs(linear.in_features, self.config),
        )
        quant_linear = self._build_linear(linear, linear.weight.dtype)
        quant_linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data, requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self._build_linear(linear, compute_dtype)

    def unflatten_state_dict(self, state_dict, metadata):
        rebuilt = dict(state_dict)
        suffix = f".{COMFY_QUANT_KEY}"
        for layer_name in [key[: -len(suffix)] for key in state_dict if key.endswith(suffix)]:
            quant_format, marker = _parse_marker(rebuilt.pop(layer_name + suffix), layer_name)
            qdata = rebuilt.get(f"{layer_name}.weight")
            scale = rebuilt.pop(f"{layer_name}.{SCALE_KEY}", None)
            for key, value in (("weight", qdata), (SCALE_KEY, scale)):
                if value is None:
                    raise ValueError(f"`{layer_name}` has a {COMFY_QUANT_KEY} marker but no `{key}`.")
            for key in quant_format.sibling_keys:
                rebuilt.pop(f"{layer_name}.{key}", None)
            params = get_layout_class(quant_format.layout_cls).Params(
                scale=scale,
                orig_dtype=self.config.orig_dtype,
                orig_shape=tuple(qdata.shape),
                **quant_format.params_from_marker(marker, qdata, layer_name),
            )
            rebuilt[f"{layer_name}.weight"] = QuantizedTensor(qdata, quant_format.layout_cls, params)
        return rebuilt

    def flatten_state_dict(self, state_dict):
        flattened = {}
        for key, value in state_dict.items():
            params = getattr(value, "_params", None)
            if not key.endswith(".weight") or params is None:
                flattened[key] = value
                continue
            layer_name = key[: -len(".weight")]
            quant_format = next((f for f in FORMATS.values() if f.layout_cls == value._layout_cls), None)
            if quant_format is None:
                raise ValueError(f"`{layer_name}` uses layout `{value._layout_cls}`, which no format claims.")
            flattened.update(value.state_dict(prefix=key))
            marker = {"format": quant_format.marker_name, **quant_format.marker_from_params(params)}
            flattened[f"{layer_name}.{COMFY_QUANT_KEY}"] = torch.frombuffer(
                bytearray(json.dumps(marker).encode()), dtype=torch.uint8
            )
        return flattened, {}

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = module.weight.dequantize().to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)

    def _active_format(self):
        marker_name = self.config.format
        if marker_name not in FORMATS:
            raise ValueError(f"Unsupported format `{marker_name}`, not in {sorted(FORMATS)}.")
        return FORMATS[marker_name]

    def _build_linear(self, linear, compute_dtype):
        quant_format = self._active_format()
        return quant_format.linear_cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            layout=quant_format.layout_cls,
            compute_dtype=compute_dtype,
        )


@dataclass
class ComfyKitchenInt8Config(BackendConfig):
    format: str = field(init=False, default=TensorWiseInt8Format.marker_name)   # pinned by the method
    per_channel: bool = True
    convrot: bool = True
    convrot_groupsize: int = TensorWiseInt8Format.DEFAULT_GROUPSIZE
    orig_dtype: torch.dtype = torch.bfloat16


@dataclass
class ComfyKitchenFp8Config(BackendConfig):
    format: str = field(init=False, default=Float8E4M3Format.marker_name)       # pinned by the method
    orig_dtype: torch.dtype = torch.bfloat16


register_quant_method("comfy_kitchen_int8_w8a8", "comfy_kitchen", ComfyKitchenInt8Config.from_kwargs, label="W8A8, int8 weight + int8 dynamic activation (ComfyUI int8_tensorwise)")
register_quant_method("comfy_kitchen_fp8_w8a8", "comfy_kitchen", ComfyKitchenFp8Config.from_kwargs, label="W8A8, fp8 E4M3 weight + fp8 activation, static input_scale when present (ComfyUI float8_e4m3fn)")
