import copy
import functools
import importlib.util
import json

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_method


COMFY_QUANT_KEY = "comfy_quant"
SCALE_KEY = "weight_scale"


@functools.cache
def _initialize_registry():
    """Disable comfy-kitchen's cuda backend below CUDA 13, where it reports available but fails."""
    import comfy_kitchen as ck
    if torch.version.cuda is None or tuple(int(v) for v in torch.version.cuda.split(".")[:2]) < (13, 0):
        ck.registry.disable("cuda")


class ComfyQuantFormat:
    """Everything specific to one weight format that ComfyUI writes."""

    marker_name = None      # the `format` string ComfyUI writes into the marker
    layout_cls = None       # name of the comfy-kitchen layout class
    extra_tensor_keys = ()  # sibling tensors consumed by unflatten (popped from state dict)

    def options_from_marker(self, marker, layer_name):
        """Per-layer settings taken from the decoded marker."""
        return {}

    def check_options(self, options, qdata, layer_name):
        """Reject settings that contradict the packed weight, before they corrupt it silently."""

    def options_for_new_weight(self, in_features, config):
        """The same settings, chosen from `config` when quantizing an fp weight online."""
        return {}

    def marker_from_params(self, params):
        """Settings to write back, so a re-saved checkpoint reloads identically."""
        return {}

    def logical_shape(self, qdata, options):
        """Shape of the dequantized weight. Formats that pack sub-byte values must override."""
        return tuple(qdata.shape)

    def params_kwargs(self, options, tensors):
        """`Params` fields beyond `scale`, `orig_dtype` and `orig_shape`."""
        return {}

    def from_float_kwargs(self, options, config):
        """Extra keywords for `QuantizedTensor.from_float`."""
        return {}


class TensorWiseInt8Format(ComfyQuantFormat):
    """int8 W8A8. The Hadamard rotation is a per-layer knob, so one file mixes group sizes."""

    marker_name = "int8_tensorwise"
    layout_cls = "TensorWiseINT8Layout"
    DEFAULT_GROUPSIZE = 256

    def options_from_marker(self, marker, layer_name):
        nested = marker.get("params", {})
        return {
            "convrot": bool(marker.get("convrot", nested.get("convrot", False))),
            "convrot_groupsize": int(marker.get(
                "convrot_groupsize", nested.get("convrot_groupsize", self.DEFAULT_GROUPSIZE))),
        }

    def check_options(self, options, qdata, layer_name):
        in_features = qdata.shape[1]
        if options["convrot"] and in_features % options["convrot_groupsize"] != 0:
            raise ValueError(
                f"The marker of `{layer_name}` asks for convrot with groupsize {options['convrot_groupsize']}, "
                f"which does not divide in_features={in_features}."
            )

    def options_for_new_weight(self, in_features, config):
        convrot, groupsize = self._resolve_rotation(in_features, config["convrot"], config["convrot_groupsize"])
        return {"convrot": convrot, "convrot_groupsize": groupsize}

    @staticmethod
    def _is_regular_hadamard_size(size: int) -> bool:
        """comfy-kitchen builds the rotation as Kronecker powers of a 4x4 Hadamard matrix."""
        return size >= 4 and size & (size - 1) == 0 and (size.bit_length() - 1) % 2 == 0

    @classmethod
    def _resolve_rotation(cls, in_features, convrot, groupsize):
        """Shrink the group size by four until it divides `in_features`, mirroring ComfyUI."""
        if not convrot:
            return False, groupsize
        if not cls._is_regular_hadamard_size(groupsize):
            raise ValueError(f"convrot_groupsize must be a power of four >= 4, got {groupsize}.")
        candidate = groupsize
        while candidate >= 4:
            if in_features % candidate == 0:
                return True, candidate
            candidate //= 4
        return False, groupsize

    def marker_from_params(self, params):
        marker = {"convrot": bool(params.convrot)}
        if params.convrot:
            marker["convrot_groupsize"] = int(params.convrot_groupsize)
        return marker

    def params_kwargs(self, options, tensors):
        return {"is_weight": True, **options}

    def from_float_kwargs(self, options, config):
        return {"is_weight": True, "per_channel": config["per_channel"], **options}


class Float8E4M3Format(ComfyQuantFormat):
    """FP8 per-tensor scaled. Weight is F8_E4M3, with scalar weight_scale and optional input_scale.

    TODO: fp8 activation quantization is not yet implemented. ComfyUI quantizes the input with
    input_scale before calling scaled_mm (fp8×fp8 on H100/4090). We currently fallback to
    dequant → bf16 matmul, which is correct but does not leverage fp8 tensor cores.
    """

    marker_name = "float8_e4m3fn"
    layout_cls = "TensorCoreFP8Layout"
    extra_tensor_keys = ("input_scale",)
    extra_tensor_optional = True


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


class ComfyKitchenLinear(torch.nn.Linear):
    """`nn.Linear` whose weight is a comfy-kitchen `QuantizedTensor`."""

    def __init__(self, in_features, out_features, bias, *, layout, compute_dtype, force_eager):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        self.layout = layout
        self.compute_dtype = compute_dtype
        self.force_eager = force_eager
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        # int8 activation quantization has no strided kernel; a non-contiguous input measured 2.5x slower.
        if not x.is_contiguous():
            x = x.contiguous()
        if not self.force_eager:
            return super().forward(x)
        import comfy_kitchen as ck
        with ck.use_backend("eager"):
            return super().forward(x)

    def extra_repr(self):
        params = getattr(self.weight, "_params", None)
        if params is None:
            return f"{super().extra_repr()}, layout={self.layout}"
        return (f"{super().extra_repr()}, layout={self.layout}, "
                f"convrot={getattr(params, 'convrot', None)}, "
                f"groupsize={getattr(params, 'convrot_groupsize', None)}")

    def __deepcopy__(self, memo):
        from comfy_kitchen.tensor import QuantizedTensor
        clone = ComfyKitchenLinear(
            self.in_features, self.out_features, bias=self.bias is not None,
            layout=self.layout, compute_dtype=self.compute_dtype, force_eager=self.force_eager,
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


@register_quant_backend("comfy_kitchen")
class ComfyKitchenQuantBackend(QuantBackend):
    """Adapter over comfy-kitchen's `QuantizedTensor`, the quantized-weight format ComfyUI writes."""

    def __init__(self, config=None):
        super().__init__(config)

    def validate_environment(self):
        if importlib.util.find_spec("comfy_kitchen") is None:
            raise ImportError(
                "comfy-kitchen is required for the ck_int8 quantization method. "
                'Please install it via `pip install comfy-kitchen` or `pip install "diffsynth[quant]"`.'
            )
        _initialize_registry()

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (ComfyKitchenLinear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        from comfy_kitchen.tensor import QuantizedTensor
        quant_format = self._active_format()
        options = quant_format.options_for_new_weight(linear.in_features, self.config)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        weight = QuantizedTensor.from_float(
            linear.weight.data, quant_format.layout_cls,
            **quant_format.from_float_kwargs(options, self.config),
        )
        quant_linear = self._build_linear(linear, quant_format.layout_cls, linear.weight.dtype)
        quant_linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data, requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self._build_linear(linear, self._active_format().layout_cls, compute_dtype)

    def unflatten_state_dict(self, state_dict, metadata):
        from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
        rebuilt = dict(state_dict)
        layer_names = [key[: -len(COMFY_QUANT_KEY) - 1] for key in state_dict if key.endswith(f".{COMFY_QUANT_KEY}")]
        for layer_name in layer_names:
            quant_format, marker = _parse_marker(rebuilt.pop(f"{layer_name}.{COMFY_QUANT_KEY}"), layer_name)
            scale = rebuilt.pop(f"{layer_name}.{SCALE_KEY}", None)
            if scale is None:
                raise ValueError(f"`{layer_name}` has a {COMFY_QUANT_KEY} marker but no {SCALE_KEY}.")
            qdata = rebuilt.get(f"{layer_name}.weight")
            if qdata is None:
                raise ValueError(f"`{layer_name}` has a {COMFY_QUANT_KEY} marker but no weight.")
            options = quant_format.options_from_marker(marker, layer_name)
            quant_format.check_options(options, qdata, layer_name)
            tensors = {}
            optional = getattr(quant_format, 'extra_tensor_optional', False)
            for key in quant_format.extra_tensor_keys:
                value = rebuilt.pop(f"{layer_name}.{key}", None)
                if value is None and not optional:
                    raise ValueError(f"`{quant_format.marker_name}` needs `{layer_name}.{key}`, which is missing.")
                if value is not None:
                    tensors[key] = value
            params = get_layout_class(quant_format.layout_cls).Params(
                scale=scale,
                orig_dtype=self.config.get("orig_dtype", torch.bfloat16),
                orig_shape=quant_format.logical_shape(qdata, options),
                **quant_format.params_kwargs(options, tensors),
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
            # Read the format off the weight, not the config, so a loaded model round-trips.
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
        marker_name = self.config["format"]
        if marker_name not in FORMATS:
            raise ValueError(f"Unsupported format `{marker_name}`, not in {sorted(FORMATS)}.")
        return FORMATS[marker_name]

    def _build_linear(self, linear, layout_cls, compute_dtype):
        from comfy_kitchen.tensor import get_layout_class
        return ComfyKitchenLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            layout=layout_cls,
            compute_dtype=compute_dtype,
            force_eager=not get_layout_class(layout_cls).supports_fast_matmul(),
        )


def _int8_config(backend_config_kwargs):
    return {
        "format": TensorWiseInt8Format.marker_name,
        "is_weight": True,
        "per_channel": True,
        "convrot": True,
        "convrot_groupsize": TensorWiseInt8Format.DEFAULT_GROUPSIZE,
        **backend_config_kwargs,
    }


register_quant_method(
    "ck_int8", "comfy_kitchen", _int8_config,
    label="8bit, int8 W8A8 (ComfyUI int8_tensorwise). `convrot` and `convrot_groupsize` apply to "
          "online quantization only; a prequantized checkpoint follows its per-layer marker.",
)


def _fp8_config(backend_config_kwargs):
    return {
        "format": Float8E4M3Format.marker_name,
        **backend_config_kwargs,
    }


register_quant_method(
    "ck_fp8", "comfy_kitchen", _fp8_config,
    label="8bit, fp8 E4M3 per-tensor scaled (ComfyUI float8_e4m3fn).",
)
