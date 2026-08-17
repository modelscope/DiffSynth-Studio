import copy
import functools
import importlib.util
import json

import torch
from ..base import QuantBackend, register_quant_backend
from ..config import register_quant_method


COMFY_QUANT_KEY = "comfy_quant"
SCALE_KEY = "weight_scale"
# Only `int8_tensorwise` is implemented, and the support is deeper than this table: every
# comfy-kitchen layout declares its own `Params` fields (INT8 has `is_weight` / `convrot` /
# `convrot_groupsize`, FP8 has none of them, NVFP4 adds `block_scale`, ConvRotW4A4 adds
# `quant_group_size` / `linear_dtype`), so adding a format means revisiting four places, not
# just this dict: the `Params` construction in `unflatten_state_dict`, the `orig_shape`
# derivation there (valid only while the packed data keeps the logical shape),
# `checkpoint_key_patterns` (FP8 also stores `input_scale`, NVFP4 a `block_scale`), and the
# convrot handling, which is an int8-only parameter. An unknown format raises in
# `_parse_marker` rather than being guessed.
SUPPORTED_FORMATS = {"int8_tensorwise": "TensorWiseINT8Layout"}
DEFAULT_CONVROT_GROUPSIZE = 256


@functools.cache
def _initialize_registry():
    """Pin comfy-kitchen to backends that actually run here, once per process.

    Its cuda backend needs CUDA >= 13 and reports itself as available regardless, so a
    call would fail with "CUDA driver version is insufficient" on an older toolkit.
    """
    import comfy_kitchen as ck
    if torch.version.cuda is None or tuple(int(v) for v in torch.version.cuda.split(".")[:2]) < (13, 0):
        ck.registry.disable("cuda")


def _is_regular_hadamard_size(size: int) -> bool:
    """comfy-kitchen builds the rotation as Kronecker powers of a 4x4 Hadamard matrix."""
    return size >= 4 and size & (size - 1) == 0 and (size.bit_length() - 1) % 2 == 0


def _resolve_convrot(in_features, convrot, groupsize):
    """Pick a rotation group size that divides `in_features`, mirroring the upstream rule.

    Returns `(convrot, groupsize)`. A requested size that does not divide `in_features` is
    divided by four until one fits, and the rotation is dropped when none does; keeping the
    requested size instead would either raise or, when it happens to divide, silently
    produce a wrongly rotated weight.
    """
    if not convrot:
        return False, groupsize
    if not _is_regular_hadamard_size(groupsize):
        raise ValueError(
            "`convrot_groupsize` must be a power of four and at least 4 (the rotation is a "
            f"regular Hadamard transform), but got {groupsize}."
        )
    candidate = groupsize
    while candidate >= 4:
        if in_features % candidate == 0:
            return True, candidate
        candidate //= 4
    return False, groupsize


class ComfyKitchenLinear(torch.nn.Linear):
    """`nn.Linear` whose weight is a comfy-kitchen `QuantizedTensor`.

    Forward dispatch lives in the tensor subclass and reads the per-layer quant state off
    the weight, so this class only records the layout and pins the comfy-kitchen backend
    when the layout has no fast kernel on the current device.
    """

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
        # comfy-kitchen's int8 activation quantization has no fast strided kernel: a
        # non-contiguous input measured 2.5x slower than the same input made contiguous,
        # while bf16 cuBLAS absorbs strides for free.
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
    """Adapter over comfy-kitchen's `QuantizedTensor`, the quantized-weight format ComfyUI writes.

    A checkpoint mixes rotation settings layer by layer (upstream lowers the group size
    when it does not divide `in_features`), so pre-quantized loading reads the per-layer
    `comfy_quant` marker rather than trusting the method's defaults.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._logged_shapes = set()

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
            "is_differentiable": False,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (ComfyKitchenLinear,)

    def checkpoint_key_patterns(self):
        """The packed weight, its scale and its marker are all siblings, so the nested default would drop the last two."""
        return ("weight", SCALE_KEY, COMFY_QUANT_KEY, "bias")

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        from comfy_kitchen.tensor import QuantizedTensor
        self.validate_environment()
        layout = self._layout()
        convrot, groupsize = self._layer_convrot(linear.in_features)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        weight = QuantizedTensor.from_float(
            linear.weight.data, layout,
            is_weight=True,
            per_channel=self.config["per_channel"],
            convrot=convrot,
            convrot_groupsize=groupsize,
        )
        quant_linear = self._build_linear(linear, layout, linear.weight.dtype)
        quant_linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data, requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        self.validate_environment()
        return self._build_linear(linear, self._layout(), compute_dtype)

    def unflatten_state_dict(self, state_dict, metadata):
        from comfy_kitchen.tensor import QuantizedTensor, get_layout_class
        self.validate_environment()
        rebuilt = dict(state_dict)
        layer_names = [key[: -len(COMFY_QUANT_KEY) - 1] for key in state_dict if key.endswith(f".{COMFY_QUANT_KEY}")]
        for layer_name in layer_names:
            marker = self._parse_marker(rebuilt.pop(f"{layer_name}.{COMFY_QUANT_KEY}"), layer_name)
            layout = SUPPORTED_FORMATS[marker["format"]]
            scale = rebuilt.pop(f"{layer_name}.{SCALE_KEY}", None)
            if scale is None:
                raise ValueError(f"The quantized layer `{layer_name}` carries a `{COMFY_QUANT_KEY}` marker but no `{SCALE_KEY}`.")
            qdata = rebuilt.get(f"{layer_name}.weight")
            if qdata is None:
                raise ValueError(f"The quantized layer `{layer_name}` carries a `{COMFY_QUANT_KEY}` marker but no packed weight.")
            convrot, groupsize = marker["convrot"], marker["convrot_groupsize"]
            in_features = qdata.shape[1]
            if convrot and in_features % groupsize != 0:
                raise ValueError(
                    f"The marker of `{layer_name}` asks for convrot with groupsize {groupsize}, which does not "
                    f"divide in_features={in_features}. The checkpoint is inconsistent; loading it would rotate "
                    "the weight against a different basis than it was quantized with."
                )
            params = get_layout_class(layout).Params(
                scale=scale,
                orig_dtype=self.config.get("orig_dtype", torch.bfloat16),
                orig_shape=tuple(qdata.shape),
                is_weight=True,
                convrot=convrot,
                convrot_groupsize=groupsize,
            )
            rebuilt[f"{layer_name}.weight"] = QuantizedTensor(qdata, layout, params)
        return rebuilt

    def flatten_state_dict(self, state_dict):
        flattened = {}
        for key, value in state_dict.items():
            params = getattr(value, "_params", None)
            if not key.endswith(".weight") or params is None:
                flattened[key] = value
                continue
            layer_name = key[: -len(".weight")]
            flattened.update(value.state_dict(prefix=key))
            marker = {"format": self._format(), "convrot": bool(params.convrot)}
            if params.convrot:
                marker["convrot_groupsize"] = int(params.convrot_groupsize)
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

    def _format(self):
        quant_format = self.config["format"]
        if quant_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported comfy-kitchen format `{quant_format}`. Supported: {sorted(SUPPORTED_FORMATS)}.")
        return quant_format

    def _layout(self):
        return SUPPORTED_FORMATS[self._format()]

    def _build_linear(self, linear, layout, compute_dtype):
        from comfy_kitchen.tensor import get_layout_class
        return ComfyKitchenLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            layout=layout,
            compute_dtype=compute_dtype,
            force_eager=not get_layout_class(layout).supports_fast_matmul(),
        )

    def _layer_convrot(self, in_features):
        requested_convrot = self.config["convrot"]
        requested_groupsize = self.config["convrot_groupsize"]
        convrot, groupsize = _resolve_convrot(in_features, requested_convrot, requested_groupsize)
        if in_features not in self._logged_shapes:
            self._logged_shapes.add(in_features)
            if (convrot, groupsize) != (requested_convrot, requested_groupsize):
                print(f"comfy_kitchen: in_features={in_features} is not divisible by the requested convrot "
                      f"groupsize {requested_groupsize}; using convrot={convrot}, groupsize={groupsize if convrot else None}.")
            else:
                print(f"comfy_kitchen: in_features={in_features} uses convrot={convrot}, "
                      f"groupsize={groupsize if convrot else None}.")
        return convrot, groupsize

    @staticmethod
    def _parse_marker(marker_tensor, layer_name):
        try:
            marker = json.loads(bytes(marker_tensor.to(torch.uint8).tolist()).decode())
        except Exception as error:
            raise ValueError(f"Could not decode the `{COMFY_QUANT_KEY}` marker of `{layer_name}`: {type(error).__name__}: {error}")
        if "format" not in marker:
            raise ValueError(
                f"The marker of `{layer_name}` has no `format` field, which identifies the deprecated "
                f"`int8_w8a8` layout. Re-export the checkpoint with a current ComfyUI build; guessing the "
                "layout would silently produce wrong weights."
            )
        if marker["format"] not in SUPPORTED_FORMATS:
            raise ValueError(
                f"The marker of `{layer_name}` requests format `{marker['format']}`, which this backend does "
                f"not implement. Supported: {sorted(SUPPORTED_FORMATS)}."
            )
        nested = marker.get("params", {})
        if nested.get("full_precision_matrix_mult", marker.get("full_precision_matrix_mult", False)):
            raise NotImplementedError(
                f"The marker of `{layer_name}` requests `full_precision_matrix_mult`, which this backend does "
                "not implement yet. Ignoring it would run the layer through the fast kernel the checkpoint "
                "asked to avoid."
            )
        return {
            "format": marker["format"],
            "convrot": bool(marker.get("convrot", nested.get("convrot", False))),
            "convrot_groupsize": int(marker.get("convrot_groupsize", nested.get("convrot_groupsize", DEFAULT_CONVROT_GROUPSIZE))),
        }


def _int8_config(backend_config_kwargs):
    return {
        "format": "int8_tensorwise",
        "is_weight": True,
        "per_channel": True,
        "convrot": True,
        "convrot_groupsize": DEFAULT_CONVROT_GROUPSIZE,
        **backend_config_kwargs,
    }


# One method, because the rotation is a single knob (on/off plus a group size that must be
# a power of four), not a distinct checkpoint format -- ComfyUI has no `int8_convrot` entry
# in `QUANT_ALGOS`. It is also read from the per-layer marker when loading a pre-quantized
# checkpoint, where one file can mix rotated and unrotated layers, so the method name could
# never describe a file anyway.
register_quant_method(
    "ck_int8", "comfy_kitchen", _int8_config,
    label="8bit, int8 W8A8 (ComfyUI int8_tensorwise). Rotation is tunable via "
          "backend_config_kwargs={'convrot': True|False, 'convrot_groupsize': <power of four>} "
          "and only affects online quantization; loading a prequantized checkpoint always "
          "follows its per-layer marker.",
)
