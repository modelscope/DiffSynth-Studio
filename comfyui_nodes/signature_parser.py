import inspect
import types
from typing import Literal, Union, get_args, get_origin
from PIL import Image


_EXCLUDED = {
    "self", "progress_bar_cmd", "tqdm",
    "lora", "negative_lora", "positive_only_lora",
    "rand_device", "output_type",
    "kv_cache", "negative_kv_cache",
    "extra_text_embedding", "negative_extra_text_embedding",
    "residual", "negative_residual",
    "tea_cache_model_id",
}


_REQUIRED_INPUTS = [
    "prompt",
    "lyrics",
    "seed",
    "num_inference_steps",
    "height",
    "width",
]


def _is_image(annotation):
    if annotation is Image.Image:
        return True
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        return len(args) == 1 and args[0] is Image.Image
    return False


def _is_image_list(annotation):
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        return len(args) == 1 and args[0] is Image.Image
    return False


def _unwrap_union(annotation):
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if not non_none:
        return annotation
    for priority in (str, int, float, bool):
        if priority in non_none:
            return priority
    return non_none[0]


def _tuple_annotation(annotation):
    inner = _unwrap_union(annotation)
    origin = get_origin(inner)
    if origin is tuple or inner is tuple:
        args = get_args(inner)
        if not args:
            return "any"
        elem_types = set()
        for a in args:
            if a is int:
                elem_types.add("int")
            elif a is float:
                elem_types.add("float")
            elif a is str:
                elem_types.add("str")
            else:
                elem_types.add("any")
        if len(elem_types) == 1:
            return elem_types.pop()
        return "any"
    return None


def _list_annotation(annotation):
    inner = _unwrap_union(annotation)
    origin = get_origin(inner)
    if origin is not list and inner is not list:
        return None
    args = get_args(inner)
    if not args:
        return "any"
    a = args[0]
    if a is int:
        return "int"
    if a is float:
        return "float"
    if a is str:
        return "str"
    return "any"


def _input_for(name, annotation, default):
    if name == "value_controller_inputs":
        return ("*", {})
    if get_origin(annotation) is Literal:
        values = [str(v) for v in get_args(annotation)]
        default_val = default if default is not inspect.Parameter.empty else values[0]
        return (values, {"default": str(default_val) if default_val is not None else values[0]})

    annotation = _unwrap_union(annotation)

    if annotation is inspect.Parameter.empty:
        return _infer_from_default(name, default)
    if annotation is str:
        value = "" if (default is inspect.Parameter.empty or default is None) else str(default)
        return ("STRING", {"default": value, "multiline": True})
    if annotation is bool:
        return ("BOOLEAN", {"default": False if (default is inspect.Parameter.empty or default is None) else default})
    if annotation is int:
        return ("INT", _int_options(name, default))
    if annotation is float:
        return ("FLOAT", _float_options(name, default))
    if _is_image(annotation):
        return ("IMAGE", {})

    tuple_elem = _tuple_annotation(annotation)
    if tuple_elem is not None:
        if default is not inspect.Parameter.empty and default is not None:
            default_str = ",".join(str(v) for v in default) if isinstance(default, (tuple, list)) else str(default)
        else:
            default_str = ""
        return ("STRING", {"default": default_str, "multiline": False})

    list_elem = _list_annotation(annotation)
    if list_elem is not None and list_elem in ("str", "int", "float"):
        if default is not inspect.Parameter.empty and default is not None:
            if list_elem == "str":
                default_str = "\n".join(str(v) for v in default) if isinstance(default, (list, tuple)) else str(default)
            else:
                default_str = ",".join(str(v) for v in default) if isinstance(default, (list, tuple)) else str(default)
        else:
            default_str = ""
        return ("STRING", {"default": default_str, "multiline": list_elem == "str"})

    if list_elem == "any":
        inner = _unwrap_union(annotation)
        inner_args = get_args(inner)
        if inner_args:
            elem = inner_args[0]
            if get_origin(elem) is tuple or elem is tuple:
                default_str = "" if (default is inspect.Parameter.empty or default is None) else str(default)
                return ("STRING", {"default": default_str, "multiline": True})
        if inner is list:
            default_str = "" if (default is inspect.Parameter.empty or default is None) else str(default)
            return ("STRING", {"default": default_str, "multiline": True})
        return ("*", {})
    return ("*", {})


def _infer_from_default(name, default):
    if default is inspect.Parameter.empty:
        return ("STRING", {"default": "", "multiline": True})
    if isinstance(default, bool):
        return ("BOOLEAN", {"default": default})
    if isinstance(default, int):
        return ("INT", _int_options(name, default))
    if isinstance(default, float):
        return ("FLOAT", _float_options(name, default))
    if isinstance(default, str):
        return ("STRING", {"default": default, "multiline": True})
    return ("*", {})


def _safe_default(default, fallback):
    if default is inspect.Parameter.empty or default is None:
        return fallback
    return default


def _int_options(name, default):
    d = _safe_default(default, 0)
    options = {"default": d, "min": 0, "max": 2**32 - 1, "step": 1}
    if name in ("width", "height"):
        options.update({"min": 64, "max": 4096, "default": _safe_default(default, 1024)})
    elif name == "num_frames":
        options.update({"min": 1, "max": 1000, "default": _safe_default(default, 81)})
    elif name == "num_inference_steps":
        options.update({"min": 1, "max": 200, "default": _safe_default(default, 30)})
    elif name == "seed":
        options["control_after_generate"] = True
    return options


def _float_options(name, default):
    d = _safe_default(default, 0.0)
    options = {"default": d, "min": 0.0, "max": 10000.0, "step": 0.1}
    if name in {"wantodance_fps", "max_audio_duration"}:
        options["max"] = 10000.0
        options["step"] = 1.0
    return options


def parse_call_signature(pipeline_class, pipeline_type):
    sig_params = {}
    for name, parameter in inspect.signature(pipeline_class.__call__).parameters.items():
        if name in _EXCLUDED or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        sig_params[name] = parameter

    required = {}
    optional = {}
    for name, parameter in sig_params.items():
        spec = _input_for(name, parameter.annotation, parameter.default)
        if name in _REQUIRED_INPUTS:
            required[name] = spec
        else:
            optional[name] = spec

    for name, parameter in sig_params.items():
        if name in _REQUIRED_INPUTS or parameter.default is not None:
            continue
        spec = optional.get(name)
        if not isinstance(spec, tuple) or len(spec) < 2:
            continue
        spec_type = spec[0]
        if isinstance(spec_type, str) and spec_type in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            opts = dict(spec[1])
            opts["forceInput"] = True
            optional[name] = (spec_type, opts)

    ordered_required = {}
    for name in _REQUIRED_INPUTS:
        if name in required:
            ordered_required[name] = required[name]
    required = ordered_required

    return required, optional
