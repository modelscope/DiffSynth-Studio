import inspect
import json
import numpy as np
import torch
from PIL import Image
from .pipeline_registry import PIPELINE_REGISTRY, get_pipeline_class
from .signature_parser import parse_call_signature, _is_image, _is_image_list, _tuple_annotation, _list_annotation, _REQUIRED_INPUTS
from .type_defs import PIPE


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    return Image.fromarray((tensor.clamp(0, 1).numpy() * 255).astype(np.uint8))


def _tensor_to_pil_list(tensor: torch.Tensor) -> list:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        return [_tensor_to_pil(tensor)]
    images = []
    for i in range(tensor.shape[0]):
        images.append(Image.fromarray((tensor[i].clamp(0, 1).numpy() * 255).astype(np.uint8)))
    return images


def _to_pil(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, torch.Tensor):
        return _tensor_to_pil(value)
    return value


def _to_pil_list(value):
    if isinstance(value, list) and value and isinstance(value[0], Image.Image):
        return value
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, torch.Tensor):
        return _tensor_to_pil_list(value)
    return value


def _parse_tuple(value, elem_type="any"):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return None
        if elem_type == "int":
            return tuple(int(p) for p in parts)
        elif elem_type == "float":
            return tuple(float(p) for p in parts)
        else:
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                try:
                    return tuple(float(p) for p in parts)
                except ValueError:
                    return tuple(parts)
    return value


def _parse_list(value, elem_type="any"):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if elem_type == "str":
            lines = [line.strip() for line in value.split("\n") if line.strip()]
            return lines if lines else None
        if elem_type == "int":
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return [int(p) for p in parts] if parts else None
        if elem_type == "float":
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return [float(p) for p in parts] if parts else None
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if not parts:
                return None
            try:
                return [int(p) for p in parts]
            except ValueError:
                try:
                    return [float(p) for p in parts]
                except ValueError:
                    return parts
    return value


def _convert_param(name, value, annotation):
    if value is None:
        return None

    if _is_image_list(annotation):
        return _to_pil_list(value)
    if annotation is Image.Image:
        return _to_pil(value)

    tuple_elem = _tuple_annotation(annotation)
    if tuple_elem is not None:
        return _parse_tuple(value, tuple_elem)

    list_elem = _list_annotation(annotation)
    if list_elem is not None:
        return _parse_list(value, list_elem)

    return value


def _parameter_needs_conversion(annotation):
    if _is_image(annotation):
        return True
    if _tuple_annotation(annotation) is not None:
        return True
    if _list_annotation(annotation) is not None:
        return True
    return False


def _to_image(value):
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Image.Image):
        arrays = [np.asarray(item.convert("RGB"), dtype=np.float32) / 255.0 for item in value]
        return torch.from_numpy(np.stack(arrays))
    if isinstance(value, torch.Tensor):
        return value.float()
    return value


def _to_audio(value, sample_rate=44100):
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("audios"))
        sr = value.get("sample_rate", sample_rate)
        if isinstance(waveform, torch.Tensor):
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)      # [T] -> [1, T]
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)      # [C, T] -> [1, C, T]
        return {"waveform": waveform, "sample_rate": int(sr)}
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            value = value.unsqueeze(0)                # [T] -> [1, T]
        if value.ndim == 2:
            value = value.unsqueeze(0)                # [C, T] -> [1, C, T]
        return {"waveform": value, "sample_rate": int(sample_rate)}
    return value


def _convert_output(value, output_type, sample_rate=44100):
    if output_type in ("image", "video"):
        return _to_image(value)
    if output_type == "audio":
        return _to_audio(value, sample_rate)
    if output_type == "audio_video":
        if isinstance(value, tuple):
            return (_to_image(value[0]), _to_audio(value[1], sample_rate))
        if isinstance(value, dict):
            return (_to_image(value.get("video", value.get("images"))), _to_audio(value.get("audio"), sample_rate))
    return value


def generate_inference_nodes():
    nodes = {}
    for type_name, meta in PIPELINE_REGISTRY.items():
        required, optional = parse_call_signature(get_pipeline_class(type_name), type_name)
        required = {"pipe": (PIPE,)} | required

        def execute(self, _meta=meta, **kwargs):
            pipe = kwargs.pop("pipe")
            sig_params = inspect.signature(pipe.__call__).parameters

            call_kwargs = {}

            for name in _REQUIRED_INPUTS:
                if name in kwargs and kwargs[name] is not None:
                    call_kwargs[name] = kwargs[name]

            for name, value in kwargs.items():
                if name in call_kwargs:
                    continue
                if name not in sig_params:
                    continue
                if value is None:
                    continue
                annotation = sig_params[name].annotation
                if _parameter_needs_conversion(annotation):
                    call_kwargs[name] = _convert_param(name, value, annotation)
                else:
                    call_kwargs[name] = value

            result = pipe(**call_kwargs)
            sample_rate = getattr(pipe, "audio_sample_rate", None)
            if sample_rate is None:
                sample_rate = getattr(pipe, "retake_audio_sample_rate", None)
            if sample_rate is None:
                sample_rate = 44100
            if _meta.output_type == "audio_video":
                return _convert_output(result, _meta.output_type, sample_rate)
            return (_convert_output(result, _meta.output_type, sample_rate),)

        def is_changed(**kwargs):
            return kwargs.get("seed", float("nan"))

        def input_types(cls, _required=required, _optional=optional):
            return {"required": _required, "optional": _optional}

        node_name = f"DiffSynth{type_name}Inference"
        node_class = type(node_name, (), {
            "INPUT_TYPES": classmethod(input_types),
            "RETURN_TYPES": ("IMAGE",) if meta.output_type in ("image", "video") else (("AUDIO",) if meta.output_type == "audio" else ("IMAGE", "AUDIO")),
            "RETURN_NAMES": ("image",) if meta.output_type == "image" else (("video",) if meta.output_type == "video" else (("audio",) if meta.output_type == "audio" else ("video", "audio"))),
            "FUNCTION": "execute", "CATEGORY": "DiffSynth/inference", "OUTPUT_NODE": True,
            "execute": execute, "IS_CHANGED": staticmethod(is_changed),
        })
        nodes[node_name] = node_class
    return nodes
