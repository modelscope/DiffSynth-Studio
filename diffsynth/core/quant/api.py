import torch
from .config import QuantizeConfig


def _name_matches(full_name, patterns):
    if patterns is None:
        return False
    if full_name in patterns:
        return True
    return any(full_name.endswith(f".{pattern}") for pattern in patterns)


def _should_quantize(full_name, module, quantize_config):
    if not isinstance(module, torch.nn.Linear):
        return False
    if quantize_config.target_modules is not None and not _name_matches(full_name, quantize_config.target_modules):
        return False
    if _name_matches(full_name, quantize_config.exclude_modules):
        return False
    return True


def _replace_target_linears(model, quantize_config, transform):
    # Snapshot via `list(...)` so replacements do not disturb the iteration.
    # Returns the full names of the replaced Linears.
    replaced = []
    for full_name, module in list(model.named_modules()):
        if full_name == "" or not _should_quantize(full_name, module, quantize_config):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf_name, transform(module))
        replaced.append(full_name)
    return replaced


def _dequantize_linears(model, backend, compute_dtype):
    # Returns the full names of the restored Linears.
    restored = []
    for full_name, module in list(model.named_modules()):
        # Only the backend can tell: torchao keeps `nn.Linear`, bnb subclasses it.
        if full_name == "" or not backend.is_quantized_linear(module):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf_name, backend.dequantize_to_linear(module, compute_dtype))
        restored.append(full_name)
    return restored


def quantize_model_weights(model: torch.nn.Module, quantize_config: QuantizeConfig, compute_dtype: torch.dtype = torch.bfloat16, device=None):
    """
    Quantize the target `nn.Linear` layers of an fp model in place (call AFTER `load_state_dict`).

    model: the fp model to quantize.
    quantize_config: preset / target_modules / exclude_modules / overrides to apply.
    compute_dtype: computation dtype of the quantized layers.
    device: target device of the quantized layers; the fp model may stay on CPU so that
        each layer transits through the device one by one. `None` quantizes in place.
    """
    backend, backend_config = quantize_config.resolve()
    backend.validate_environment(backend_config)

    def quantize(linear):
        return backend.quantize_linear_from_fp(linear, backend_config, compute_dtype, device=device)

    replaced = _replace_target_linears(model, quantize_config, quantize)
    print(f"{len(replaced)} nn.Linear layers quantized (preset: {quantize_config.preset}).")
    return model


def dequantize_model_weights(model: torch.nn.Module, quantize_config: QuantizeConfig, compute_dtype: torch.dtype = torch.bfloat16):
    """
    Replace every quantized Linear in the model by a plain fp `nn.Linear`.

    model: a model holding quantized Linears (from online quantization or a checkpoint).
    quantize_config: identifies the backend whose quantized Linears are restored.
    compute_dtype: dtype of the restored fp weights.
    """
    backend, _ = quantize_config.resolve()
    _dequantize_linears(model, backend, compute_dtype)
    return model


def replace_linear_for_quantized_load(model: torch.nn.Module, quantize_config: QuantizeConfig):
    """
    Replace the target `nn.Linear` layers by empty quantized Linears matching a
    pre-quantized checkpoint (call BEFORE `load_state_dict(assign=True)`).

    model: the freshly constructed fp model.
    quantize_config: preset / target_modules / exclude_modules deciding which layers to replace.
    """
    backend, backend_config = quantize_config.resolve()
    backend.validate_environment(backend_config)

    def build_shell(linear):
        return backend.create_quantized_linear_for_load(
            linear.in_features, linear.out_features, linear.bias is not None, backend_config,
        )

    replaced = _replace_target_linears(model, quantize_config, build_shell)
    print(f"{len(replaced)} nn.Linear layers replaced for loading the pre-quantized checkpoint (preset: {quantize_config.preset}).")
    return model


def save_quantized_model(model: torch.nn.Module, path: str, quantize_config: QuantizeConfig):
    """
    Save a quantized model to a safetensors file in its backend's own layout.

    model: the quantized model to save.
    path: output safetensors file path.
    quantize_config: identifies the backend whose serialization layout is used.
    """
    from safetensors.torch import save_file
    backend, backend_config = quantize_config.resolve()
    tensors, metadata = backend.flatten_state_dict(model.state_dict())
    tensors = {key: value.contiguous() for key, value in tensors.items()}
    # safetensors headers hold strings only.
    metadata = {"format": "pt", **{key: value if isinstance(value, str) else str(value) for key, value in metadata.items()}}
    save_file(tensors, path, metadata=metadata)
    return path
