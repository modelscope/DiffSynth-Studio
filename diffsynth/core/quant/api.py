import torch
from .config import QuantizeConfig


def _name_matches(full_name, patterns):
    if patterns is None:
        return False
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(pattern in full_name for pattern in patterns)


def _should_quantize(full_name, module, quantize_config):
    if not isinstance(module, torch.nn.Linear):
        return False
    if quantize_config.target_modules is not None and not _name_matches(full_name, quantize_config.target_modules):
        return False
    if _name_matches(full_name, quantize_config.exclude_modules):
        return False
    return True


def _replace_target_linears(model, quantize_config, transform, name_prefix=""):
    # Replace every targeted `nn.Linear` by `transform(linear)`. The traversal holds no
    # quantization logic, so the caller composes what "replace" means.
    for name, module in model.named_children():
        full_name = name if name_prefix == "" else f"{name_prefix}.{name}"
        if _should_quantize(full_name, module, quantize_config):
            setattr(model, name, transform(module))
        else:
            _replace_target_linears(module, quantize_config, transform, full_name)


def _dequantize_recursively(model, backend, compute_dtype):
    for name, module in model.named_children():
        # Only the backend can tell: torchao keeps `nn.Linear`, bnb subclasses it.
        if backend.is_quantized_linear(module):
            setattr(model, name, backend.dequantize_to_linear(module, compute_dtype))
        else:
            _dequantize_recursively(module, backend, compute_dtype)


def quantize_model_weights(model: torch.nn.Module, quantize_config: QuantizeConfig, default_compute_dtype: torch.dtype = None, device=None):
    """
    One-shot weight-only quantization of an fp model (call AFTER `load_state_dict`).
    Replaces target `nn.Linear` layers with backend-native quantized Linears.

    Streaming to avoid a full-fp peak on GPU: load the fp model on CPU and pass the
    target `device` here -- each layer is moved to `device` and quantized one by one
    (its fp copy is released immediately), so the GPU never holds the whole fp model.
    Non-quantized modules are moved to `device` afterwards (device-only, packed
    dtypes are never cast). With `device=None`, quantization happens in place on
    whatever device the model already resides.

    With `mode="dequant_once"`, each layer goes back to a plain fp `nn.Linear` right after
    being quantized, so the model runs at full precision while carrying the quantization
    error once.
    """
    backend, backend_config = quantize_config.resolve()
    backend.validate_environment(backend_config)
    compute_dtype = quantize_config.compute_dtype or default_compute_dtype or torch.bfloat16

    def quantize(linear):
        return backend.quantize_linear_from_fp(linear, backend_config, compute_dtype, device=device)

    if quantize_config.mode == "dequant_once":
        transform = lambda linear: backend.dequantize_to_linear(quantize(linear), compute_dtype)
    else:
        transform = quantize
    _replace_target_linears(model, quantize_config, transform)
    if device is not None:
        # Device-only move: packed quantized weights must not be dtype-cast.
        model = model.to(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def dequantize_model_weights(model: torch.nn.Module, quantize_config: QuantizeConfig, default_compute_dtype: torch.dtype = None):
    """
    Replace every quantized Linear by a plain fp `nn.Linear` (`mode="dequant_once"`).
    Called after a pre-quantized checkpoint has been loaded, so the model runs at full
    precision while having gone through the checkpoint's quantization once.
    """
    backend, _ = quantize_config.resolve()
    compute_dtype = quantize_config.compute_dtype or default_compute_dtype or torch.bfloat16
    _dequantize_recursively(model, backend, compute_dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def replace_linear_for_quantized_load(model: torch.nn.Module, quantize_config: QuantizeConfig):
    """
    Structure-only pass for loading a pre-quantized checkpoint (call BEFORE `load_state_dict`).

    Every targeted `nn.Linear` is replaced by an empty quantized Linear from the backend,
    whose own state dict keys match the checkpoint; `load_state_dict(assign=True)` then
    fills it. Which layers are targeted is driven purely by `target_modules` /
    `exclude_modules` -- the checkpoint is not sniffed.
    """
    backend, backend_config = quantize_config.resolve()
    backend.validate_environment(backend_config)

    replaced = 0

    def build_shell(linear):
        nonlocal replaced
        replaced += 1
        return backend.create_quantized_linear_for_load(
            linear.in_features, linear.out_features, linear.bias is not None, backend_config,
        )

    _replace_target_linears(model, quantize_config, build_shell)
    if replaced == 0:
        raise ValueError(
            f"No `nn.Linear` matched the quantization target for backend `{backend.name}`. "
            "Check `target_modules` / `exclude_modules`."
        )
    return model


def save_quantized_model(model: torch.nn.Module, path: str, quantize_config: QuantizeConfig):
    """
    Write a quantized model to a safetensors file in its backend's own layout.

    Each backend keeps the layout its own ecosystem uses -- torchao splits its tensor
    subclasses into plain tensors plus a metadata header (`torchao.prototype.safetensors`),
    bnb writes the packed weight next to its `weight.*` quant state -- so the file stays
    readable by other tools built on the same backend. No DiffSynth-specific metadata is
    added: which preset a published file needs is recorded in MODEL_CONFIGS, mirroring how
    the community keeps that in `config.json`.

    What lands in the file follows the model in memory: with `mode="dynamic"` the layers are
    quantized, so the file is a quantized checkpoint; with `mode="dequant_once"` they are
    already back to fp, so the file is a plain fp checkpoint.
    """
    from safetensors.torch import save_file
    backend, backend_config = quantize_config.resolve()
    tensors, metadata = backend.flatten_state_dict(model.state_dict())
    tensors = {key: value.contiguous() for key, value in tensors.items()}
    # safetensors headers hold strings only.
    metadata = {"format": "pt", **{key: value if isinstance(value, str) else str(value) for key, value in metadata.items()}}
    save_file(tensors, path, metadata=metadata)
    return path
