from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict, load_metadata_from_safetensors
import os
import torch
from contextlib import contextmanager
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers


def _materialize_module_from_disk(module, disk_map, prefix="", torch_dtype=None, device=None, skip_prefixes=()):
    """Load one submodule's parameters/buffers from a `DiskMap`, by name.

    Only the tensors under `prefix` are touched, so a multi-GB checkpoint never has
    to be resident in host memory all at once. Subtrees listed in `skip_prefixes`
    are left alone, which is how the block-wise quantization path avoids
    overwriting weights it has already packed.
    """
    if any(prefix == skip or prefix.startswith(skip) for skip in skip_prefixes):
        return module
    state = {}
    for name, _param in module.named_parameters(recurse=False):
        key = prefix + name
        if key in disk_map:
            tensor = disk_map[key]
            state[name] = tensor.to(torch_dtype) if torch_dtype is not None else tensor
    for name, _buf in module.named_buffers(recurse=False):
        key = prefix + name
        if key in disk_map:
            state[name] = disk_map[key]
    if state:
        module.load_state_dict(state, assign=True, strict=False)
    # Children first: `.to()` on a module that still holds meta parameters raises
    # ("Cannot copy out of meta tensor"), so nothing is moved until it is real.
    for child_name, child in module.named_children():
        _materialize_module_from_disk(child, disk_map, prefix + child_name + ".", torch_dtype, device, skip_prefixes)
    if device is not None and not any(
        param.is_meta for param in list(module.parameters()) + list(module.buffers()) if param is not None
    ):
        module.to(device=device)
    return module


def _split_units(model):
    """Top-most submodules the model declares indivisible (`_no_split_modules`).

    The declaration may live on a submodule rather than on the wrapper: DiffSynth
    wraps a `transformers` model whose own class carries `_no_split_modules` (e.g.
    `Qwen3VLForConditionalGeneration` -> `Qwen3VLTextDecoderLayer`), so the names
    are collected from every class in the tree.
    """
    split_names = set()
    for module in model.modules():
        split_names.update(getattr(module, "_no_split_modules", None) or [])
    if not split_names:
        return []
    units, taken = [], []
    for name, module in model.named_modules():
        if name and type(module).__name__ in split_names:
            if any(name.startswith(prefix) for prefix in taken):
                continue
            taken.append(name + ".")
            units.append((name, module))
    return units


def _stream_quantize_model(model, disk_map, quantize, torch_dtype, compute_device, model_device):
    """Quantize a model one indivisible block at a time, straight from disk.

    The default online path materializes the whole fp checkpoint before packing a
    single layer, so host memory has to hold all of it and, when `offload_device` is
    the accelerator, VRAM does too: 13.3 GB for the Qwen-Image-2.1 DiT, 16.3 GB for
    its Qwen3-VL text encoder. Neither fits a 16 GB card. Here each block is read,
    packed and freed in turn, so the host peak is one block (~0.2 GB) plus the
    embeddings, and the accelerator only ever accumulates the packed weights.

    Enabled with `DIFFSYNTH_QUANT_STREAM=1`; requires the model to declare
    `_no_split_modules`.
    """
    units = _split_units(model)
    if not units:
        raise ValueError(
            "DIFFSYNTH_QUANT_STREAM=1 needs a model that declares `_no_split_modules` "
            f"(got {type(model).__name__}). Unset the variable to load the fp weights normally."
        )
    taken = [name + "." for name, _ in units]
    print(f"Streaming quantization: {len(units)} blocks of {type(units[0][1]).__name__}, one resident in host memory at a time.")
    for name, unit in units:
        # No device move here: `quantize_model` already parks the packed weights on
        # `model_device`, and the caller finishes with a whole-model `.to()`. Moving
        # blocks in between would just shuttle the packed tensors back and forth.
        _materialize_module_from_disk(unit, disk_map, name + ".", torch_dtype)
        quantize.quantize_model(unit, compute_device=compute_device, model_device=model_device)
    # Everything outside the blocks (embeddings, in/out projections, norms).
    _materialize_module_from_disk(model, disk_map, "", torch_dtype, skip_prefixes=taken)
    # Idempotent: layers the backend already packed are skipped by `_should_quantize`.
    quantize.quantize_model(model, compute_device=compute_device, model_device=model_device)
    # `load_state_dict(strict=True)` is what normally catches a checkpoint missing a
    # tensor; with assign-per-block that check is gone, so verify nothing stayed meta.
    still_meta = [name for name, param in model.named_parameters() if param is not None and param.is_meta]
    if still_meta:
        raise ValueError(
            f"{len(still_meta)} parameters were never loaded from the checkpoint and are still on "
            f"the meta device (e.g. {', '.join(still_meta[:5])}). The safetensors files do not cover "
            "this model; unset DIFFSYNTH_QUANT_STREAM to get the standard load error."
        )
    return model


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None, state_dict=None, quantize=None):
    config = {} if config is None else config
    with ContextManagers(get_init_context(torch_dtype=torch_dtype, device=device)):
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None and quantize is None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            if state_dict is None: state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    elif quantize is not None and module_map is not None:
        if "disk" in vram_config.values():
            if not quantize.load_prequantized:
                raise ValueError("Disk offload with quantization is only supported for pre-quantized checkpoints (load_prequantized=True).")
            devices = [vram_config[k] for k in ("offload_device", "onload_device", "preparing_device", "computation_device")]
            load_device = [d for d in devices if d != "disk"][0]
            disk_map = DiskMap(path, load_device, torch_dtype=None, state_dict_converter=state_dict_converter)
            metadata = load_metadata_from_safetensors(path)
            model = quantize.prepare_for_prequantized_load(model, compute_dtype=vram_config["computation_dtype"])
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit, quantize=quantize, metadata=metadata)
        else:
            offload_device = vram_config["offload_device"]
            computation_device = vram_config["computation_device"]
            computation_dtype = vram_config["computation_dtype"]
            load_dtype = None if quantize.load_prequantized else computation_dtype
            # Online quantization streams the fp weights through host memory: reading
            # them straight into `offload_device` would make the whole fp model
            # resident on the accelerator before a single layer is packed. Pre-quantized
            # checkpoints are already packed and small, so they load where they live.
            load_device = offload_device if quantize.load_prequantized else "cpu"
            # `DIFFSYNTH_QUANT_STREAM=1` keeps the checkpoint lazy and packs it block by
            # block (see `_stream_quantize_model`). It only applies to online
            # quantization of a model we read ourselves; a caller-supplied state dict is
            # already in memory, so streaming it would buy nothing.
            stream = (
                os.environ.get("DIFFSYNTH_QUANT_STREAM", "0") == "1"
                and not quantize.load_prequantized
                and state_dict is None
            )
            if stream:
                # DiskMap applies `state_dict_converter` as a name map, so the renamed
                # keys resolve without materializing a single tensor.
                state_dict = DiskMap(path, load_device, torch_dtype=load_dtype, state_dict_converter=state_dict_converter)
                model = _stream_quantize_model(
                    model, state_dict, quantize, load_dtype, computation_device, offload_device,
                )
                state_dict = None
            else:
                if state_dict is None: state_dict = DiskMap(path, load_device, torch_dtype=load_dtype)
                if state_dict_converter is not None:
                    state_dict = state_dict_converter(state_dict)
                else:
                    state_dict = {i: state_dict[i] for i in state_dict}

                if quantize.load_prequantized:
                    model = quantize.prepare_for_prequantized_load(model, compute_dtype=computation_dtype)
                    state_dict = quantize.unflatten_state_dict(state_dict, load_metadata_from_safetensors(path))

                model.load_state_dict(state_dict, assign=True)
                state_dict = None

                model = quantize.quantize_model(model, compute_device=computation_device, model_device=offload_device)

            model = quantize.dequantize_model(model, compute_dtype=computation_dtype, compute_device=computation_device, model_device=offload_device)
            model = model.to(dtype=computation_dtype, device=offload_device)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit, quantize=quantize)
    elif quantize is not None:
        # Weight-only quantization (see `diffsynth.core.quant`), isolated from the normal path below.
        if quantize.load_prequantized:
            load_device, load_dtype = device, None
        else:
            load_device, load_dtype = "cpu", torch_dtype

        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, load_device, torch_dtype=load_dtype)
        else:
            state_dict = load_state_dict(path, load_dtype, load_device)

        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}

        if quantize.load_prequantized:
            model = quantize.prepare_for_prequantized_load(model, compute_dtype=torch_dtype or torch.bfloat16)
            state_dict = quantize.unflatten_state_dict(state_dict, load_metadata_from_safetensors(path))

        model.load_state_dict(state_dict, assign=True)
        model = quantize.quantize_model(model, compute_device=device, model_device=device)
        model = quantize.dequantize_model(model, compute_dtype=torch_dtype or torch.bfloat16)
        model = model.to(dtype=torch_dtype, device=device)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        # Why does DeepSpeed ZeRO Stage 3 need to be handled separately?
        # Because at this stage, model parameters are partitioned across multiple GPUs.
        # Loading them directly could lead to excessive GPU memory consumption.
        if is_deepspeed_zero3_enabled():
            from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model
            _load_state_dict_into_zero3_model(model, state_dict)
        else:
            model.load_state_dict(state_dict, assign=True)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if quantize is not None:
        # Downstream steps (e.g. LoRA hot-loading) need the config to handle the quantized layers.
        model.quantize_config = quantize
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model


def get_init_context(torch_dtype, device):
    if is_deepspeed_zero3_enabled():
        from transformers.modeling_utils import set_zero3_state
        import deepspeed
        # Why do we use "deepspeed.zero.Init"?
        # Weight segmentation of the model can be performed on the CPU side
        # and loading the segmented weights onto the computing card
        init_contexts = [deepspeed.zero.Init(remote_device=device, dtype=torch_dtype), set_zero3_state()]
    else:
        # Why do we use `skip_model_initialization`?
        # It skips the random initialization of model parameters,
        # thereby speeding up model loading and avoiding excessive memory usage.
        init_contexts = [skip_model_initialization()]

    return init_contexts
