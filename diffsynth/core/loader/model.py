from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict, load_metadata_from_safetensors
import torch
from contextlib import contextmanager
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers


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
            metadata = load_metadata_from_safetensors(path[0] if isinstance(path, list) else path)
            model = quantize.prepare_for_prequantized_load(model, compute_dtype=vram_config["computation_dtype"])
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit, quantize=quantize, metadata=metadata)
        else:
            offload_device = vram_config["offload_device"]
            computation_device = vram_config["computation_device"]
            computation_dtype = vram_config["computation_dtype"]
            offload_dtype = vram_config["offload_dtype"]
            load_dtype = None if quantize.load_prequantized else computation_dtype
            if state_dict is None: state_dict = DiskMap(path, offload_device, torch_dtype=load_dtype)
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
            model = model.to(dtype=offload_dtype, device=offload_device)
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
