from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict, load_state_dict_metadata
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
    if module_map is not None:
        if quantize is not None:
            raise NotImplementedError("Quantization combined with VRAM offload is not supported yet. Please remove `offload_dtype`/`offload_device` from the quantized ModelConfig for now.")
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
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        # When online quantization is requested, keep the fp weights on CPU so that
        # `quantize_model_weights` can stream them to the target device layer by layer.
        online_quantize = quantize is not None and not quantize.load_prequantized
        load_device = "cpu" if online_quantize else device
        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, load_device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, load_device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        if quantize is not None and quantize.load_prequantized:
            from ..quant import replace_linear_for_quantized_load
            model = replace_linear_for_quantized_load(model, quantize)
            backend, _ = quantize.resolve()
            state_dict = backend.unflatten_state_dict(state_dict, load_state_dict_metadata(path))
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
        if torch_dtype is not None and not (quantize is not None and quantize.load_prequantized):
            # Preserve quantized weights
            model = model.to(dtype=torch_dtype, device=load_device)
        else:
            # Device-only move: packed quantized weights must not be dtype-cast.
            model = model.to(device=load_device)
        # Online weight-only quantization of the fp model (see `diffsynth.core.quant`).
        if online_quantize:
            from ..quant import quantize_model_weights
            model = quantize_model_weights(model, quantize, default_compute_dtype=torch_dtype, device=device)
        elif quantize is not None and quantize.mode == "dequant_once":
            # The checkpoint was quantized; run at full precision from here on.
            from ..quant import dequantize_model_weights
            model = dequantize_model_weights(model, quantize, default_compute_dtype=torch_dtype)
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
