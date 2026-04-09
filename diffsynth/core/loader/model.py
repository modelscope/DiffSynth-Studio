from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import torch
from contextlib import contextmanager
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None, state_dict=None):
    config = {} if config is None else config
    with ContextManagers(get_init_context(torch_dtype=torch_dtype, device=device)):
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
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

            # MODIFY: depth adapter
            if not isinstance(state_dict, dict):
                state_dict = {i: state_dict[i] for i in state_dict}
            if not any("depth_adapter" in k for k in model.state_dict()):
                depth_adapter_keys = [k for k in state_dict if "depth_adapter" in k]
                if len(depth_adapter_keys) > 0:
                    print(f"Dropping depth_adapter params for concat mode: {len(depth_adapter_keys)} keys")
                    for k in depth_adapter_keys:
                        del state_dict[k]
            
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
        

        # MODIFY: depth adapter
        if not isinstance(state_dict, dict):
            state_dict = {i: state_dict[i] for i in state_dict}
        if not any("depth_adapter" in k for k in model.state_dict()):
            depth_adapter_keys = [k for k in state_dict if "depth_adapter" in k]
            if len(depth_adapter_keys) > 0:
                print(f"Dropping depth_adapter params for concat mode: {len(depth_adapter_keys)} keys")
                for k in depth_adapter_keys:
                    del state_dict[k]
        
        # MODIFY: VACE context embedder additional channel
        # Patch for VACE weight mismatch
        for key in ["vace.vace_patch_embedding.weight", "vace_patch_embedding.weight"]:
            if key in state_dict and key in model.state_dict():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    print(f"Patching {key}: {state_dict[key].shape} -> {model.state_dict()[key].shape}")
                    if state_dict[key].shape[1] == 96 and model.state_dict()[key].shape[1] == 112:
                        w = state_dict[key]
                        # Channels 16-32 represent "Reactive"
                        w_depth = w[:, 16:32].clone()
                        state_dict[key] = torch.cat([w, w_depth], dim=1)
                        print("copy channels 16-32 as initial weight of new channels 96-112")
                    elif state_dict[key].shape[1] == 96 and model.state_dict()[key].shape[1] == 128:
                        w = state_dict[key]
                        # Initialize both depth(96-112) and normal(112-128) from reactive channels.
                        w_reactive = w[:, 16:32].clone()
                        state_dict[key] = torch.cat([w, w_reactive, w_reactive], dim=1)
                        print("copy channels 16-32 as initial weight of new channels 96-128")
                    elif state_dict[key].shape[1] == 112 and model.state_dict()[key].shape[1] == 128:
                        w = state_dict[key]
                        # Initialize normal(112-128) from existing depth(96-112) weights.
                        w_depth = w[:, 96:112].clone()
                        state_dict[key] = torch.cat([w, w_depth], dim=1)
                        print("copy channels 96-112 as initial weight of new channels 112-128")
                    elif state_dict[key].shape[1] > model.state_dict()[key].shape[1]:
                        target_dim = model.state_dict()[key].shape[1]
                        state_dict[key] = state_dict[key][:, :target_dim].contiguous()
                        print(f"slice first {target_dim} channels to fit target input dim")
                    elif state_dict[key].shape[1] < model.state_dict()[key].shape[1]:
                        pad_dim = model.state_dict()[key].shape[1] - state_dict[key].shape[1]
                        pad = torch.zeros(
                            (state_dict[key].shape[0], pad_dim, *state_dict[key].shape[2:]),
                            dtype=state_dict[key].dtype,
                            device=state_dict[key].device
                        )
                        state_dict[key] = torch.cat([state_dict[key], pad], dim=1)

        # Patch for newly added VACE depth adapter weights.
        model_sd = model.state_dict()
        missing_depth_adapter_keys = [
            k for k in model_sd
            if "depth_adapter" in k and k not in state_dict
        ]
        if len(missing_depth_adapter_keys) > 0:
            print(f"Initializing missing depth_adapter params: {len(missing_depth_adapter_keys)} keys")
            for key in missing_depth_adapter_keys:
                template = model_sd[key]
                if getattr(template, "is_meta", False):
                    initialized = torch.zeros(template.shape, dtype=template.dtype, device="cpu")
                    if torch.is_floating_point(initialized):
                        if key.endswith("bias"):
                            torch.nn.init.zeros_(initialized)
                        elif initialized.ndim >= 2:
                            torch.nn.init.kaiming_uniform_(initialized, a=5 ** 0.5)
                    state_dict[key] = initialized
                else:
                    state_dict[key] = template.detach().clone().to("cpu")
        
        
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
