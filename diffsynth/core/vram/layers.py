import torch, copy
from typing import Union
from .initialization import skip_model_initialization
from .disk_map import DiskMap


class AutoTorchModule(torch.nn.Module):

    def __init__(
        self,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
    ):
        super().__init__()
        self.set_dtype_and_device(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.state = 0
        self.name = ""

    def set_dtype_and_device(
        self,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
    ):
        self.offload_dtype = offload_dtype or computation_dtype
        self.offload_device = offload_device or computation_dtype
        self.onload_dtype = onload_dtype or computation_dtype
        self.onload_device = onload_device or computation_dtype
        self.preparing_dtype = preparing_dtype or computation_dtype
        self.preparing_device = preparing_device or computation_dtype
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit

    def cast_to(self, weight, dtype, device):
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight)
        return r

    def check_free_vram(self):
        gpu_mem_state = torch.cuda.mem_get_info(self.computation_device)
        used_memory = (gpu_mem_state[1] - gpu_mem_state[0]) / (1024**3)
        return used_memory < self.vram_limit

    def offload(self):
        if self.state != 0:
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state != 1:
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            
    def param_name(self, name):
        if self.name == "":
            return name
        else:
            return self.name + "." + name


class AutoWrappedModule(AutoTorchModule):

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs
    ):
        super().__init__(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.module = module
        if offload_dtype == "disk":
            self.name = name
            self.disk_map = disk_map
            self.required_params = [name for name, _ in self.module.named_parameters()]
            self.disk_offload = True
        else:
            self.disk_offload = False
            
    def load_from_disk(self, torch_dtype, device, copy_module=False):
        if copy_module:
            module = copy.deepcopy(self.module)
        else:
            module = self.module
        state_dict = {}
        for name in self.required_params:
            param = self.disk_map[self.param_name(name)]
            param = param.to(dtype=torch_dtype, device=device)
            state_dict[name] = param
        module.load_state_dict(state_dict, assign=True)
        module.to(dtype=torch_dtype, device=device)
        return module
    
    def offload_to_disk(self, model: torch.nn.Module):
        for buf in model.buffers():
            # If there are some parameters are registed in buffers (not in state dict),
            # We cannot offload the model.
            for children in model.children():
                self.offload_to_disk(children)
            break
        else:
            model.to("meta")

    def offload(self):
        # offload / onload / preparing -> offload
        if self.state != 0:
            if self.disk_offload:
                self.offload_to_disk(self.module)
            else:
                self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        # offload / onload / preparing -> onload
        if self.state < 1:
            if self.disk_offload and self.onload_device != "disk" and self.offload_device == "disk":
                self.load_from_disk(self.onload_dtype, self.onload_device)
            elif self.onload_device != "disk":
                self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            
    def preparing(self):
        # onload / preparing -> preparing
        if self.state != 2:
            if self.disk_offload and self.preparing_device != "disk" and self.onload_device == "disk":
                self.load_from_disk(self.preparing_dtype, self.preparing_device)
            elif self.preparing_device != "disk":
                self.to(dtype=self.preparing_dtype, device=self.preparing_device)
            self.state = 2

    def cast_to(self, module, dtype, device):
        return copy.deepcopy(module).to(dtype=dtype, device=device)
            
    def computation(self):
        # onload / preparing -> computation (temporary)
        if self.state == 2:
            torch_dtype, device = self.preparing_dtype, self.preparing_device
        else:
            torch_dtype, device = self.onload_dtype, self.onload_device
        if torch_dtype == self.computation_dtype and device == self.computation_device:
            module = self.module
        elif self.disk_offload and device == "disk":
            module = self.load_from_disk(self.computation_dtype, self.computation_device, copy_module=True)
        else:
            module = self.cast_to(self.module, dtype=self.computation_dtype, device=self.computation_device)
        return module

    def forward(self, *args, **kwargs):
        if self.state == 1 and (self.vram_limit is None or self.check_free_vram()):
            self.preparing()
        module = self.computation()
        return module(*args, **kwargs)
    
    def __getattr__(self, name):
        if name in self.__dict__ or name == "module":
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)


class AutoWrappedNonRecurseModule(AutoWrappedModule):

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs
    ):
        super().__init__(
            module,
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
            name,
            disk_map,
            **kwargs
        )
        if self.disk_offload:
            self.required_params = [name for name, _ in self.module.named_parameters(recurse=False)]
            
    def load_from_disk(self, torch_dtype, device, copy_module=False):
        if copy_module:
            module = copy.deepcopy(self.module)
        else:
            module = self.module
        state_dict = {}
        for name in self.required_params:
            param = self.disk_map[self.param_name(name)]
            param = param.to(dtype=torch_dtype, device=device)
            state_dict[name] = param
        module.load_state_dict(state_dict, assign=True, strict=False)
        return module
    
    def offload_to_disk(self, model: torch.nn.Module):
        for name in self.required_params:
            getattr(self, name).to("meta")
    
    def cast_to(self, module, dtype, device):
        # Parameter casting is implemented in the model architecture.
        return module
    
    def __getattr__(self, name):
        if name in self.__dict__ or name == "module":
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)


class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype: torch.dtype = None,
        offload_device: Union[str, torch.device] = None,
        onload_dtype: torch.dtype = None,
        onload_device: Union[str, torch.device] = None,
        preparing_dtype: torch.dtype = None,
        preparing_device: Union[str, torch.device] = None,
        computation_dtype: torch.dtype = None,
        computation_device: Union[str, torch.device] = None,
        vram_limit: float = None,
        name: str = "",
        disk_map: DiskMap = None,
        **kwargs
    ):
        with skip_model_initialization():
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        self.set_dtype_and_device(
            offload_dtype,
            offload_device,
            onload_dtype,
            onload_device,
            preparing_dtype,
            preparing_device,
            computation_dtype,
            computation_device,
            vram_limit,
        )
        self.weight = module.weight
        self.bias = module.bias
        self.state = 0
        self.name = name
        self.lora_A_weights = []
        self.lora_B_weights = []
        self.lora_merger = None
        self.enable_fp8 = computation_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
        
        if offload_dtype == "disk":
            self.disk_map = disk_map
            self.disk_offload = True
        else:
            self.disk_offload = False
    
    def fp8_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        device = input.device
        origin_dtype = input.dtype
        origin_shape = input.shape
        input = input.reshape(-1, origin_shape[-1])

        x_max = torch.max(torch.abs(input), dim=-1, keepdim=True).values
        fp8_max = 448.0
        # For float8_e4m3fnuz, the maximum representable value is half of that of e4m3fn.
        # To avoid overflow and ensure numerical compatibility during FP8 computation,
        # we scale down the input by 2.0 in advance.
        # This scaling will be compensated later during the final result scaling.
        if self.computation_dtype == torch.float8_e4m3fnuz:
            fp8_max = fp8_max / 2.0
        scale_a = torch.clamp(x_max / fp8_max, min=1.0).float().to(device=device)
        scale_b = torch.ones((weight.shape[0], 1)).to(device=device)
        input = input / (scale_a + 1e-8)
        input = input.to(self.computation_dtype)
        weight = weight.to(self.computation_dtype)
        bias = bias.to(torch.bfloat16)

        result = torch._scaled_mm(
            input,
            weight.T,
            scale_a=scale_a,
            scale_b=scale_b.T,
            bias=bias,
            out_dtype=origin_dtype,
        )
        new_shape = origin_shape[:-1] + result.shape[-1:]
        result = result.reshape(new_shape)
        return result
            
    def load_from_disk(self, torch_dtype, device, assign=True):
        weight = self.disk_map[self.name + ".weight"].to(dtype=torch_dtype, device=device)
        bias = None if self.bias is None else self.disk_map[self.name + ".bias"].to(dtype=torch_dtype, device=device)
        if assign:
            state_dict = {"weight": weight}
            if bias is not None: state_dict["bias"] = bias
            self.load_state_dict(state_dict, assign=True)
        return weight, bias
    
    def offload(self):
        # offload / onload / preparing -> offload
        if self.state != 0:
            if self.disk_offload:
                self.to("meta")
            else:
                self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        # offload / onload / preparing -> onload
        if self.state < 1:
            if self.disk_offload and self.onload_device != "disk" and self.offload_device == "disk":
                self.load_from_disk(self.onload_dtype, self.onload_device)
            elif self.onload_device != "disk":
                self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            
    def preparing(self):
        # onload / preparing -> preparing
        if self.state != 2:
            if self.disk_offload and self.preparing_device != "disk" and self.onload_device == "disk":
                self.load_from_disk(self.preparing_dtype, self.preparing_device)
            elif self.preparing_device != "disk":
                self.to(dtype=self.preparing_dtype, device=self.preparing_device)
            self.state = 2
            
    def computation(self):
        # onload / preparing -> computation (temporary)
        if self.state == 2:
            torch_dtype, device = self.preparing_dtype, self.preparing_device
        else:
            torch_dtype, device = self.onload_dtype, self.onload_device
        if torch_dtype == self.computation_dtype and device == self.computation_device:
            weight, bias = self.weight, self.bias
        elif self.disk_offload and device == "disk":
            weight, bias = self.load_from_disk(self.computation_dtype, self.computation_device, assign=False)
        else:
            weight = self.cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else self.cast_to(self.bias, self.computation_dtype, self.computation_device)
        return weight, bias

    def linear_forward(self, x, weight, bias):
        if self.enable_fp8:
            out = self.fp8_linear(x, weight, bias)
        else:
            out = torch.nn.functional.linear(x, weight, bias)
        return out

    def lora_forward(self, x, out):
        if self.lora_merger is None:
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                out = out + x @ lora_A.T @ lora_B.T
        else:
            lora_output = []
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                lora_output.append(x @ lora_A.T @ lora_B.T)
            lora_output = torch.stack(lora_output)
            out = self.lora_merger(out, lora_output)
        return out
    
    def forward(self, x, *args, **kwargs):
        if self.state == 1 and (self.vram_limit is None or self.check_free_vram()):
            self.preparing()
        weight, bias = self.computation()
        out = self.linear_forward(x, weight, bias)
        if len(self.lora_A_weights) > 0:
            out = self.lora_forward(x, out)
        return out


def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, vram_config: dict, vram_limit=None, name_prefix="", disk_map=None, **kwargs):
    if isinstance(model, AutoWrappedNonRecurseModule):
        model = model.module
    for name, module in model.named_children():
        layer_name = name if name_prefix == "" else name_prefix + "." + name
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                module_ = target_module(module, **vram_config, vram_limit=vram_limit, name=layer_name, disk_map=disk_map, **kwargs)
                if isinstance(module_, AutoWrappedNonRecurseModule):
                    enable_vram_management_recursively(module_, module_map, vram_config, vram_limit=vram_limit, name_prefix=layer_name, disk_map=disk_map, **kwargs)
                setattr(model, name, module_)
                break
        else:
            enable_vram_management_recursively(module, module_map, vram_config, vram_limit=vram_limit, name_prefix=layer_name, disk_map=disk_map, **kwargs)


def fill_vram_config(model, vram_config):
    vram_config_ = vram_config.copy()
    vram_config_["onload_dtype"] = vram_config["computation_dtype"]
    vram_config_["onload_device"] = vram_config["computation_device"]
    vram_config_["preparing_dtype"] = vram_config["computation_dtype"]
    vram_config_["preparing_device"] = vram_config["computation_device"]
    for k in vram_config:
        if vram_config[k] != vram_config_[k]:
            print(f"No fine-grained VRAM configuration is provided for {model.__class__.__name__}. [`onload`, `preparing`, `computation`] will be the same state. `vram_config` is set to {vram_config_}")
            break
    return vram_config_


def enable_vram_management(model: torch.nn.Module, module_map: dict, vram_config: dict, vram_limit=None, disk_map=None, **kwargs):
    for source_module, target_module in module_map.items():
        # If no fine-grained VRAM configuration is provided, the entire model will be managed uniformly.
        if isinstance(model, source_module):
            vram_config = fill_vram_config(model, vram_config)
            model = target_module(model, **vram_config, vram_limit=vram_limit, disk_map=disk_map, **kwargs)
            break
    else:
        enable_vram_management_recursively(model, module_map, vram_config, vram_limit=vram_limit, disk_map=disk_map, **kwargs)
    # `vram_management_enabled` is a flag that allows the pipeline to determine whether VRAM management is enabled.
    model.vram_management_enabled = True
    return model
