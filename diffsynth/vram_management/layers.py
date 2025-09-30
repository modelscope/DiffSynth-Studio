import torch, copy
from ..models.utils import init_weights_on_device


def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoTorchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def check_free_vram(self):
        gpu_mem_state = torch.cuda.mem_get_info(self.computation_device)
        used_memory = (gpu_mem_state[1] - gpu_mem_state[0]) / (1024 ** 3)
        return used_memory < self.vram_limit

    def offload(self):
        if self.state != 0:
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state != 1:
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1
            
    def keep(self):
        if self.state != 2:
            self.to(dtype=self.computation_dtype, device=self.computation_device)
            self.state = 2


class AutoWrappedModule(AutoTorchModule):
    def __init__(self, module: torch.nn.Module, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device, vram_limit, **kwargs):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0

    def forward(self, *args, **kwargs):
        if self.state == 2:
            module = self.module
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                module = self.module
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                module = self.module
            else:
                module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)
    

class WanAutoCastLayerNorm(torch.nn.LayerNorm, AutoTorchModule):
    def __init__(self, module: torch.nn.LayerNorm, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device, vram_limit, **kwargs):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine, bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0

    def forward(self, x, *args, **kwargs):
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = None if self.weight is None else cast_to(self.weight, self.computation_dtype, self.computation_device)
                bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        with torch.amp.autocast(device_type=x.device.type):
            x = torch.nn.functional.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)
        return x
    

class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device, vram_limit, name="", **kwargs):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0
        self.name = name
        self.lora_A_weights = []
        self.lora_B_weights = []
        self.lora_merger = None
        self.enable_fp8 = computation_dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]
        
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

    def forward(self, x, *args, **kwargs):
        # VRAM management
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
                bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        
        # Linear forward
        if self.enable_fp8:
            out = self.fp8_linear(x, weight, bias)
        else:
            out = torch.nn.functional.linear(x, weight, bias)
        
        # LoRA
        if len(self.lora_A_weights) == 0:
            # No LoRA
            return out
        elif self.lora_merger is None:
            # Native LoRA inference
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                out = out + x @ lora_A.T @ lora_B.T
        else:
            # LoRA fusion
            lora_output = []
            for lora_A, lora_B in zip(self.lora_A_weights, self.lora_B_weights):
                lora_output.append(x @ lora_A.T @ lora_B.T)
            lora_output = torch.stack(lora_output)
            out = self.lora_merger(out, lora_output)
        return out


def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, total_num_param=0, vram_limit=None, name_prefix=""):
    for name, module in model.named_children():
        layer_name = name if name_prefix == "" else name_prefix + "." + name
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_, vram_limit=vram_limit, name=layer_name)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(module, module_map, module_config, max_num_param, overflow_module_config, total_num_param, vram_limit=vram_limit, name_prefix=layer_name)
    return total_num_param


def enable_vram_management(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, vram_limit=None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0, vram_limit=vram_limit)
    model.vram_management_enabled = True

