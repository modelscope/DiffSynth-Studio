import torch, copy
from ..models.utils import init_weights_on_device


def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)
    

class AutoWrappedLinear(torch.nn.Linear):
    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
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
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)

class AutoLoRALinear(torch.nn.Linear):
    def __init__(self, name='', in_features=1, out_features=2, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.name = name
    
    def forward(self, x, lora_state_dicts=[], lora_alpahs=[1.0,1.0], **kwargs):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'

        for i, lora_state_dict in enumerate(lora_state_dicts):
            if lora_state_dict is None:
                break
            if lora_a_name in lora_state_dict and lora_b_name in lora_state_dict:
                lora_A = lora_state_dict[lora_a_name].to(dtype=self.weight.dtype,device=self.weight.device)
                lora_B = lora_state_dict[lora_b_name].to(dtype=self.weight.dtype,device=self.weight.device)
                out_lora = x @ lora_A.T @ lora_B.T
                out = out + out_lora * lora_alpahs[i]
        return out

def enable_auto_lora(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module,targets[1]):
            # print(full_name)
            # print(module)
            # ToDo: replace the linear to the AutoLoRALinear 
            new_module = AutoLoRALinear(
                name=full_name, 
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None, 
                device=module.weight.device, 
                dtype=module.weight.dtype)
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            enable_auto_lora(module, module_map, full_name)
       

def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, total_num_param=0):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(module, module_map, module_config, max_num_param, overflow_module_config, total_num_param)
    return total_num_param


def enable_vram_management(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0)
    model.vram_management_enabled = True

