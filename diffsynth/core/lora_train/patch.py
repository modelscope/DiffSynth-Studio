import math, re, torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Linear, lora_rank: int, lora_alpha: float = None, dtype: torch.dtype = None, lora_bias: bool = False):
        super().__init__()
        if lora_alpha is None:
            lora_alpha = lora_rank
        self.base_layer = base_layer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        weight = base_layer.weight
        dtype = dtype or (weight.dtype if weight.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32)
        self.lora_A = torch.nn.Linear(base_layer.in_features, lora_rank, bias=False)
        self.lora_B = torch.nn.Linear(lora_rank, base_layer.out_features, bias=lora_bias)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        if lora_bias:
            torch.nn.init.zeros_(self.lora_B.bias)
        self.lora_A.to(device=weight.device, dtype=dtype)
        self.lora_B.to(device=weight.device, dtype=dtype)

    def forward(self, x, *args, **kwargs):
        out = self.base_layer(x, *args, **kwargs)
        out_dtype = out.dtype
        out = out + self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling
        return out.to(out_dtype)


def match_lora_target_module(name, target_modules):
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == target or name.endswith("." + target) for target in target_modules)


def inject_lora_into_model(model: torch.nn.Module, target_modules, lora_rank: int, lora_alpha: float = None, dtype: torch.dtype = None):
    replaced_names, skipped_names = [], []
    for name, module in list(model.named_modules()):
        if name == "" or not match_lora_target_module(name, target_modules):
            continue
        if not isinstance(module, torch.nn.Linear):
            skipped_names.append(f"{name} ({type(module).__name__})")
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name != "" else model
        setattr(parent, child_name, LoRALinear(module, lora_rank, lora_alpha=lora_alpha, dtype=dtype))
        replaced_names.append(name)
    if len(skipped_names) > 0:
        print(f"These matched modules are not `torch.nn.Linear`, so LoRA is not patched on them: {skipped_names}.")
    if len(replaced_names) == 0:
        raise ValueError(f"No `torch.nn.Linear` module matches the LoRA target modules: {target_modules}.")
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    print(f"LoRA is patched on {len(replaced_names)} modules.")
    return model
