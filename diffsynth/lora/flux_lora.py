import torch
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.lora import FluxLoRAFromCivitai


class FluxLoRALoader(GeneralLoRALoader):
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.loader = FluxLoRAFromCivitai()

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        lora_prefix, model_resource = self.loader.match(model, state_dict_lora)
        self.loader.load(model, state_dict_lora, lora_prefix, alpha=alpha, model_resource=model_resource)

class LoraMerger(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_base = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_lora = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_cross = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_out = torch.nn.Parameter(torch.ones((dim,)))
        self.bias = torch.nn.Parameter(torch.randn((dim,)))
        self.activation = torch.nn.Sigmoid()
        self.norm_base = torch.nn.LayerNorm(dim, eps=1e-5)
        self.norm_lora = torch.nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, base_output, lora_outputs):
        norm_base_output = self.norm_base(base_output)
        norm_lora_outputs = self.norm_lora(lora_outputs)
        gate = self.activation(
            norm_base_output * self.weight_base \
            + norm_lora_outputs * self.weight_lora \
            + norm_base_output * norm_lora_outputs * self.weight_cross + self.bias
        )
        output = base_output + (self.weight_out * gate * lora_outputs).sum(dim=0)
        return output

class LoraPatcher(torch.nn.Module):
    def __init__(self, lora_patterns=None):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"]
            model_dict[name.replace(".", "___")] = LoraMerger(dim)
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
    def default_lora_patterns(self):
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": 9216, "attn.a_to_out": 3072, "ff_a.0": 12288, "ff_a.2": 3072, "norm1_a.linear": 18432,
            "attn.b_to_qkv": 9216, "attn.b_to_out": 3072, "ff_b.0": 12288, "ff_b.2": 3072, "norm1_b.linear": 18432,
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        lora_dict = {"to_qkv_mlp": 21504, "proj_out": 3072, "norm.linear": 9216}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        return lora_patterns
        
    def forward(self, base_output, lora_outputs, name):
        return self.model_dict[name.replace(".", "___")](base_output, lora_outputs)
