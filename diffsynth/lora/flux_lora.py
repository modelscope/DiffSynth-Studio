import torch, math
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.lora import FluxLoRAFromCivitai


class FluxLoRALoader(GeneralLoRALoader):
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        super().__init__(device=device, torch_dtype=torch_dtype)

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        super().load(model, state_dict_lora, alpha)
        
    def convert_state_dict(self, state_dict):
        # TODO: support other lora format
        rename_dict = {
            "lora_unet_double_blocks_blockid_img_mod_lin.lora_down.weight": "blocks.blockid.norm1_a.linear.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mod_lin.lora_up.weight": "blocks.blockid.norm1_a.linear.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mod_lin.lora_down.weight": "blocks.blockid.norm1_b.linear.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mod_lin.lora_up.weight": "blocks.blockid.norm1_b.linear.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_qkv.lora_down.weight": "blocks.blockid.attn.a_to_qkv.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_qkv.lora_up.weight": "blocks.blockid.attn.a_to_qkv.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_qkv.lora_down.weight": "blocks.blockid.attn.b_to_qkv.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_qkv.lora_up.weight": "blocks.blockid.attn.b_to_qkv.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_proj.lora_down.weight": "blocks.blockid.attn.a_to_out.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_proj.lora_up.weight": "blocks.blockid.attn.a_to_out.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_proj.lora_down.weight": "blocks.blockid.attn.b_to_out.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_proj.lora_up.weight": "blocks.blockid.attn.b_to_out.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_0.lora_down.weight": "blocks.blockid.ff_a.0.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_0.lora_up.weight": "blocks.blockid.ff_a.0.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_2.lora_down.weight": "blocks.blockid.ff_a.2.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_2.lora_up.weight": "blocks.blockid.ff_a.2.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_0.lora_down.weight": "blocks.blockid.ff_b.0.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_0.lora_up.weight": "blocks.blockid.ff_b.0.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_2.lora_down.weight": "blocks.blockid.ff_b.2.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_2.lora_up.weight": "blocks.blockid.ff_b.2.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_modulation_lin.lora_down.weight": "single_blocks.blockid.norm.linear.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_modulation_lin.lora_up.weight": "single_blocks.blockid.norm.linear.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_linear1.lora_down.weight": "single_blocks.blockid.to_qkv_mlp.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_linear1.lora_up.weight": "single_blocks.blockid.to_qkv_mlp.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_linear2.lora_down.weight": "single_blocks.blockid.proj_out.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_linear2.lora_up.weight": "single_blocks.blockid.proj_out.lora_B.default.weight",
        }
        def guess_block_id(name):
            names = name.split("_")
            for i in names:
                if i.isdigit():
                    return i, name.replace(f"_{i}_", "_blockid_")
            return None, None
        def guess_alpha(state_dict):
            for name, param in state_dict.items():
                if ".alpha" in name:
                    name_ = name.replace(".alpha", ".lora_down.weight")
                    if name_ in state_dict:
                        lora_alpha = param.item() / state_dict[name_].shape[0]
                        lora_alpha = math.sqrt(lora_alpha)
                        return lora_alpha
            return 1
        alpha = guess_alpha(state_dict)
        state_dict_ = {}
        for name, param in state_dict.items():
            block_id, source_name = guess_block_id(name)
            if alpha != 1:
                param *= alpha
            if source_name in rename_dict:
                target_name = rename_dict[source_name]
                target_name = target_name.replace(".blockid.", f".{block_id}.")
                state_dict_[target_name] = param
            else:
                state_dict_[name] = param
        return state_dict_


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


class FluxLoraPatcher(torch.nn.Module):
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
    
    @staticmethod
    def state_dict_converter():
        return FluxLoraPatcherStateDictConverter()
    

class FluxLoraPatcherStateDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        return state_dict
