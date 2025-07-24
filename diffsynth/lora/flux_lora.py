import torch, math
from . import GeneralLoRALoader
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from typing import Union


class FluxLoRALoader(GeneralLoRALoader):
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        super().__init__(device=device, torch_dtype=torch_dtype)
    
        self.diffusers_rename_dict = {
            "transformer.single_transformer_blocks.blockid.attn.to_k.lora_A.weight":"single_blocks.blockid.a_to_k.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.attn.to_k.lora_B.weight":"single_blocks.blockid.a_to_k.lora_B.default.weight",
            "transformer.single_transformer_blocks.blockid.attn.to_q.lora_A.weight":"single_blocks.blockid.a_to_q.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.attn.to_q.lora_B.weight":"single_blocks.blockid.a_to_q.lora_B.default.weight",
            "transformer.single_transformer_blocks.blockid.attn.to_v.lora_A.weight":"single_blocks.blockid.a_to_v.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.attn.to_v.lora_B.weight":"single_blocks.blockid.a_to_v.lora_B.default.weight",
            "transformer.single_transformer_blocks.blockid.norm.linear.lora_A.weight":"single_blocks.blockid.norm.linear.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.norm.linear.lora_B.weight":"single_blocks.blockid.norm.linear.lora_B.default.weight",
            "transformer.single_transformer_blocks.blockid.proj_mlp.lora_A.weight":"single_blocks.blockid.proj_in_besides_attn.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.proj_mlp.lora_B.weight":"single_blocks.blockid.proj_in_besides_attn.lora_B.default.weight",
            "transformer.single_transformer_blocks.blockid.proj_out.lora_A.weight":"single_blocks.blockid.proj_out.lora_A.default.weight",
            "transformer.single_transformer_blocks.blockid.proj_out.lora_B.weight":"single_blocks.blockid.proj_out.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_k_proj.lora_A.weight":"blocks.blockid.attn.b_to_k.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_k_proj.lora_B.weight":"blocks.blockid.attn.b_to_k.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_q_proj.lora_A.weight":"blocks.blockid.attn.b_to_q.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_q_proj.lora_B.weight":"blocks.blockid.attn.b_to_q.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_v_proj.lora_A.weight":"blocks.blockid.attn.b_to_v.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.add_v_proj.lora_B.weight":"blocks.blockid.attn.b_to_v.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_add_out.lora_A.weight":"blocks.blockid.attn.b_to_out.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_add_out.lora_B.weight":"blocks.blockid.attn.b_to_out.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_k.lora_A.weight":"blocks.blockid.attn.a_to_k.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_k.lora_B.weight":"blocks.blockid.attn.a_to_k.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_out.0.lora_A.weight":"blocks.blockid.attn.a_to_out.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_out.0.lora_B.weight":"blocks.blockid.attn.a_to_out.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_q.lora_A.weight":"blocks.blockid.attn.a_to_q.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_q.lora_B.weight":"blocks.blockid.attn.a_to_q.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_v.lora_A.weight":"blocks.blockid.attn.a_to_v.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.attn.to_v.lora_B.weight":"blocks.blockid.attn.a_to_v.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.ff.net.0.proj.lora_A.weight":"blocks.blockid.ff_a.0.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.ff.net.0.proj.lora_B.weight":"blocks.blockid.ff_a.0.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.ff.net.2.lora_A.weight":"blocks.blockid.ff_a.2.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.ff.net.2.lora_B.weight":"blocks.blockid.ff_a.2.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.ff_context.net.0.proj.lora_A.weight":"blocks.blockid.ff_b.0.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.ff_context.net.0.proj.lora_B.weight":"blocks.blockid.ff_b.0.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.ff_context.net.2.lora_A.weight":"blocks.blockid.ff_b.2.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.ff_context.net.2.lora_B.weight":"blocks.blockid.ff_b.2.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.norm1.linear.lora_A.weight":"blocks.blockid.norm1_a.linear.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.norm1.linear.lora_B.weight":"blocks.blockid.norm1_a.linear.lora_B.default.weight",
            "transformer.transformer_blocks.blockid.norm1_context.linear.lora_A.weight":"blocks.blockid.norm1_b.linear.lora_A.default.weight",
            "transformer.transformer_blocks.blockid.norm1_context.linear.lora_B.weight":"blocks.blockid.norm1_b.linear.lora_B.default.weight",
        }

        self.civitai_rename_dict = {
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

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        super().load(model, state_dict_lora, alpha)

    
    def convert_state_dict(self,state_dict):

        def guess_block_id(name,model_resource):
            if model_resource == 'civitai':
                names = name.split("_")
                for i in names:
                    if i.isdigit():
                        return i, name.replace(f"_{i}_", "_blockid_")
            if model_resource == 'diffusers':
                names = name.split(".")
                for i in names:
                    if i.isdigit():
                        return i, name.replace(f"transformer_blocks.{i}.", "transformer_blocks.blockid.")
            return None, None

        def guess_resource(state_dict):
            for k in state_dict:
                if "lora_unet_" in k:
                    return 'civitai'
                elif k.startswith("transformer."):
                    return 'diffusers'
                else:
                    None
        
        model_resource = guess_resource(state_dict)
        if model_resource is None:
            return state_dict

        rename_dict = self.diffusers_rename_dict if model_resource == 'diffusers' else self.civitai_rename_dict
        def guess_alpha(state_dict):
                for name, param in state_dict.items():
                    if ".alpha" in name:
                        for suffix in [".lora_down.weight", ".lora_A.weight"]:
                            name_ = name.replace(".alpha", suffix)
                            if name_ in state_dict:
                                lora_alpha = param.item() / state_dict[name_].shape[0]
                                lora_alpha = math.sqrt(lora_alpha)
                                return lora_alpha

                return 1
        
        alpha = guess_alpha(state_dict)
        
        state_dict_ = {}
        for name, param in state_dict.items():
            block_id, source_name = guess_block_id(name,model_resource)
            if alpha != 1:
                param *= alpha
            if source_name in rename_dict:
                target_name = rename_dict[source_name]
                target_name = target_name.replace(".blockid.", f".{block_id}.")
                state_dict_[target_name] = param
            else:
                state_dict_[name] = param
        
        if model_resource == 'diffusers':
            for name in list(state_dict_.keys()):
                if "single_blocks." in name and ".a_to_q." in name:
                    mlp = state_dict_.get(name.replace(".a_to_q.", ".proj_in_besides_attn."), None)
                    if mlp is None:
                        dim = 4
                        if 'lora_A' in name:
                            dim = 1
                        mlp = torch.zeros(dim * state_dict_[name].shape[0],
                                        *state_dict_[name].shape[1:],
                                        dtype=state_dict_[name].dtype)
                    else:
                        state_dict_.pop(name.replace(".a_to_q.", ".proj_in_besides_attn."))
                    if 'lora_A' in name:
                        param = torch.concat([
                            state_dict_.pop(name),
                            state_dict_.pop(name.replace(".a_to_q.", ".a_to_k.")),
                            state_dict_.pop(name.replace(".a_to_q.", ".a_to_v.")),
                            mlp,
                        ], dim=0)
                    elif 'lora_B' in name:
                        d, r = state_dict_[name].shape
                        param = torch.zeros((3*d+mlp.shape[0], 3*r+mlp.shape[1]), dtype=state_dict_[name].dtype, device=state_dict_[name].device)
                        param[:d, :r] = state_dict_.pop(name)
                        param[d:2*d, r:2*r] = state_dict_.pop(name.replace(".a_to_q.", ".a_to_k."))
                        param[2*d:3*d, 2*r:3*r] = state_dict_.pop(name.replace(".a_to_q.", ".a_to_v."))
                        param[3*d:, 3*r:] = mlp
                    else:
                        param = torch.concat([
                            state_dict_.pop(name),
                            state_dict_.pop(name.replace(".a_to_q.", ".a_to_k.")),
                            state_dict_.pop(name.replace(".a_to_q.", ".a_to_v.")),
                            mlp,
                        ], dim=0)
                    name_ = name.replace(".a_to_q.", ".to_qkv_mlp.")
                    state_dict_[name_] = param
            for name in list(state_dict_.keys()):
                for component in ["a", "b"]:
                    if f".{component}_to_q." in name:
                        name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                        concat_dim = 0
                        if 'lora_A' in name:
                            param = torch.concat([
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                            ], dim=0)
                        elif 'lora_B' in name:
                            origin = state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")]
                            d, r = origin.shape
                            # print(d, r)
                            param = torch.zeros((3*d, 3*r), dtype=origin.dtype, device=origin.device)
                            param[:d, :r] = state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")]
                            param[d:2*d, r:2*r] = state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")]
                            param[2*d:3*d, 2*r:3*r] = state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")]
                        else:
                            param = torch.concat([
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                                state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                            ], dim=0)
                        state_dict_[name_] = param
                        state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                        state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                        state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))  
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


class FluxLoRAFuser:
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        
    def Matrix_Decomposition_lowrank(self, A, k):
        U, S, V = torch.svd_lowrank(A.float(), q=k)
        S_k = torch.diag(S[:k])
        U_hat = U @ S_k
        return U_hat, V.t()

    def LoRA_State_Dicts_Decomposition(self, lora_state_dicts=[], q=4):
        lora_1 = lora_state_dicts[0]
        state_dict_ = {}
        for k,v in lora_1.items():
            if 'lora_A.' in k:
                lora_B_name = k.replace('lora_A.', 'lora_B.')
                lora_B = lora_1[lora_B_name]
                weight = torch.mm(lora_B, v)
                for lora_dict in lora_state_dicts[1:]:
                    lora_A_ = lora_dict[k]
                    lora_B_ = lora_dict[lora_B_name]
                    weight_ = torch.mm(lora_B_, lora_A_)
                    weight += weight_
                new_B, new_A = self.Matrix_Decomposition_lowrank(weight, q)
                state_dict_[lora_B_name] = new_B.to(dtype=torch.bfloat16)
                state_dict_[k] = new_A.to(dtype=torch.bfloat16)
        return state_dict_
        
    def __call__(self, lora_configs: list[Union[ModelConfig, str]]):
        loras = []
        loader = FluxLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        for lora_config in lora_configs:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
            lora = loader.convert_state_dict(lora)
            loras.append(lora)
        lora = self.LoRA_State_Dicts_Decomposition(loras)
        return lora
