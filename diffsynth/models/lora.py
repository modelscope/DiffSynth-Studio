import torch
from .sd_unet import SDUNet
from .sdxl_unet import SDXLUNet
from .sd_text_encoder import SDTextEncoder
from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sd3_dit import SD3DiT
from .flux_dit import FluxDiT
from .hunyuan_dit import HunyuanDiT
from .cog_dit import CogDiT
from .hunyuan_video_dit import HunyuanVideoDiT
from .wan_video_dit import WanModel



class LoRAFromCivitai:
    def __init__(self):
        self.supported_model_classes = []
        self.lora_prefix = []
        self.renamed_lora_prefix = {}
        self.special_keys = {}


    def convert_state_dict(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        for key in state_dict:
            if ".lora_up" in key:
                return self.convert_state_dict_up_down(state_dict, lora_prefix, alpha)
        return self.convert_state_dict_AB(state_dict, lora_prefix, alpha)


    def convert_state_dict_up_down(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        renamed_lora_prefix = self.renamed_lora_prefix.get(lora_prefix, "")
        state_dict_ = {}
        for key in state_dict:
            if ".lora_up" not in key:
                continue
            if not key.startswith(lora_prefix):
                continue
            weight_up = state_dict[key].to(device="cuda", dtype=torch.float16)
            weight_down = state_dict[key.replace(".lora_up", ".lora_down")].to(device="cuda", dtype=torch.float16)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2).to(torch.float32)
                weight_down = weight_down.squeeze(3).squeeze(2).to(torch.float32)
                lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_up, weight_down)
            target_name = key.split(".")[0].replace(lora_prefix, renamed_lora_prefix).replace("_", ".") + ".weight"
            for special_key in self.special_keys:
                target_name = target_name.replace(special_key, self.special_keys[special_key])
            state_dict_[target_name] = lora_weight.cpu()
        return state_dict_
    

    def convert_state_dict_AB(self, state_dict, lora_prefix="", alpha=1.0, device="cuda", torch_dtype=torch.float16):
        state_dict_ = {}
        for key in state_dict:
            if ".lora_B." not in key:
                continue
            if not key.startswith(lora_prefix):
                continue
            weight_up = state_dict[key].to(device=device, dtype=torch_dtype)
            weight_down = state_dict[key.replace(".lora_B.", ".lora_A.")].to(device=device, dtype=torch_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_up, weight_down)
            keys = key.split(".")
            keys.pop(keys.index("lora_B"))
            target_name = ".".join(keys)
            target_name = target_name[len(lora_prefix):]
            state_dict_[target_name] = lora_weight.cpu()
        return state_dict_
    

    def load(self, model, state_dict_lora, lora_prefix, alpha=1.0, model_resource=None):
        state_dict_model = model.state_dict()
        state_dict_lora = self.convert_state_dict(state_dict_lora, lora_prefix=lora_prefix, alpha=alpha)
        if model_resource == "diffusers":
            state_dict_lora = model.__class__.state_dict_converter().from_diffusers(state_dict_lora)
        elif model_resource == "civitai":
            state_dict_lora = model.__class__.state_dict_converter().from_civitai(state_dict_lora)
        if isinstance(state_dict_lora, tuple):
            state_dict_lora = state_dict_lora[0]
        if len(state_dict_lora) > 0:
            print(f"    {len(state_dict_lora)} tensors are updated.")
            for name in state_dict_lora:
                fp8=False
                if state_dict_model[name].dtype == torch.float8_e4m3fn:
                    state_dict_model[name]= state_dict_model[name].to(state_dict_lora[name].dtype)
                    fp8=True
                state_dict_model[name] += state_dict_lora[name].to(
                    dtype=state_dict_model[name].dtype, device=state_dict_model[name].device)
                if fp8:
                    state_dict_model[name] = state_dict_model[name].to(torch.float8_e4m3fn)
            model.load_state_dict(state_dict_model)
    

    def match(self, model, state_dict_lora):
        for lora_prefix, model_class in zip(self.lora_prefix, self.supported_model_classes):
            if not isinstance(model, model_class):
                continue
            state_dict_model = model.state_dict()
            for model_resource in ["diffusers", "civitai"]:
                try:
                    state_dict_lora_ = self.convert_state_dict(state_dict_lora, lora_prefix=lora_prefix, alpha=1.0)
                    converter_fn = model.__class__.state_dict_converter().from_diffusers if model_resource == "diffusers" \
                        else model.__class__.state_dict_converter().from_civitai
                    state_dict_lora_ = converter_fn(state_dict_lora_)
                    if isinstance(state_dict_lora_, tuple):
                        state_dict_lora_ = state_dict_lora_[0]
                    if len(state_dict_lora_) == 0:
                        continue
                    for name in state_dict_lora_:
                        if name not in state_dict_model:
                            break
                    else:
                        return lora_prefix, model_resource
                except:
                    pass
        return None



class SDLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [SDUNet, SDTextEncoder]
        self.lora_prefix = ["lora_unet_", "lora_te_"]
        self.special_keys = {
            "down.blocks": "down_blocks",
            "up.blocks": "up_blocks",
            "mid.block": "mid_block",
            "proj.in": "proj_in",
            "proj.out": "proj_out",
            "transformer.blocks": "transformer_blocks",
            "to.q": "to_q",
            "to.k": "to_k",
            "to.v": "to_v",
            "to.out": "to_out",
            "text.model": "text_model",
            "self.attn.q.proj": "self_attn.q_proj",
            "self.attn.k.proj": "self_attn.k_proj",
            "self.attn.v.proj": "self_attn.v_proj",
            "self.attn.out.proj": "self_attn.out_proj",
            "input.blocks": "model.diffusion_model.input_blocks",
            "middle.block": "model.diffusion_model.middle_block",
            "output.blocks": "model.diffusion_model.output_blocks",
        }


class SDXLLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [SDXLUNet, SDXLTextEncoder, SDXLTextEncoder2]
        self.lora_prefix = ["lora_unet_", "lora_te1_", "lora_te2_"]
        self.renamed_lora_prefix = {"lora_te2_": "2"}
        self.special_keys = {
            "down.blocks": "down_blocks",
            "up.blocks": "up_blocks",
            "mid.block": "mid_block",
            "proj.in": "proj_in",
            "proj.out": "proj_out",
            "transformer.blocks": "transformer_blocks",
            "to.q": "to_q",
            "to.k": "to_k",
            "to.v": "to_v",
            "to.out": "to_out",
            "text.model": "conditioner.embedders.0.transformer.text_model",
            "self.attn.q.proj": "self_attn.q_proj",
            "self.attn.k.proj": "self_attn.k_proj",
            "self.attn.v.proj": "self_attn.v_proj",
            "self.attn.out.proj": "self_attn.out_proj",
            "input.blocks": "model.diffusion_model.input_blocks",
            "middle.block": "model.diffusion_model.middle_block",
            "output.blocks": "model.diffusion_model.output_blocks",
            "2conditioner.embedders.0.transformer.text_model.encoder.layers": "text_model.encoder.layers"
        }
        

class FluxLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [FluxDiT, FluxDiT]
        self.lora_prefix = ["lora_unet_", "transformer."]
        self.renamed_lora_prefix = {}
        self.special_keys = {
            "single.blocks": "single_blocks",
            "double.blocks": "double_blocks",
            "img.attn": "img_attn",
            "img.mlp": "img_mlp",
            "img.mod": "img_mod",
            "txt.attn": "txt_attn",
            "txt.mlp": "txt_mlp",
            "txt.mod": "txt_mod",
        }


class GeneralLoRAFromPeft:
    def __init__(self):
        self.supported_model_classes = [SDUNet, SDXLUNet, SD3DiT, HunyuanDiT, FluxDiT, CogDiT, WanModel]

    def fetch_device_dtype_from_state_dict(self, target_param):
        """Get device and dtype from a parameter"""
        return target_param.device, target_param.dtype

    def _get_target_name(self, key):
        """Extract target parameter name from LoRA key"""
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        target_name = ".".join(keys)
        if target_name.startswith("diffusion_model."):
            target_name = target_name[len("diffusion_model."):]
        return target_name

    def convert_state_dict(self, state_dict, alpha=1.0, target_state_dict={}):
        """Original method kept for compatibility with match method"""
        device, torch_dtype = None, None
        for name, param in target_state_dict.items():
            device, torch_dtype = param.device, param.dtype
            break
            
        if torch_dtype == torch.float8_e4m3fn:
            torch_dtype = torch.float32
            
        state_dict_ = {}
        for key in state_dict:
            if ".lora_B." not in key:
                continue
                
            weight_up = state_dict[key].to(device=device, dtype=torch_dtype)
            weight_down = state_dict[key.replace(".lora_B.", ".lora_A.")].to(device=device, dtype=torch_dtype)
            
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_up, weight_down)
                
            target_name = self._get_target_name(key)
            
            if target_name not in target_state_dict:
                return {}
                
            state_dict_[target_name] = lora_weight.cpu()
            
        return state_dict_

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        """Apply LoRA weights directly to model parameters without loading entire state dict"""
        # Create parameter name mapping for faster lookup
        param_dict = {}
        for name, param in model.named_parameters():
            param_dict[name] = param
        
        # Process each LoRA parameter pair
        modified_count = 0
        for key in state_dict_lora:
            if ".lora_B." not in key:
                continue
            
            # Get target parameter name and make sure the parameter exists
            target_name = self._get_target_name(key)
            if target_name not in param_dict:
                continue
            
            # Get the target parameter
            param = param_dict[target_name]
            
            # Calculate LoRA weight update
            device, dtype = param.device, param.dtype
            dtype_for_calc = torch.float32 if dtype == torch.float8_e4m3fn else dtype
            
            # Process weights and calculate LoRA update
            weight_b = state_dict_lora[key].to(device=device, dtype=dtype_for_calc)
            weight_a = state_dict_lora[key.replace(".lora_B.", ".lora_A.")].to(device=device, dtype=dtype_for_calc)
            
            if len(weight_b.shape) == 4:
                weight_b = weight_b.squeeze(3).squeeze(2)
                weight_a = weight_a.squeeze(3).squeeze(2)
                lora_weight = alpha * torch.mm(weight_b, weight_a).unsqueeze(2).unsqueeze(3)
            else:
                lora_weight = alpha * torch.mm(weight_b, weight_a)
            
            # Apply update to parameter
            if dtype == torch.float8_e4m3fn:
                param_float = param.to(torch.float32)
                param.data = (param_float + lora_weight).to(dtype)
                del param_float
            else:
                param.data += lora_weight
            
            # Clean up temporary tensors
            del weight_a, weight_b, lora_weight
            modified_count += 1
        
        print(f"    {modified_count} tensors are updated.")

    def match(self, model, state_dict_lora):
        """Check if LoRA parameters match model parameters without loading full state dict"""
        for model_class in self.supported_model_classes:
            if not isinstance(model, model_class):
                continue
            
            # Create set of parameter names
            param_names = set()
            for name, _ in model.named_parameters():
                param_names.add(name)
            
            # Check if a sample of LoRA keys map to model parameters
            matched_count = 0
            checked_count = 0
            
            for key in state_dict_lora:
                if ".lora_B." not in key:
                    continue
                
                target_name = self._get_target_name(key)
                if target_name in param_names:
                    matched_count += 1
                
                checked_count += 1
                if matched_count >= 5:  # Found enough matches
                    return "", ""
                if checked_count >= 50 and matched_count == 0:  # Checked enough without matches
                    break
            
            if matched_count > 0:
                return "", ""
                
        return None


class HunyuanVideoLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [HunyuanVideoDiT, HunyuanVideoDiT]
        self.lora_prefix = ["diffusion_model.", "transformer."]
        self.special_keys = {}
    

class FluxLoRAConverter:
    def __init__(self):
        pass

    @staticmethod
    def align_to_opensource_format(state_dict, alpha=1.0):
        prefix_rename_dict = {
            "single_blocks": "lora_unet_single_blocks",
            "blocks": "lora_unet_double_blocks",
        }
        middle_rename_dict = {
            "norm.linear": "modulation_lin",
            "to_qkv_mlp": "linear1",
            "proj_out": "linear2",

            "norm1_a.linear": "img_mod_lin",
            "norm1_b.linear": "txt_mod_lin",
            "attn.a_to_qkv": "img_attn_qkv",
            "attn.b_to_qkv": "txt_attn_qkv",
            "attn.a_to_out": "img_attn_proj",
            "attn.b_to_out": "txt_attn_proj",
            "ff_a.0": "img_mlp_0",
            "ff_a.2": "img_mlp_2",
            "ff_b.0": "txt_mlp_0",
            "ff_b.2": "txt_mlp_2",
        }
        suffix_rename_dict = {
            "lora_B.weight": "lora_up.weight",
            "lora_A.weight": "lora_down.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            names = name.split(".")
            if names[-2] != "lora_A" and names[-2] != "lora_B":
                names.pop(-2)
            prefix = names[0]
            middle = ".".join(names[2:-2])
            suffix = ".".join(names[-2:])
            block_id = names[1]
            if middle not in middle_rename_dict:
                continue
            rename = prefix_rename_dict[prefix] + "_" + block_id + "_" + middle_rename_dict[middle] + "." + suffix_rename_dict[suffix]
            state_dict_[rename] = param
            if rename.endswith("lora_up.weight"):
                state_dict_[rename.replace("lora_up.weight", "alpha")] = torch.tensor((alpha,))[0]
        return state_dict_
    
    @staticmethod
    def align_to_diffsynth_format(state_dict):
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
        state_dict_ = {}
        for name, param in state_dict.items():
            block_id, source_name = guess_block_id(name)
            if source_name in rename_dict:
                target_name = rename_dict[source_name]
                target_name = target_name.replace(".blockid.", f".{block_id}.")
                state_dict_[target_name] = param
            else:
                state_dict_[name] = param
        return state_dict_
    

def get_lora_loaders():
    return [SDLoRAFromCivitai(), SDXLLoRAFromCivitai(), FluxLoRAFromCivitai(), HunyuanVideoLoRAFromCivitai(), GeneralLoRAFromPeft()]
