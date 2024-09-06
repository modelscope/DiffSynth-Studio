import torch
from .sd_unet import SDUNet
from .sdxl_unet import SDXLUNet
from .sd_text_encoder import SDTextEncoder
from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sd3_dit import SD3DiT
from .flux_dit import FluxDiT
from .hunyuan_dit import HunyuanDiT



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
        if len(state_dict_lora) > 0:
            print(f"    {len(state_dict_lora)} tensors are updated.")
            for name in state_dict_lora:
                state_dict_model[name] += state_dict_lora[name].to(
                    dtype=state_dict_model[name].dtype, device=state_dict_model[name].device)
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
        self.supported_model_classes = [SDUNet, SDXLUNet, SD3DiT, HunyuanDiT, FluxDiT]


    def fetch_device_dtype_from_state_dict(self, state_dict):
        device, torch_dtype = None, None
        for name, param in state_dict.items():
            device, torch_dtype = param.device, param.dtype
            break
        return device, torch_dtype


    def convert_state_dict(self, state_dict, alpha=1.0, target_state_dict={}):
        device, torch_dtype = self.fetch_device_dtype_from_state_dict(target_state_dict)
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
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            target_name = ".".join(keys)
            if target_name not in target_state_dict:
                return {}
            state_dict_[target_name] = lora_weight.cpu()
        return state_dict_
    

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        state_dict_model = model.state_dict()
        state_dict_lora = self.convert_state_dict(state_dict_lora, alpha=alpha, target_state_dict=state_dict_model)
        if len(state_dict_lora) > 0:
            print(f"    {len(state_dict_lora)} tensors are updated.")
            for name in state_dict_lora:
                state_dict_model[name] += state_dict_lora[name].to(
                    dtype=state_dict_model[name].dtype,
                    device=state_dict_model[name].device
                )
            model.load_state_dict(state_dict_model)
    

    def match(self, model, state_dict_lora):
        for model_class in self.supported_model_classes:
            if not isinstance(model, model_class):
                continue
            state_dict_model = model.state_dict()
            try:
                state_dict_lora_ = self.convert_state_dict(state_dict_lora, alpha=1.0, target_state_dict=state_dict_model)
                if len(state_dict_lora_) > 0:
                    return "", ""
            except:
                pass
        return None
    

def get_lora_loaders():
    return [SDLoRAFromCivitai(), SDXLLoRAFromCivitai(), GeneralLoRAFromPeft(), FluxLoRAFromCivitai()]
