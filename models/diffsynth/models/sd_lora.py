import torch
from .sd_unet import SDUNetStateDictConverter, SDUNet
from .sd_text_encoder import SDTextEncoderStateDictConverter, SDTextEncoder


class SDLoRA:
    def __init__(self):
        pass

    def convert_state_dict(self, state_dict, lora_prefix="lora_unet_", alpha=1.0, device="cuda"):
        special_keys = {
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
        }
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
            target_name = key.split(".")[0].replace("_", ".")[len(lora_prefix):] + ".weight"
            for special_key in special_keys:
                target_name = target_name.replace(special_key, special_keys[special_key])
            state_dict_[target_name] = lora_weight.cpu()
        return state_dict_
    
    def add_lora_to_unet(self, unet: SDUNet, state_dict_lora, alpha=1.0, device="cuda"):
        state_dict_unet = unet.state_dict()
        state_dict_lora = self.convert_state_dict(state_dict_lora, lora_prefix="lora_unet_", alpha=alpha, device=device)
        state_dict_lora = SDUNetStateDictConverter().from_diffusers(state_dict_lora)
        if len(state_dict_lora) > 0:
            for name in state_dict_lora:
                state_dict_unet[name] += state_dict_lora[name].to(device=device)
            unet.load_state_dict(state_dict_unet)

    def add_lora_to_text_encoder(self, text_encoder: SDTextEncoder, state_dict_lora, alpha=1.0, device="cuda"):
        state_dict_text_encoder = text_encoder.state_dict()
        state_dict_lora = self.convert_state_dict(state_dict_lora, lora_prefix="lora_te_", alpha=alpha, device=device)
        state_dict_lora = SDTextEncoderStateDictConverter().from_diffusers(state_dict_lora)
        if len(state_dict_lora) > 0:
            for name in state_dict_lora:
                state_dict_text_encoder[name] += state_dict_lora[name].to(device=device)
            text_encoder.load_state_dict(state_dict_text_encoder)

