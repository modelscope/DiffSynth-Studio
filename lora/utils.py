from diffsynth import load_state_dict
import math, torch


def load_lora(file_path, device):
    sd = load_state_dict(file_path, torch_dtype=torch.bfloat16, device=device)
    scale = math.sqrt(sd["lora_unet_single_blocks_9_modulation_lin.alpha"] / sd["lora_unet_single_blocks_9_modulation_lin.lora_down.weight"].shape[0])
    if scale != 1:
        sd = {i: sd[i] * scale for i in sd}
    return sd


