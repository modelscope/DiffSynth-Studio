from diffsynth import load_state_dict
import torch
from safetensors.torch import save_file
from tqdm import tqdm


def dequantize(source_path, target_path, device="cuda", torch_dtype=torch.bfloat16):
    sd = load_state_dict(source_path, device=device)
    for k in tqdm([k for k in sd if k.endswith(".weight_scale")]):
        weight_key = k[:-13] + ".weight"
        weight = sd.pop(weight_key).to(torch_dtype)
        scale = sd.pop(k).to(torch_dtype).unsqueeze(1)
        sd[weight_key] = weight * scale
    if target_path is not None:
        save_file(sd, target_path)
