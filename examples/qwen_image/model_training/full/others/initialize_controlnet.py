# This script is for initializing a Qwen-Image-ControlNet
from diffsynth import load_state_dict, hash_state_dict_keys
from diffsynth.pipelines.qwen_image import QwenImageControlNet
import torch
from safetensors.torch import save_file


state_dict_dit = {}
for i in range(1, 10):
    state_dict_dit.update(load_state_dict(f"models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-0000{i}-of-00009.safetensors", torch_dtype=torch.bfloat16, device="cuda"))

controlnet = QwenImageControlNet().to(dtype=torch.bfloat16, device="cuda")
state_dict_controlnet = controlnet.state_dict()

state_dict_init = {}
for k in state_dict_controlnet:
    if k in state_dict_dit:
        if state_dict_dit[k].shape == state_dict_controlnet[k].shape:
            state_dict_init[k] = state_dict_dit[k]
        elif k == "img_in.weight":
            state_dict_init[k] = torch.concat(
                [
                    state_dict_dit[k],
                    state_dict_dit[k],
                ],
                dim=-1
            )
    elif k == "alpha":
        state_dict_init[k] = torch.zeros_like(state_dict_controlnet[k])
controlnet.load_state_dict(state_dict_init)

print(hash_state_dict_keys(state_dict_controlnet))
save_file(state_dict_controlnet, "models/controlnet.safetensors")
