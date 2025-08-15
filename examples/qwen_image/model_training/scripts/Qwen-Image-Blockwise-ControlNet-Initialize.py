# This script is for initializing a Qwen-Image-Blockwise-ControlNet
from diffsynth import hash_state_dict_keys
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet
import torch
from safetensors.torch import save_file


controlnet = QwenImageBlockWiseControlNet().to(dtype=torch.bfloat16, device="cuda")
controlnet.init_weight()
state_dict_controlnet = controlnet.state_dict()

print(hash_state_dict_keys(state_dict_controlnet))
save_file(state_dict_controlnet, "models/blockwise_controlnet.safetensors")
