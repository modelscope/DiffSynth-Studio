# This script is for initializing a Inpaint Qwen-Image-ControlNet
import torch
from diffsynth import hash_state_dict_keys
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet
from safetensors.torch import save_file

controlnet = QwenImageBlockWiseControlNet(additional_in_dim=4).to(dtype=torch.bfloat16, device="cuda")
controlnet.init_weight()
state_dict_controlnet = controlnet.state_dict()

print(hash_state_dict_keys(state_dict_controlnet))
save_file(state_dict_controlnet, "models/blockwise_controlnet_inpaint.safetensors")
