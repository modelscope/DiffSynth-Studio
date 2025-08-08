# This script is for initializing a Qwen-Image-Accelerate-Adapter
from diffsynth import load_state_dict, hash_state_dict_keys
from diffsynth.pipelines.qwen_image import QwenImageAccelerateAdapter
import torch
from safetensors.torch import save_file


state_dict_dit = {}
for i in range(1, 10):
    state_dict_dit.update(load_state_dict(f"models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-0000{i}-of-00009.safetensors", torch_dtype=torch.bfloat16, device="cuda"))

adapter = QwenImageAccelerateAdapter().to(dtype=torch.bfloat16, device="cuda")
state_dict_adapter = adapter.state_dict()

state_dict_init = {}
for k in state_dict_adapter:
    if k.startswith("transformer_blocks"):
        name = k.replace("transformer_blocks.0.", "transformer_blocks.59.")
        param = state_dict_dit[name]
        if "_mod." in k:
            param[2*3072: 3*3072] = 0
            param[5*3072: 6*3072] = 0
        state_dict_init[k] = param
    elif k in state_dict_dit:
        state_dict_init[k] = state_dict_dit[k]
    else:
        state_dict_init[k] = torch.zeros_like(state_dict_adapter[k])
        print("Zero initialized:", k)
adapter.load_state_dict(state_dict_init)

print(hash_state_dict_keys(state_dict_init))
save_file(state_dict_init, "models/adapter.safetensors")