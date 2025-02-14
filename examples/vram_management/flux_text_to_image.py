import torch
from diffsynth import ModelManager, FluxImagePipeline


model_manager = ModelManager(
    file_path_list=[
        "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
        "models/FLUX/FLUX.1-dev/text_encoder_2",
        "models/FLUX/FLUX.1-dev/flux1-dev.safetensors",
        "models/FLUX/FLUX.1-dev/ae.safetensors",
    ],
    torch_dtype=torch.float8_e4m3fn,
    device="cpu"
)
pipe = FluxImagePipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

# Enable VRAM management
# `num_persistent_param_in_dit` indicates the number of parameters that reside persistently in VRAM within the DiT model.
# When `num_persistent_param_in_dit=None`, it means all parameters reside persistently in memory.
# When `num_persistent_param_in_dit=7*10**9`, it indicates that 7 billion parameters reside persistently in memory.
# When `num_persistent_param_in_dit=0`, it means no parameters reside persistently in memory, and they are loaded layer by layer during inference.
pipe.enable_vram_management(num_persistent_param_in_dit=None)

image = pipe(prompt="a beautiful orange cat", seed=0)
image.save("image.jpg")
