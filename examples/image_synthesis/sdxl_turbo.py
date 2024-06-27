from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl_turbo/sd_xl_turbo_1.0_fp16.safetensors`: [link](https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors)
download_models(["StableDiffusionXL_Turbo"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion_xl_turbo/sd_xl_turbo_1.0_fp16.safetensors"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

# Text to image
torch.manual_seed(0)
image = pipe(
    prompt="black car",
    # Do not modify the following parameters!
    cfg_scale=1, height=512, width=512, num_inference_steps=1, progress_bar_cmd=lambda x:x
)
image.save(f"black_car.jpg")

# Image to image
torch.manual_seed(0)
image = pipe(
    prompt="red car",
    input_image=image, denoising_strength=0.7,
    # Do not modify the following parameters!
    cfg_scale=1, height=512, width=512, num_inference_steps=1, progress_bar_cmd=lambda x:x
)
image.save(f"black_car_to_red_car.jpg")
