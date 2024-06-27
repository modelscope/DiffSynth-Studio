from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors)
# `models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors)
download_models(["StableDiffusionXL_v1", "IP-Adapter-SDXL"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors",
    "models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

torch.manual_seed(123456)
style_image = pipe(
    prompt="a rabbit in a garden, colorful flowers",
    negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
)
style_image.save("rabbit.jpg")

image = pipe(
    prompt="a cat",
    negative_prompt="",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
    ipadapter_images=[style_image], ipadapter_use_instant_style=True
)
image.save("rabbit_to_cat.jpg")

image = pipe(
    prompt="a rabbit is jumping",
    negative_prompt="",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
    ipadapter_images=[style_image], ipadapter_use_instant_style=False, ipadapter_scale=0.5
)
image.save("rabbit_to_jumping_rabbit.jpg")

image = pipe(
    prompt="a cat",
    negative_prompt="",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
)
image.save("rabbit_to_cat_without_ipa.jpg")

image = pipe(
    prompt="a rabbit is jumping",
    negative_prompt="",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
)
image.save("rabbit_to_jumping_rabbit_without_ipa.jpg")