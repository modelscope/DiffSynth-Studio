from diffsynth import ModelManager, SDXLImagePipeline
import torch


# Download models
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/IpAdapter/image_encoder/model.safetensors`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors)
# `models/IpAdapter/ip-adapter_sdxl.bin`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors)

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/IpAdapter/image_encoder/model.safetensors",
    "models/IpAdapter/ip-adapter_sdxl.bin"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager)
pipe.ipadapter.set_less_adapter()

torch.manual_seed(0)
style_image = pipe(
    prompt="Starry Night, blue sky, by van Gogh",
    negative_prompt="dark, gray",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=30,
)
style_image.save("style_image.jpg")

image = pipe(
    prompt="a cat",
    negative_prompt="",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=30,
    ipadapter_images=[style_image]
)
image.save("transferred_image.jpg")
