from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch, requests
from PIL import Image


# Download models (automatically)
# `models/stable_diffusion_xl/bluePencilXL_v200.safetensors`: [link](https://civitai.com/api/download/models/245614?type=Model&format=SafeTensor&size=pruned&fp=fp16)
# `models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors)
# `models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors)
download_models(["BluePencilXL_v200", "IP-Adapter-SDXL"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors",
    "models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

image_1 = Image.open(requests.get("https://media.52poke.com/wiki/7/7e/006Charizard.png", stream=True).raw).convert("RGB").resize((1024, 1024))
image_1.save("Charizard.jpg")
image_2 = Image.open(requests.get("https://media.52poke.com/wiki/0/0d/025Pikachu.png", stream=True).raw).convert("RGB").resize((1024, 1024))
image_2.save("Pikachu.jpg")

torch.manual_seed(0)
image = pipe(
    prompt="a pokemon, maybe Charizard, maybe Pikachu",
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
    ipadapter_images=[image_1, image_2], ipadapter_use_instant_style=False, ipadapter_scale=0.7
)
image.save(f"Pikazard.jpg")
