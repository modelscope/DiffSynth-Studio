from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/bluePencilXL_v200.safetensors`: [link](https://civitai.com/api/download/models/245614?type=Model&format=SafeTensor&size=pruned&fp=fp16)
download_models(["BluePencilXL_v200"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion_xl/bluePencilXL_v200.safetensors"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=6,
    height=1024, width=1024, num_inference_steps=60,
)
image.save("1024.jpg")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=6,
    input_image=image.resize((2048, 2048)),
    height=2048, width=2048, num_inference_steps=60, denoising_strength=0.5
)
image.save("2048.jpg")

