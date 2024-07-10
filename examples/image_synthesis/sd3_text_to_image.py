from diffsynth import ModelManager, SD3ImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_3/sd3_medium_incl_clips.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors)
download_models(["StableDiffusion3_without_T5"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)


prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(7)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_1024.jpg")

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    input_image=image.resize((2048, 2048)), denoising_strength=0.5,
    num_inference_steps=50, width=2048, height=2048,
    tiled=True
)
image.save("image_2048.jpg")
