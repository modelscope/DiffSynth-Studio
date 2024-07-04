from diffsynth import ModelManager, SD3ImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_3/sd3_medium_incl_clips.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors)
download_models(["StableDiffusion3_without_T5"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)


torch.manual_seed(0)
image = pipe(
    prompt="a white cat, colorful ink painting, cyberpunk, unreal", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi",
    cfg_scale=4.5,
    num_inference_steps=50, width=1024, height=1024,
)
image.save("image_1024.jpg")

image = pipe(
    prompt="a white cat, colorful ink painting, cyberpunk, unreal", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi",
    input_image=image.resize((2048, 2048)), denoising_strength=0.5,
    cfg_scale=4.5,
    num_inference_steps=50, width=2048, height=2048,
    tiled=True
)
image.save("image_2048.jpg")
