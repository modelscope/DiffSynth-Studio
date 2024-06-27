from diffsynth import ModelManager, SDXLVideoPipeline, save_video, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/AnimateDiff/mm_sdxl_v10_beta.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt)
download_models(["StableDiffusionXL_v1", "AnimateDiff_xl_beta"])

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/AnimateDiff/mm_sdxl_v10_beta.ckpt"
])
pipe = SDXLVideoPipeline.from_model_manager(model_manager)

prompt = "A panda standing on a surfboard in the ocean in sunset, 4k, high resolution.Realistic, Cinematic, high resolution"
negative_prompt = ""

torch.manual_seed(0)
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=8.5,
    height=1024, width=1024, num_frames=16,
    num_inference_steps=100,
)
save_video(video, "output_video.mp4", fps=16)
