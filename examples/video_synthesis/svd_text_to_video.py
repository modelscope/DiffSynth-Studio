from diffsynth import save_video, SDXLImagePipeline, ModelManager, SVDVideoPipeline
from diffsynth import ModelManager
import torch


# Download models
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/stable_video_diffusion/svd_xt.safetensors`: [link](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors)


prompt = "cloud, wind"
torch.manual_seed(0)

# 1. Text-to-image using SD-XL
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion_xl/sd_xl_base_1.0.safetensors"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)
image = pipe(
    prompt=prompt,
    negative_prompt="",
    cfg_scale=6,
    height=1024, width=1024, num_inference_steps=50,
)
pipe.to("cpu")
torch.cuda.empty_cache()

# 2. Image-to-video using SVD
model_manager = ModelManager()
model_manager.load_models(["models/stable_video_diffusion/svd_xt.safetensors"])
pipe = SVDVideoPipeline.from_model_manager(model_manager)
video = pipe(
    input_image=image,
    num_frames=25, fps=15, height=1024, width=1024,
    motion_bucket_id=127,
    num_inference_steps=50
)
save_video(video, "video.mp4", fps=15)
