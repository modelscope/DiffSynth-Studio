from diffsynth import ModelManager, SDImagePipeline, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video, save_frames
from diffsynth.extensions.RIFE import RIFEInterpolater
import torch


# Download models
# `models/stable_diffusion/dreamshaper_8.safetensors`: [link](https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16)
# `models/AnimateDiff/mm_sd_v15_v2.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)
# `models/RIFE/flownet.pkl`: [link](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing)


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion/dreamshaper_8.safetensors",
    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
    "models/RIFE/flownet.pkl"
])

# Text -> Image
pipe_image = SDImagePipeline.from_model_manager(model_manager)
torch.manual_seed(0)
image = pipe_image(
    prompt = "lightning storm, sea",
    negative_prompt = "",
    cfg_scale=7.5,
    num_inference_steps=30, height=512, width=768,
)

# Text + Image -> Video (6GB VRAM is enough!)
pipe = SDVideoPipeline.from_model_manager(model_manager)
output_video = pipe(
    prompt = "lightning storm, sea",
    negative_prompt = "",
    cfg_scale=7.5,
    num_frames=64,
    num_inference_steps=10, height=512, width=768,
    animatediff_batch_size=16, animatediff_stride=1, input_frames=[image]*64, denoising_strength=0.9,
    vram_limit_level=0,
)

# Video -> Video with high fps
interpolater = RIFEInterpolater.from_model_manager(model_manager)
output_video = interpolater.interpolate(output_video, num_iter=3)

# Save images and video
save_video(output_video, "output_video.mp4", fps=120)
