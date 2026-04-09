import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/krea-realtime-video", origin_file_pattern="krea-realtime-video-14b.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)

pipe.load_lora(pipe.dit, "models/train/krea-realtime-video_lora/epoch-4.safetensors", alpha=1)

# Text-to-video
video = pipe(
    prompt="a cat sitting on a boat",
    num_inference_steps=6, num_frames=81,
    seed=0, tiled=True,
    cfg_scale=1,
    sigma_shift=20,
)
save_video(video, "video_krea-realtime-video.mp4", fps=15, quality=5)
