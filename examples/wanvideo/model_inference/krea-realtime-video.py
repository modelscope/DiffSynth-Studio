import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/krea-realtime-video", origin_file_pattern="krea-realtime-video-14b.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# Text-to-video
video = pipe(
    prompt="a cat sitting on a boat",
    num_inference_steps=6, num_frames=81,
    seed=0, tiled=True,
    cfg_scale=1,
    sigma_shift=20,
)
save_video(video, "video1.mp4", fps=15, quality=5)
