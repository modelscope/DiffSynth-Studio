import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors", alpha=1)
pipe.load_lora(pipe.dit2, "models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

input_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    num_frames=49,
    seed=1, tiled=False,
)
save_video(video, "video_Wan2.2-I2V-A14B.mp4", fps=15, quality=5)
