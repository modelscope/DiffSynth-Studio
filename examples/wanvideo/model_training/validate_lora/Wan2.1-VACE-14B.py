import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-14B_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
video = [video[i] for i in range(17)]
reference_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, vace_reference_image=reference_image, num_frames=17,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.1-VACE-14B.mp4", fps=15, quality=5)
