import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-1.3B_lora/step-1100.safetensors", alpha=0.2)

video = VideoData("data_infer/processed/pose/dance-4_1_pose.mp4", height=480, width=832)
video = [video[i] for i in range(65)]
reference_image = VideoData("data_infer/ref_img.jpg", height=480, width=832)[0]

video = pipe(
    prompt="Person is dancing by following the video pose exactly. Dance is natural and smooth. Maintain the exact facial features, hair, clothing, and background from the reference image. Keep the background consistent with the reference image. He is wearing black t-shirt, gray jeans, and white sneakers shoes. The background contain white wall, and wooden flooring.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, vace_reference_image=reference_image, num_frames=65,
    seed=1, tiled=True
)
save_video(video, "results/lora_dance.mp4", fps=15, quality=9)
