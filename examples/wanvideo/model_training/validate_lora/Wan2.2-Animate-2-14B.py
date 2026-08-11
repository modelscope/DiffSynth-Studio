import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="wan_animate_2/wan_animate_2_bf16.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/umt5-xxl/"),
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-Animate-2-14B_lora/epoch-4.safetensors", alpha=1)

reference_image = Image.open("data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B/refimage.jpg").convert("RGB")
reference_video = VideoData("data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B/refvideo.mp4").raw_data()
num_frames = 81
video = pipe(
    prompt="人物外观描述：一名长黑发女性，穿着白色半透明蕾丝长袖上衣，衣身带有花卉刺绣，下身搭配白色百褶短裙和黑色腰带，脚穿米白色厚底运动鞋。 背景描述：背景为现代室内空间，墙面和柜体以浅灰色为主，后方设有两扇深色落地窗或玻璃门，顶部安装长条形灯具，中央有一块浅色长方形台面。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    animate2_prompt_ref="视频中的人在做动作，背景静止",
    animate2_reference_image=reference_image,
    animate2_reference_video=reference_video[:num_frames],
    animate2_offload_kv=True,
    num_frames=num_frames, height=640, width=352,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0, tiled=True,
)
save_video(video, "video_Wan2.2-Animate-2-14B.mp4", fps=24, quality=5)
