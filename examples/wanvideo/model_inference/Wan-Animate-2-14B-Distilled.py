import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

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
        ModelConfig(model_id="Wan-AI/Wan-Animate-2-14B", origin_file_pattern="wan_animate_2/wan_animate_2_bf16_distillation.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan-Animate-2-14B", origin_file_pattern="videomodel/Wan-AI/umt5-xxl/"),
)

# Character animation: reference image (identity) + reference video (motion) -> animated video.
dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="wanvideo/Wan-Animate-2-14B-Distilled/*"
)
reference_image = Image.open("data/diffsynth_example_dataset/wanvideo/Wan-Animate-2-14B-Distilled/refimage.jpg").convert("RGB")
reference_video = VideoData("data/diffsynth_example_dataset/wanvideo/Wan-Animate-2-14B-Distilled/refvideo.mp4").raw_data()

num_frames = 81
# For distilled model, set animate2_log_scale to -1.3, num_inference_steps to 10, and cfg_scale to 1.0.
video = pipe(
    prompt="人物外观描述：一名长黑发女性，穿着白色半透明蕾丝长袖上衣，衣身带有花卉刺绣，下身搭配白色百褶短裙和黑色腰带，脚穿米白色厚底运动鞋。 背景描述：背景为现代室内空间，墙面和柜体以浅灰色为主，后方设有两扇深色落地窗或玻璃门，顶部安装长条形灯具，中央有一块浅色长方形台面。",
    animate2_prompt_ref="视频中的人在做动作，背景静止",
    animate2_reference_image=reference_image,
    animate2_reference_video=reference_video[:num_frames],
    animate2_offload_kv=True,
    animate2_log_scale=-1.3,
    num_frames=num_frames, height=1280, width=720,
    num_inference_steps=10, cfg_scale=1.0,
    seed=0, tiled=True,
)
save_video(video, "video_Wan-Animate-2-14B-Distilled.mp4", fps=24, quality=5)
