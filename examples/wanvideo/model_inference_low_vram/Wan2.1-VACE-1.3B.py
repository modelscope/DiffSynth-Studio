import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)


dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
)

# Depth video -> Video
control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    seed=1, tiled=True
)
save_video(video, "video_1_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)

# Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video_2_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)

# Depth video + Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video_3_Wan2.1-VACE-1.3B.mp4", fps=15, quality=5)
