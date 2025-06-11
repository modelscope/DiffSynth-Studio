import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
snapshot_download("iic/VACE-Wan2.1-1.3B-Preview", local_dir="models/iic/VACE-Wan2.1-1.3B-Preview")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
        "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
        "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Download example video
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
    num_inference_steps=50,
    height=480, width=832, num_frames=81,
    vace_video=control_video,
    seed=1, tiled=True
)
save_video(video, "video1.mp4", fps=15, quality=5)

# Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    height=480, width=832, num_frames=81,
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video2.mp4", fps=15, quality=5)

# Depth video + Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    height=480, width=832, num_frames=81,
    vace_video=control_video,
    vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "video3.mp4", fps=15, quality=5)
