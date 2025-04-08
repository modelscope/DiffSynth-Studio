import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
snapshot_download("PAI/Wan2.1-Fun-1.3B-Control", local_dir="models/PAI/Wan2.1-Fun-1.3B-Control")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/PAI/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
        "models/PAI/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
        "models/PAI/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth",
        "models/PAI/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Download example video
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/control_video.mp4"
)

# Control-to-video
control_video = VideoData("data/examples/wan/control_video.mp4", height=832, width=576)
video = pipe(
    prompt="扁平风格动漫，一位长发少女优雅起舞。她五官精致，大眼睛明亮有神，黑色长发柔顺光泽。身穿淡蓝色T恤和深蓝色牛仔短裤。背景是粉色。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    control_video=control_video, height=832, width=576, num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video1.mp4", fps=15, quality=5)
