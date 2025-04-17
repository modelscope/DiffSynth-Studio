import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
snapshot_download("Wan-AI/Wan2.1-FLF2V-14B-720P", local_dir="models/Wan-AI/Wan2.1-FLF2V-14B-720P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-FLF2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-FLF2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Download example image
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/first_frame.jpeg", "data/examples/wan/last_frame.jpeg"]
)

# First and last frame to video
video = pipe(
    prompt="写实风格，一个女生手持枯萎的花站在花园中，镜头逐渐拉远，记录下花园的全貌。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=30,
    input_image=Image.open("data/examples/wan/first_frame.jpeg").resize((960, 960)),
    end_image=Image.open("data/examples/wan/last_frame.jpeg").resize((960, 960)),
    height=960, width=960,
    seed=1, tiled=True
)
save_video(video, "video.mp4", fps=15, quality=5)
