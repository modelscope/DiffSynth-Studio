import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
snapshot_download("PAI/Wan2.1-Fun-1.3B-InP", local_dir="models/PAI/Wan2.1-Fun-1.3B-InP")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
        "models/PAI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
        "models/PAI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
        "models/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Download example image
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/input_image.jpg"
)
image = Image.open("data/examples/wan/input_image.jpg")

# Image-to-video
video = pipe(
    prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    input_image=image,
    # You can input `end_image=xxx` to control the last frame of the video.
    # The model will automatically generate the dynamic content between `input_image` and `end_image`.
    seed=1, tiled=True
)
save_video(video, "video1.mp4", fps=15, quality=5)
