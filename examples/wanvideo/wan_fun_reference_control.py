import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image


# Download models
# snapshot_download("PAI/Wan2.1-Fun-1.3B-Control", local_dir="models/PAI/Wan2.1-Fun-V1.1-1.3B-Control")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/PAI/Wan2.1-Fun-V1.1-14B-Control/diffusion_pytorch_model.safetensors",
        "models/PAI/Wan2.1-Fun-V1.1-14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
        "models/PAI/Wan2.1-Fun-V1.1-14B-Control/Wan2.1_VAE.pth",
        "models/PAI/Wan2.1-Fun-V1.1-14B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

# Control-to-video
control_video = VideoData("xxx/pose.mp4", height=832, width=480)
control_video = [control_video[i] for i in range(49)]
video = pipe(
    prompt="一位年轻女性穿着一件粉色的连衣裙，裙子上有白色的装饰和粉色的纽扣。她的头发是紫色的，头上戴着一个红色的大蝴蝶结，显得非常可爱和精致。她还戴着一个红色的领结，整体造型充满了少女感和活力。她的表情温柔，双手轻轻交叉放在身前，姿态优雅。背景是简单的灰色，没有任何多余的装饰，使得人物更加突出。她的妆容清淡自然，突显了她的清新气质。整体画面给人一种甜美、梦幻的感觉，仿佛置身于童话世界中。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    reference_image=Image.open("xxx/6.png").convert("RGB").resize((480, 832)),
    control_video=control_video, height=832, width=480, num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video1.mp4", fps=15, quality=5)
