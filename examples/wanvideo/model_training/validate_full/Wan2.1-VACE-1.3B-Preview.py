import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
state_dict = load_state_dict("models/train/Wan2.1-VACE-1.3B-Preview_full/epoch-1.safetensors")
pipe.vace.load_state_dict(state_dict)

video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
video = [video[i] for i in range(49)]
reference_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, vace_reference_image=reference_image, num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.1-VACE-1.3B-Preview.mp4", fps=15, quality=5)
