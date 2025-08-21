import torch
from diffsynth import save_video,VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/control_video.mp4", "data/examples/wan/reference_image_girl.png"]
)

# Control video
control_video = VideoData("data/examples/wan/control_video.mp4", height=832, width=576)
reference_image = Image.open("data/examples/wan/reference_image_girl.png").resize((576, 832))
video = pipe(
    prompt="扁平风格动漫，一位长发少女优雅起舞。她五官精致，大眼睛明亮有神，黑色长发柔顺光泽。身穿淡蓝色T恤和深蓝色牛仔短裤。背景是粉色。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video=control_video, reference_image=reference_image,
    height=832, width=576, num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video.mp4", fps=15, quality=5)