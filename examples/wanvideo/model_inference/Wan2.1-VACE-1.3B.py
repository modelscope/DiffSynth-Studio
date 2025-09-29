import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

pipe.enable_vram_management()

control_video = VideoData("/eva_data0/lynn/VideoGAI/DiffSynth-Studio/depth_0030.mp4", height=480, width=832)
video = pipe(
    prompt="保持输入视频中的原始场景布局和运动；将所提供的参考图像的艺术风格在整个片段中一致地应用；保留人脸特征和细节",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=Image.open("/eva_data0/lynn/VideoGAI/P018_VPWIP_029_0150_Styleframe001_TargetStyle_Frame.1009.png").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "stylized_org_0030_style.mp4", fps=24, quality=5)