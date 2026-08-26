import torch
from modelscope import dataset_snapshot_download

from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_video_edit import QwenVideoEditPipeline
from diffsynth.utils.data import VideoData, save_video


dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="wanvideo/Wan2.2-Animate-2-14B/*",
)

input_video = VideoData("data/example_image_dataset/wanvideo/Wan2.2-Animate-2-14B/video.mp4")
prompts = [
    "Transform the video into Japanese anime style with cel shading and clean line art, preserving the original dance motion and composition.",
    "Apply a warm golden-hour color grading with soft cinematic lighting, keeping the dance motion and composition intact.",
]

pipe = QwenVideoEditPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
    ],
    video_vae_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    checkpoint=ModelConfig(model_id="yunpeng1998/Qwen-Video-Edit", origin_file_pattern="360P/step-30000.safetensors"),
)
video = pipe(input_video, prompts=prompts, cfg_scale=4.0, zero_cond_t=False, num_inference_steps=40, seed=42)
save_video(video, "video_Qwen-Video-Edit.mp4", fps=16)
