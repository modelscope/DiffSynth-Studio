import torch
from modelscope import dataset_snapshot_download
from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_video_edit import QwenVideoEditPipeline
from diffsynth.utils.data import VideoData, save_video

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="qwen_video_edit/Qwen-Video-Edit/*"
)

edit_video = VideoData("data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit/source.mp4")
prompts = [
    "Transform the video into Japanese anime style",
]
pipe = QwenVideoEditPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="yunpeng1998/Qwen-Video-Edit", origin_file_pattern="360P/step-30000.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
video = pipe(edit_video=edit_video, prompts=prompts, height=640, width=384, num_frames=45, cfg_scale=4.0, num_inference_steps=40, seed=0)
save_video(video, "video_Qwen-Video-Edit.mp4", fps=16)
