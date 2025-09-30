import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
)
state_dict = load_state_dict("models/train/Wan2.2-Animate-14B_full/epoch-1.safetensors")
pipe.animate_adapter.load_state_dict(state_dict, strict=False)
pipe.enable_vram_management()

input_image = VideoData("data/example_video_dataset/animate/animate_output.mp4", height=480, width=832)[0]
animate_pose_video = VideoData("data/examples/wan/animate/animate_pose_video.mp4", height=480, width=832).raw_data()[:81-4]
animate_face_video = VideoData("data/examples/wan/animate/animate_face_video.mp4", height=512, width=512).raw_data()[:81-4]
video = pipe(
    prompt="视频中的人在做动作",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    num_frames=81, height=480, width=832,
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_Wan2.2-Animate-14B.mp4", fps=15, quality=5)