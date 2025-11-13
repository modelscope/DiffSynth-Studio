import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern="data/examples/wan/animate/*",
)

# Animate
input_image = Image.open("data/examples/wan/animate/animate_input_image.png")
animate_pose_video = VideoData("data/examples/wan/animate/animate_pose_video.mp4").raw_data()[:81-4]
animate_face_video = VideoData("data/examples/wan/animate/animate_face_video.mp4").raw_data()[:81-4]
video = pipe(
    prompt="视频中的人在做动作",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    num_frames=81, height=720, width=1280,
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_1_Wan2.2-Animate-14B.mp4", fps=15, quality=5)

# Replace
pipe.load_lora(pipe.dit, ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="relighting_lora.ckpt"))
input_image = Image.open("data/examples/wan/animate/replace_input_image.png")
animate_pose_video = VideoData("data/examples/wan/animate/replace_pose_video.mp4").raw_data()[:81-4]
animate_face_video = VideoData("data/examples/wan/animate/replace_face_video.mp4").raw_data()[:81-4]
animate_inpaint_video = VideoData("data/examples/wan/animate/replace_inpaint_video.mp4").raw_data()[:81-4]
animate_mask_video = VideoData("data/examples/wan/animate/replace_mask_video.mp4").raw_data()[:81-4]
video = pipe(
    prompt="视频中的人在做动作",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    animate_inpaint_video=animate_inpaint_video,
    animate_mask_video=animate_mask_video,
    num_frames=81, height=720, width=1280,
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_2_Wan2.2-Animate-14B.mp4", fps=15, quality=5)
