import torch
from PIL import Image
from diffsynth import save_video, VideoData, save_frames
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ],
    redirect_common_files=False,
)
pipe.load_lora(pipe.dit, "models/train/Wan2.1-Fun-1.3B-Control_lora_albedo_10k/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

video = VideoData("/eva_data0/lynn/VideoGAI/DiffSynth-Studio/input/ai_008_001_cam00_0000.png", height=480, width=832)
video = [video[i] for i in range(1)]


# Control video
out_video = pipe(
    prompt="generate albedo map",
    control_video=video,
    seed=42, tiled=True,
    num_frames=len(video),
)
# save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
save_frames(out_video, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/albedo_v2/seed42/008")




video9 = VideoData("/eva_data0/lynn/VideoGAI/DiffSynth-Studio/input/ai_009_001_cam00_0000.png", height=480, width=832)
video9 = [video9[i] for i in range(1)]

# Control video
out_video9 = pipe(
    prompt="generate albedo map",
    control_video=video9,
    seed=42, tiled=True,
    num_frames=len(video9),
)
# save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
save_frames(out_video9, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/albedo_v2/seed42/009")









# # Control video
# out_video = pipe(
#     prompt="generate irradiance map",
#     control_video=video,
#     seed=42, tiled=True,
#     num_frames=len(video),
# )
# # save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
# save_frames(out_video, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/all/009/irradiance")

# # Control video
# out_video = pipe(
#     prompt="generate normal map",
#     control_video=video,
#     seed=42, tiled=True,
#     num_frames=len(video),
# )
# # save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
# save_frames(out_video, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/all/009/normal")

