import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict, save_frames
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
state_dict = load_state_dict("models/train/Wan2.1-Fun-1.3B-Control_full_albedo/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)
pipe.enable_vram_management()

'''
001
'''
video1 = VideoData("/eva_data0/lynn/VideoGAI/DiffSynth-Studio/input/girl.jpg", height=480, width=832)
video1 = [video1[i] for i in range(1)]

# Control video
out_video1 = pipe(
    prompt="generate albedo map",
    control_video=video1,
    seed=42, tiled=True,
    num_frames=len(video1),
)
# save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
save_frames(out_video1, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/full_albedo/seed42/girl/albdeo")

# # Control video
# out_video1 = pipe(
#     prompt="generate irradiance map",
#     control_video=video1,
#     seed=42, tiled=True,
#     num_frames=len(video1),
# )
# # save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
# save_frames(out_video1, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/full_all/seed42/001/irradiance")

# # Control video
# out_video1 = pipe(
#     prompt="generate normal map",
#     control_video=video1,
#     seed=42, tiled=True,
#     num_frames=len(video1),
# )
# # save_video(video, "video_Wan2.1-Fun-1.3B-Control.mp4", fps=15, quality=5)
# save_frames(out_video1, "/eva_data0/lynn/VideoGAI/DiffSynth-Studio/output/full_all/seed42/001/normal")

