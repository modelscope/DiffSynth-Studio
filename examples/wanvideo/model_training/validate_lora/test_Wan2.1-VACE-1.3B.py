import csv
import re
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from pathlib import Path
import os
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-1.3B_lora/epoch-6.safetensors", alpha=1)

out_dir = Path("/mnt/bucket/dawy/video_generation/vace_training_exps/vace_refactor/batching/epoch_6/")
out_dir.mkdir(parents=True, exist_ok=True)

dataset_path = Path("/mnt/bucket/dawy/video_generation/two_stage_dataset/")
test_csv_path = Path("/mnt/bucket/dawy/video_generation/two_stage_dataset/Vace_Video_Generation_Dataset_Analysis_test.csv")

with test_csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    video_folder = row["video_id"]
    camera_motion = row["camera_motion"]
    motion_slug = re.sub(r"[^a-z0-9]+", "_", camera_motion.strip().lower()).strip("_")
    video_path = dataset_path / video_folder / f"vace_video_{video_folder}.mp4"
    mask_path = dataset_path / video_folder / f"vace_video_mask_{video_folder}.mp4"
    if not video_path.exists() or not mask_path.exists():
        print(f"[SKIP] {video_folder}: missing vace_video or mask")
        continue

    print(video_folder, camera_motion)

    video = VideoData(str(video_path), height=480, width=832)
    if len(video) >= 81 :
        video = [video[i] for i in range(81)]
    else:
        video = [video[i] for i in range(len(video))]
    reference_image = VideoData(str(video_path), height=480, width=832)[0]
    vace_mask = VideoData(str(mask_path), height=480, width=832)
    vace_mask = [vace_mask[i] for i in range(len(video))]
    video = pipe(
        prompt=" ",
        vace_video=video, vace_reference_image=reference_image, num_frames=len(video), vace_video_mask=vace_mask,
        seed=1, tiled=True
    )
    out_path = out_dir / f"video_{video_folder}_{motion_slug}.mp4"
    save_video(video, str(out_path), fps=15, quality=5)
