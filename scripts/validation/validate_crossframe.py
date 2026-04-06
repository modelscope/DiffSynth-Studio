#!/usr/bin/env python3
"""
Cross-Frame Variance Experiment: Generate outputs from different SPAD frame
realizations of the same scene using the same seed.

For each scene, generates outputs conditioned on each of 7 different binary
SPAD frames (different temporal realizations). This lets us measure:
  1. Cross-frame output variance (should be low if model is frame-invariant)
  2. Whether consistency training reduces this variance
"""

import argparse
import torch
import os
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import numpy as np

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict
from validate_flow_dps import load_spad_image


FRAME_FOLDERS = [
    ("bits",            "frames0-0"),
    ("bits_frame_1000", "frames1000-1000"),
    ("bits_frame_4000", "frames4000-4000"),
    ("bits_frame_5000", "frames5000-5000"),
    ("bits_frame_8000", "frames8000-8000"),
    ("bits_frame_12000","frames12000-12000"),
    ("bits_frame_16000","frames16000-16000"),
]


def scene_id_from_path(rel_path: str) -> str:
    """Extract scene ID from path like 'bits/0724-dgp-001_RAW_empty_frames0-0_p.png'."""
    fname = os.path.basename(rel_path)
    m = re.match(r"(.+?)_RAW_empty_frames", fname)
    return m.group(1) if m else None


def build_frame_path(scene_id: str, folder: str, frame_tag: str) -> str:
    return f"{folder}/{scene_id}_RAW_empty_{frame_tag}_p.png"


def main():
    parser = argparse.ArgumentParser(description="Cross-frame generation for variance analysis")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--lora_target", type=str, default="controlnet", choices=["dit", "controlnet"])
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_crossframe")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated frame folder names (default: all 7)")
    args = parser.parse_args()

    if args.frames:
        folder_lookup = {f: t for f, t in FRAME_FOLDERS}
        frame_folders = [(f, folder_lookup[f]) for f in args.frames.split(",")]
    else:
        frame_folders = FRAME_FOLDERS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)

    # Create per-frame output dirs
    for folder, _ in frame_folders:
        (output_dir / folder / "input").mkdir(parents=True, exist_ok=True)
        (output_dir / folder / "output").mkdir(parents=True, exist_ok=True)

    print("Loading FLUX pipeline...")
    vram_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    vram_limit = torch.cuda.mem_get_info()[1] / (1024 ** 3) - 0.5
    model_configs = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        vram_limit=vram_limit,
    )

    print(f"Loading LoRA ({args.lora_target}): {args.lora_checkpoint}")
    target_module = pipe.dit if args.lora_target == "dit" else pipe.controlnet
    if args.lora_target == "controlnet":
        state_dict = load_state_dict(args.lora_checkpoint, torch_dtype=pipe.torch_dtype, device=pipe.device)
        loader = FluxLoRALoader(torch_dtype=pipe.torch_dtype, device=pipe.device)
        loader.fuse_lora_to_base_model(target_module, state_dict, alpha=1.0)
    else:
        pipe.load_lora(target_module, args.lora_checkpoint, alpha=1.0)

    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Processing {len(samples)} samples × {len(frame_folders)} frames")
    print(f"Frame folders: {[f for f, _ in frame_folders]}")

    for fi, (folder, frame_tag) in enumerate(frame_folders):
        print(f"\n=== Frame {fi+1}/{len(frame_folders)}: {folder} ===")

        for idx, sample in enumerate(tqdm(samples, desc=folder)):
            output_path = output_dir / folder / "output" / f"output_{idx:04d}.png"
            if output_path.exists():
                continue

            # Get scene ID from original controlnet_image path
            control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
            original_path = sample[control_key]
            scene_id = scene_id_from_path(original_path)

            if scene_id is None:
                print(f"WARNING: Could not extract scene ID from {original_path}, skipping")
                continue

            # Build path for this frame folder
            frame_rel_path = build_frame_path(scene_id, folder, frame_tag)
            frame_full_path = csv_path.parent / frame_rel_path

            if not frame_full_path.exists():
                print(f"WARNING: Frame not found: {frame_full_path}, skipping")
                continue

            control_img = load_spad_image(frame_full_path)
            gt_path = csv_path.parent / sample["image"]
            gt_img = Image.open(gt_path).convert("RGB")

            controlnet_input = ControlNetInput(
                image=control_img,
                processor_id="gray",
                scale=1.0,
            )

            # Same seed for all frames of same scene
            seed = args.seed + idx

            result_image = pipe(
                prompt=sample.get("prompt", ""),
                cfg_scale=1.0,
                embedded_guidance=3.5,
                height=args.height,
                width=args.width,
                seed=seed,
                num_inference_steps=args.steps,
                controlnet_inputs=[controlnet_input],
            )

            control_img.save(output_dir / folder / "input" / f"input_{idx:04d}.png")
            result_image.save(output_path)

            # Save GT only once (shared across frames)
            gt_out = output_dir / "ground_truth" / f"gt_{idx:04d}.png"
            if not gt_out.exists():
                gt_img.save(gt_out)

    print(f"\nDone! Results saved to {output_dir}")
    print(f"Frames generated: {[f for f, _ in frame_folders]}")


if __name__ == "__main__":
    main()
