#!/usr/bin/env python3
"""
Phase 3b: FlowDPS Validation -- Run inference with measurement-guided sampling.

Uses the same checkpoint as standard validation but adds SPAD physics
guidance at each denoising step via FlowDPS.
"""

import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import numpy as np

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict
from diffsynth.diffusion.flow_dps import FlowDPSConfig, flux_dps_inference
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser(description="FLUX LoRA validation with FlowDPS guidance")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--lora_target", type=str, default="controlnet", choices=["dit", "controlnet"])
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_dps")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--dps_alpha", type=float, default=1.0, help="SPAD forward model alpha")
    parser.add_argument("--dps_num_frames", type=int, default=1, help="Number of SPAD frames")
    parser.add_argument("--dps_guidance_scale", type=float, default=0.1, help="DPS guidance strength")
    parser.add_argument("--dps_schedule", type=str, default="linear_decay", choices=["constant", "linear_decay", "cosine"])
    parser.add_argument("--dps_start_step", type=int, default=0)
    parser.add_argument("--dps_stop_step", type=int, default=-1)
    parser.add_argument("--dps_gradient_clamp", type=float, default=1.0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "output").mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

    print("Loading FLUX pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        [
            ModelConfig("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder_2/*.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
            ModelConfig("InstantX/FLUX.1-dev-Controlnet-Union-alpha", "diffusion_pytorch_model.safetensors"),
        ],
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA from {args.lora_checkpoint} into {args.lora_target}...")
    if args.lora_target == "controlnet":
        lora_loader = FluxLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = load_state_dict(args.lora_checkpoint, torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = lora_loader.convert_state_dict(lora_sd)
        lora_loader.fuse_lora(pipe.controlnet, lora_sd, alpha=1.0)
    else:
        pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=1.0)

    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Processing {len(samples)} samples with FlowDPS guidance...")
    print(f"  DPS alpha={args.dps_alpha}, scale={args.dps_guidance_scale}, schedule={args.dps_schedule}")

    to_tensor = transforms.ToTensor()

    for idx, sample in enumerate(tqdm(samples, desc="FlowDPS inference")):
        output_path = output_dir / "output" / f"output_{idx:04d}.png"
        if output_path.exists():
            continue

        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        gt_path = csv_path.parent / sample["image"]

        control_img = Image.open(control_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        spad_tensor = to_tensor(control_img).unsqueeze(0)

        dps_config = FlowDPSConfig(
            spad_measurement=spad_tensor,
            alpha=args.dps_alpha,
            num_frames=args.dps_num_frames,
            guidance_scale=args.dps_guidance_scale,
            guidance_schedule=args.dps_schedule,
            start_step=args.dps_start_step,
            stop_step=args.dps_stop_step,
            gradient_clamp=args.dps_gradient_clamp,
        )

        controlnet_input = ControlNetInput(
            image=control_img,
            processor_id="gray",
            scale=1.0,
        )

        seed = args.seed + idx

        result_image = flux_dps_inference(
            pipe,
            dps_config=dps_config,
            prompt=sample.get("prompt", ""),
            cfg_scale=1.0,
            embedded_guidance=3.5,
            height=args.height,
            width=args.width,
            seed=seed,
            num_inference_steps=args.steps,
            controlnet_inputs=[controlnet_input],
        )

        control_img.save(output_dir / "input" / f"input_{idx:04d}.png")
        result_image.save(output_dir / "output" / f"output_{idx:04d}.png")
        gt_img.save(output_dir / "ground_truth" / f"gt_{idx:04d}.png")

    print(f"\nDone! Results saved to {output_dir}")
    print(f"Run metrics: python run_metrics.py {output_dir} --save")


if __name__ == "__main__":
    main()
