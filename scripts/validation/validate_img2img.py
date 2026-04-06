#!/usr/bin/env python3
"""
FLUX img2img validation — LoRA on DiT, NO ControlNet.

Feeds the SPAD binary image as input_image (VAE-encoded, partial denoising)
instead of as a ControlNet conditioning signal. The DiT LoRA adapts the
generative prior to the target domain.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict


def load_spad_image(path) -> Image.Image:
    """Load a SPAD image, handling 16-bit grayscale correctly."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="FLUX img2img validation (no ControlNet)")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str,
                        default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_img2img")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--embedded_guidance", type=float, default=3.5)
    parser.add_argument("--denoising_strength", type=float, default=0.7,
                        help="How much to denoise (0=return input, 1=full denoise from noise)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset_base", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "output").mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLUX img2img Validation (NO ControlNet)")
    print("=" * 60)
    print(f"  Checkpoint: {args.lora_checkpoint}")
    print(f"  Denoising strength: {args.denoising_strength}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load pipeline — NO ControlNet model
    print("Loading FLUX pipeline (no ControlNet)...")
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
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="ae.safetensors", **vram_config),
        # NO ControlNet
    ]

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        vram_limit=vram_limit,
    )

    # Load LoRA onto DiT
    print(f"Loading LoRA (dit): {args.lora_checkpoint}")
    pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=1.0)

    # Load validation samples
    csv_path = Path(args.metadata_csv)
    csv_base = Path(args.dataset_base) if args.dataset_base else csv_path.parent
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Processing {len(samples)} samples (denoising_strength={args.denoising_strength})")

    for idx, sample in enumerate(tqdm(samples, desc="img2img inference")):
        output_path = output_dir / "output" / f"output_{idx:04d}.png"
        if output_path.exists() and not args.overwrite:
            continue

        # Load SPAD input as the img2img input_image
        controlnet_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        input_path = csv_base / sample[controlnet_key]
        gt_path = csv_base / sample["image"]

        spad_img = load_spad_image(input_path)
        gt_img = Image.open(gt_path).convert("RGB")

        with torch.no_grad():
            result = pipe(
                prompt=sample.get("prompt", ""),
                input_image=spad_img,
                denoising_strength=args.denoising_strength,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
                seed=args.seed + idx,
                rand_device="cuda",
                # NO controlnet_inputs
            )

        spad_img.save(output_dir / "input" / f"input_{idx:04d}.png")
        result.save(output_path)
        gt_img.save(output_dir / "ground_truth" / f"gt_{idx:04d}.png")

    print(f"\nDone! Results saved to {output_dir}")
    print(f"Run metrics: python run_metrics.py {output_dir} --save")


if __name__ == "__main__":
    main()
