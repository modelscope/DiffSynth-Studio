#!/usr/bin/env python3
"""
Phase 3b: FlowDPS Validation -- Latent-space measurement-guided sampling.

Monkeypatches the pipeline's step function to inject a latent-space DPS
correction at each denoising step. This avoids the OOM issue of decoding
through the VAE during the denoising loop.

The SPAD measurement is pre-encoded through the VAE encoder once, and then
a simple latent-space consistency gradient is applied at each step:
  x_0_hat = x_t - sigma * v_theta
  correction = -eta * normalize(x_0_hat - z_spad)
"""

import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import numpy as np
from functools import partial

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict
from diffsynth.diffusion.latent_dps import LatentDPSConfig, compute_latent_dps_correction, get_guidance_weight


def load_spad_image(path) -> Image.Image:
    """Load a SPAD image, handling 16-bit grayscale correctly."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


def make_dps_step_fn(original_step, dps_config, total_steps):
    """Wrap the pipeline's step to inject DPS correction into noise_pred."""

    def dps_step(scheduler, latents, progress_id, noise_pred, **kwargs):
        sigma = scheduler.sigmas[progress_id].item() if hasattr(scheduler, 'sigmas') else 1.0
        sigma_max = scheduler.sigmas[0].item() if hasattr(scheduler, 'sigmas') and len(scheduler.sigmas) > 0 else 1.0

        weight = get_guidance_weight(
            progress_id, total_steps,
            schedule=dps_config.guidance_schedule,
            start_step=dps_config.start_step,
            stop_step=dps_config.stop_step,
            sigma=sigma,
            sigma_max=sigma_max,
        )
        if weight > 0 and dps_config.spad_latent is not None:
            if sigma > 0.01:
                correction = compute_latent_dps_correction(
                    latents=latents,
                    noise_pred=noise_pred,
                    sigma=sigma,
                    spad_latent=dps_config.spad_latent,
                    guidance_scale=dps_config.guidance_scale * weight,
                )
                noise_pred = noise_pred + correction

        return original_step(scheduler, latents=latents, progress_id=progress_id, noise_pred=noise_pred, **kwargs)

    return dps_step


def main():
    parser = argparse.ArgumentParser(description="FLUX LoRA validation with latent-space DPS guidance")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--lora_target", type=str, default="controlnet", choices=["dit", "controlnet"])
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_dps")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--controlnet_fp8", action="store_true")

    parser.add_argument("--dps_guidance_scale", type=float, default=0.05, help="DPS guidance strength (eta)")
    parser.add_argument("--dps_schedule", type=str, default="ramp_up", choices=["constant", "linear_decay", "cosine", "ramp_up", "sigma_ramp"])
    parser.add_argument("--dps_start_step", type=int, default=0)
    parser.add_argument("--dps_stop_step", type=int, default=-1)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "output").mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

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
    controlnet_vram_config = vram_config if args.controlnet_fp8 else {}
    model_configs = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors", **controlnet_vram_config),
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

    print(f"Processing {len(samples)} samples with Latent-DPS guidance")
    print(f"  scale={args.dps_guidance_scale}, schedule={args.dps_schedule}")

    original_step = pipe.step

    for idx, sample in enumerate(tqdm(samples, desc="Latent-DPS inference")):
        output_path = output_dir / "output" / f"output_{idx:04d}.png"
        if output_path.exists():
            continue

        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        gt_path = csv_path.parent / sample["image"]

        control_img = load_spad_image(control_path)
        gt_img = Image.open(gt_path).convert("RGB")

        pipe.load_models_to_device(["vae_encoder"])
        spad_pil = control_img.resize((args.width, args.height), Image.LANCZOS)
        spad_tensor = pipe.preprocess_image(spad_pil).to(device=pipe.device, dtype=pipe.torch_dtype)
        spad_latent = pipe.vae_encoder(spad_tensor, tiled=False)
        pipe.load_models_to_device([])
        torch.cuda.empty_cache()

        dps_config = LatentDPSConfig(
            spad_latent=spad_latent.detach(),
            guidance_scale=args.dps_guidance_scale,
            guidance_schedule=args.dps_schedule,
            start_step=args.dps_start_step,
            stop_step=args.dps_stop_step,
        )

        pipe.step = make_dps_step_fn(original_step, dps_config, args.steps)

        controlnet_input = ControlNetInput(
            image=control_img,
            processor_id="gray",
            scale=1.0,
        )

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

        control_img.save(output_dir / "input" / f"input_{idx:04d}.png")
        result_image.save(output_dir / "output" / f"output_{idx:04d}.png")
        gt_img.save(output_dir / "ground_truth" / f"gt_{idx:04d}.png")

    pipe.step = original_step

    print(f"\nDone! Results saved to {output_dir}")
    print(f"Run metrics: python run_metrics.py {output_dir} --save")


if __name__ == "__main__":
    main()
