#!/usr/bin/env python3
"""
Pixel-Space DPS Validation -- Physics-consistent measurement-guided sampling.

Uses the full SPAD forward model (Bernoulli NLL) in pixel space. At each
denoising step, decodes the predicted clean latent through the VAE, computes
the Bernoulli measurement likelihood gradient, and corrects the velocity.

Unlike latent DPS (||x_0_hat - z_spad||^2), this directly optimizes
-log p(y | D(x_0)), the actual measurement likelihood.

Uses the same monkeypatch-step approach as validate_dps.py so that the
pipeline's own VRAM management handles model offloading correctly.
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
from diffsynth.diffusion.flow_dps import FlowDPSConfig, compute_dps_correction, get_guidance_weight
from diffsynth.diffusion.spad_forward import SPADForwardModel


def load_spad_image(path) -> Image.Image:
    """Load a SPAD image, handling 16-bit grayscale correctly."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


def load_spad_measurement(path, height, width) -> torch.Tensor:
    """Load SPAD binary frame as [1, 1, H, W] tensor with values in {0, 1}."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (1.0 / 65535.0)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    arr_img = Image.fromarray((arr * 255).astype(np.uint8))
    arr_img = arr_img.resize((width, height), Image.NEAREST)
    arr = np.array(arr_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def make_pixel_dps_step_fn(original_step, pipe, dps_config, spad_model, total_steps):
    """Wrap the pipeline's step to inject pixel-space physics DPS correction."""

    def pixel_dps_step(scheduler, latents, progress_id, noise_pred, **kwargs):
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

        if weight > 0 and dps_config.spad_measurement is not None and sigma > 0.01:
            # Offload DiT/ControlNet, load VAE decoder for DPS gradient
            pipe.load_models_to_device([])
            torch.cuda.empty_cache()
            pipe.load_models_to_device(["vae_decoder"])

            # Re-enable gradients: pipe.__call__ runs under @torch.no_grad()
            # but we need autograd for the physics DPS gradient through the VAE
            with torch.enable_grad():
                correction = compute_dps_correction(
                    latents=latents,
                    noise_pred=noise_pred,
                    sigma=sigma,
                    vae_decoder=pipe.vae_decoder,
                    spad_measurement=dps_config.spad_measurement,
                    spad_model=spad_model,
                    guidance_scale=dps_config.guidance_scale * weight,
                    gradient_clamp=dps_config.gradient_clamp,
                    use_l2=dps_config.use_l2_loss,
                    use_nll=dps_config.use_nll_loss,
                    nll_mode=dps_config.nll_mode,
                    device=pipe.device,
                    tiled=True,
                    tile_size=64,
                    tile_stride=32,
                )
            noise_pred = noise_pred + correction
            del correction

            # Reload iteration models for next step
            pipe.load_models_to_device([])
            torch.cuda.empty_cache()
            pipe.load_models_to_device(pipe.in_iteration_models)

        return original_step(scheduler, latents=latents, progress_id=progress_id, noise_pred=noise_pred, **kwargs)

    return pixel_dps_step


def main():
    parser = argparse.ArgumentParser(description="FLUX validation with pixel-space physics DPS")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--lora_target", type=str, default="controlnet", choices=["dit", "controlnet"])
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_flow_dps")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)

    # Physics DPS parameters
    parser.add_argument("--dps_eta", type=float, default=0.1, help="DPS guidance strength")
    parser.add_argument("--dps_schedule", type=str, default="ramp_up",
                        choices=["constant", "linear_decay", "cosine", "ramp_up", "sigma_ramp"])
    parser.add_argument("--dps_start_step", type=int, default=0)
    parser.add_argument("--dps_stop_step", type=int, default=-1)
    parser.add_argument("--dps_alpha", type=float, default=1.0, help="SPAD forward model sensitivity")
    parser.add_argument("--dps_beta", type=float, default=0.0, help="SPAD forward model offset")
    parser.add_argument("--gradient_clamp", type=float, default=1.0)
    parser.add_argument("--nll_mode", type=str, default="balanced",
                        choices=["full", "balanced", "detections"],
                        help="NLL weighting: full (standard), balanced (equal weight SPAD 0/1), detections (SPAD=1 only)")

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

    spad_model = SPADForwardModel(
        alpha=args.dps_alpha, beta=args.dps_beta, num_frames=1
    ).cuda()

    print(f"Processing {len(samples)} samples with pixel-space physics DPS")
    print(f"  eta={args.dps_eta}, schedule={args.dps_schedule}, "
          f"steps={args.dps_start_step}-{args.dps_stop_step}, "
          f"alpha={args.dps_alpha}, beta={args.dps_beta}")

    original_step = pipe.step

    for idx, sample in enumerate(tqdm(samples, desc="Pixel-DPS inference")):
        output_path = output_dir / "output" / f"output_{idx:04d}.png"
        if output_path.exists():
            continue

        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        gt_path = csv_path.parent / sample["image"]

        control_img = load_spad_image(control_path)
        gt_img = Image.open(gt_path).convert("RGB")

        # Load SPAD binary measurement for the physics model
        spad_measurement = load_spad_measurement(
            control_path, args.height, args.width
        ).to(device="cuda", dtype=torch.float32)

        dps_config = FlowDPSConfig(
            spad_measurement=spad_measurement,
            alpha=args.dps_alpha,
            beta=args.dps_beta,
            num_frames=1,
            guidance_scale=args.dps_eta,
            guidance_schedule=args.dps_schedule,
            start_step=args.dps_start_step,
            stop_step=args.dps_stop_step,
            nll_mode=args.nll_mode,
            gradient_clamp=args.gradient_clamp,
        )

        # Monkeypatch the step function to inject pixel-space DPS
        pipe.step = make_pixel_dps_step_fn(
            original_step, pipe, dps_config, spad_model, args.steps
        )

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
