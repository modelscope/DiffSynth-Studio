#!/usr/bin/env python3
"""
Fast pixel-space DPS hyperparameter sweep on N samples.
Loads the pipeline ONCE, then iterates over all (eta, schedule, step_range) configs.
"""

import argparse
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import numpy as np
from itertools import product

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict
from diffsynth.diffusion.flow_dps import FlowDPSConfig, compute_dps_correction, get_guidance_weight
from diffsynth.diffusion.spad_forward import SPADForwardModel

from validate_flow_dps import load_spad_image, load_spad_measurement, make_pixel_dps_step_fn


def quick_metrics(output_dir, baseline_dir=None):
    """Compute lightweight metrics (no FID/CFID -- too few samples)."""
    import torchvision.transforms as transforms
    from diffsynth.diffusion.spad_forward import SPADForwardModel, srgb_to_linear

    output_dir = Path(output_dir)
    out_imgs = sorted((output_dir / "output").glob("*.png"))
    gt_imgs = sorted((output_dir / "ground_truth").glob("*.png"))
    inp_imgs = sorted((output_dir / "input").glob("*.png"))

    if not out_imgs:
        return {}

    to_tensor = transforms.ToTensor()
    spad_model = SPADForwardModel(alpha=1.0, beta=0.0, num_frames=1)

    psnr_vals, ssim_vals, nll_vals = [], [], []
    delta_vs_baseline = []

    for i in range(len(out_imgs)):
        out = to_tensor(Image.open(out_imgs[i]).convert("RGB")).unsqueeze(0)
        gt = to_tensor(Image.open(gt_imgs[i]).convert("RGB")).unsqueeze(0)

        # Grayscale for PSNR
        out_gray = out.mean(dim=1, keepdim=True)
        gt_gray = gt.mean(dim=1, keepdim=True)
        mse = ((out_gray - gt_gray) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        psnr_vals.append(psnr)

        # Measurement NLL
        if inp_imgs:
            inp = to_tensor(Image.open(inp_imgs[i]).convert("RGB")).unsqueeze(0)
            inp_gray = inp.mean(dim=1, keepdim=True)
            linear = srgb_to_linear(out)
            intensity = linear.mean(dim=1, keepdim=True)
            H = spad_model.intensity_to_exposure(intensity)
            log_p = torch.log(-torch.expm1(-H))
            log_1mp = -H
            nll_map = -(inp_gray * log_p + (1.0 - inp_gray) * log_1mp)
            nll_vals.append(nll_map.mean().item())

        # Compare to baseline if available
        if baseline_dir:
            bl_path = Path(baseline_dir) / "output" / out_imgs[i].name
            if bl_path.exists():
                bl = to_tensor(Image.open(bl_path).convert("RGB")).unsqueeze(0)
                bl_gray = bl.mean(dim=1, keepdim=True)
                bl_mse = ((bl_gray - gt_gray) ** 2).mean().item()
                bl_psnr = 10 * np.log10(1.0 / max(bl_mse, 1e-10))
                delta_vs_baseline.append(psnr - bl_psnr)

    results = {
        "psnr": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "n_samples": len(psnr_vals),
    }
    if nll_vals:
        results["measurement_nll"] = float(np.mean(nll_vals))
    if delta_vs_baseline:
        results["delta_psnr_vs_baseline"] = float(np.mean(delta_vs_baseline))
        results["pct_better_than_baseline"] = float(np.mean([d > 0 for d in delta_vs_baseline]) * 100)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pixel-space DPS hyperparameter sweep")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output_root", type=str, default="./sweep_pixel_dps_results")
    parser.add_argument("--baseline_dir", type=str, default="./validation_outputs_multiseed/seed_42")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Focused sweep: nll_mode is the key variable (fixes dark-collapse problem)
    # Test balanced + detections modes with moderate etas
    etas = [0.1, 0.5, 1.0]
    nll_modes = ["balanced", "detections"]
    step_ranges = [
        (0, -1, "full"),
        (4, 14, "mid_4_14"),
    ]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
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

    print(f"Loading LoRA: {args.lora_checkpoint}")
    state_dict = load_state_dict(args.lora_checkpoint, torch_dtype=pipe.torch_dtype, device=pipe.device)
    loader = FluxLoRALoader(torch_dtype=pipe.torch_dtype, device=pipe.device)
    loader.fuse_lora_to_base_model(pipe.controlnet, state_dict, alpha=1.0)

    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))[:args.max_samples]

    spad_model = SPADForwardModel(alpha=1.0, beta=0.0, num_frames=1).cuda()
    original_step = pipe.step

    # Preload control images and measurements
    print(f"Preloading {len(samples)} samples...")
    sample_data = []
    for sample in samples:
        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        gt_path = csv_path.parent / sample["image"]
        control_img = load_spad_image(control_path)
        gt_img = Image.open(gt_path).convert("RGB")
        spad_meas = load_spad_measurement(control_path, args.height, args.width).to(device="cuda", dtype=torch.float32)
        sample_data.append((sample, control_img, gt_img, spad_meas))

    configs = list(product(etas, nll_modes, step_ranges))
    all_results = {}

    print(f"\n{'='*80}")
    print(f"SWEEP: {len(configs)} configs × {len(samples)} samples = {len(configs)*len(samples)} runs")
    print(f"{'='*80}\n")

    for ci, (eta, nll_mode, (start, stop, range_name)) in enumerate(configs):
        config_name = f"{nll_mode}_eta{eta}_{range_name}"
        config_dir = output_root / config_name
        (config_dir / "input").mkdir(parents=True, exist_ok=True)
        (config_dir / "output").mkdir(parents=True, exist_ok=True)
        (config_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

        # Skip if already done
        existing = list((config_dir / "output").glob("*.png"))
        if len(existing) >= len(samples):
            print(f"[{ci+1}/{len(configs)}] {config_name}: already done, computing metrics...")
            results = quick_metrics(config_dir, args.baseline_dir)
            all_results[config_name] = results
            continue

        print(f"[{ci+1}/{len(configs)}] {config_name} (eta={eta}, mode={nll_mode}, steps={start}-{stop})")

        for idx, (sample, control_img, gt_img, spad_meas) in enumerate(tqdm(sample_data, desc=config_name, leave=False)):
            output_path = config_dir / "output" / f"output_{idx:04d}.png"
            if output_path.exists():
                continue

            dps_config = FlowDPSConfig(
                spad_measurement=spad_meas,
                alpha=1.0, beta=0.0, num_frames=1,
                guidance_scale=eta,
                guidance_schedule="constant",
                start_step=start,
                stop_step=stop,
                nll_mode=nll_mode,
                gradient_clamp=1.0,
            )

            pipe.step = make_pixel_dps_step_fn(original_step, pipe, dps_config, spad_model, args.steps)

            controlnet_input = ControlNetInput(image=control_img, processor_id="gray", scale=1.0)
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

            control_img.save(config_dir / "input" / f"input_{idx:04d}.png")
            result_image.save(config_dir / "output" / f"output_{idx:04d}.png")
            gt_img.save(config_dir / "ground_truth" / f"gt_{idx:04d}.png")

        pipe.step = original_step

        results = quick_metrics(config_dir, args.baseline_dir)
        all_results[config_name] = results
        print(f"  → PSNR={results.get('psnr', 0):.2f} dB, "
              f"NLL={results.get('measurement_nll', 0):.4f}, "
              f"Δ={results.get('delta_psnr_vs_baseline', 0):+.2f} dB, "
              f"better={results.get('pct_better_than_baseline', 0):.0f}%")

    # Save summary
    with open(output_root / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print leaderboard
    print(f"\n{'='*100}")
    print(f"{'Config':<45} {'PSNR':>8} {'NLL':>10} {'ΔPSNR':>8} {'%Better':>8}")
    print(f"{'='*100}")

    sorted_configs = sorted(all_results.items(),
                            key=lambda x: x[1].get("measurement_nll", 999))
    for name, r in sorted_configs:
        print(f"{name:<45} {r.get('psnr',0):>8.2f} {r.get('measurement_nll',0):>10.4f} "
              f"{r.get('delta_psnr_vs_baseline',0):>+8.2f} {r.get('pct_better_than_baseline',0):>7.0f}%")

    print(f"\nResults saved to {output_root / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
