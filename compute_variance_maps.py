#!/usr/bin/env python3
"""
Phase 2a: Per-pixel RGB variance maps across K seeds.

For each validation image, loads K generated outputs (from different seeds),
computes per-pixel variance, and correlates with SPAD bit density.

Usage:
  python compute_variance_maps.py \
    --seed-dirs seed_0/output seed_13/output ... \
    --gt-dir seed_0/ground_truth \
    --control-dir seed_0/input \
    --output-dir variance_analysis/
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


def load_images_as_array(image_dir: Path, prefix: str = "") -> dict[str, np.ndarray]:
    """Load all images from a directory into a dict of numpy arrays."""
    images = {}
    for f in sorted(image_dir.glob(f"{prefix}*.png")):
        img = np.array(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
        images[f.stem] = img
    return images


def compute_variance_maps(seed_images: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Compute per-pixel variance across seeds for each image."""
    all_names = set(seed_images[0].keys())
    for si in seed_images[1:]:
        all_names &= set(si.keys())

    variance_maps = {}
    mean_maps = {}
    for name in sorted(all_names):
        stack = np.stack([si[name] for si in seed_images if name in si], axis=0)
        variance_maps[name] = np.var(stack, axis=0)
        mean_maps[name] = np.mean(stack, axis=0)

    return variance_maps, mean_maps


def compute_bit_density(control_images: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute mean bit density for each SPAD control image."""
    densities = {}
    for name, img in control_images.items():
        densities[name] = float(np.mean(img))
    return densities


def main():
    parser = argparse.ArgumentParser(description="Compute per-pixel variance maps across seeds")
    parser.add_argument("--seed-dirs", type=str, nargs="+", required=True,
                        help="Directories containing output images from different seeds")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Ground truth directory")
    parser.add_argument("--control-dir", type=str, required=True,
                        help="Control (SPAD) image directory")
    parser.add_argument("--output-dir", type=str, default="./variance_analysis")
    parser.add_argument("--save-maps", action="store_true",
                        help="Save variance heatmap images")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {len(args.seed_dirs)} seed directories...")
    seed_images = []
    for sd in args.seed_dirs:
        imgs = load_images_as_array(Path(sd))
        seed_images.append(imgs)
        print(f"  {sd}: {len(imgs)} images")

    print("Computing variance maps...")
    variance_maps, mean_maps = compute_variance_maps(seed_images)

    gt_images = load_images_as_array(Path(args.gt_dir))
    control_images = load_images_as_array(Path(args.control_dir))
    bit_densities = compute_bit_density(control_images)

    per_image_variance = {}
    per_image_mean_error = {}
    bit_density_list = []
    variance_list = []

    for name in sorted(variance_maps.keys()):
        var_map = variance_maps[name]
        mean_var = float(np.mean(var_map))
        per_image_variance[name] = mean_var

        gt_name = name.replace("output_", "gt_")
        if gt_name in gt_images:
            error = np.mean((mean_maps[name] - gt_images[gt_name]) ** 2)
            per_image_mean_error[name] = float(error)

        ctrl_name = name.replace("output_", "input_")
        if ctrl_name in bit_densities:
            bit_density_list.append(bit_densities[ctrl_name])
            variance_list.append(mean_var)

        if args.save_maps:
            var_gray = np.mean(var_map, axis=2)
            var_normalized = var_gray / (var_gray.max() + 1e-8)
            var_img = (var_normalized * 255).astype(np.uint8)
            Image.fromarray(var_img).save(out / f"variance_{name}.png")

    # Correlation analysis
    if bit_density_list:
        correlation = float(np.corrcoef(bit_density_list, variance_list)[0, 1])
        print(f"\nBit density vs variance correlation: {correlation:.4f}")
    else:
        correlation = None

    summary = {
        "num_seeds": len(args.seed_dirs),
        "num_images": len(variance_maps),
        "mean_variance": float(np.mean(list(per_image_variance.values()))),
        "std_variance": float(np.std(list(per_image_variance.values()))),
        "bit_density_variance_correlation": correlation,
        "per_image_variance": per_image_variance,
        "per_image_mean_error": per_image_mean_error,
    }

    with open(out / "variance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean per-pixel variance: {summary['mean_variance']:.6f}")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
