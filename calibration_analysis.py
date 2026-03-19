#!/usr/bin/env python3
"""
Phase 2f: Calibration Analysis

Builds empirical confidence intervals from K seed outputs and checks
coverage rates against ground truth. Produces calibration curves.

For each pixel, we build an empirical distribution from K seeds.
At nominal level alpha, we compute the alpha/2 and 1-alpha/2 quantiles.
Coverage = fraction of GT pixels falling within the confidence interval.

A well-calibrated model has coverage ~ nominal level for all levels.

Usage:
  python calibration_analysis.py \
    --seed-dirs seed_0/output seed_13/output ... \
    --gt-dir seed_0/ground_truth \
    --output-dir calibration/
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis from multi-seed outputs")
    parser.add_argument("--seed-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./calibration_analysis")
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all seed outputs
    print(f"Loading images from {len(args.seed_dirs)} seed directories...")
    seed_images: list[dict[str, np.ndarray]] = []
    for sd in args.seed_dirs:
        images = {}
        for f in sorted(Path(sd).glob("output_*.png")):
            img = np.array(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
            images[f.stem] = img
        seed_images.append(images)
        print(f"  {sd}: {len(images)} images")

    # Load GT
    gt_images = {}
    for f in sorted(Path(args.gt_dir).glob("gt_*.png")):
        img = np.array(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
        gt_images[f.stem] = img

    # Find common names (match output_NNNN to gt_NNNN)
    output_names = set(seed_images[0].keys())
    for si in seed_images[1:]:
        output_names &= set(si.keys())

    gt_to_output = {}
    for name in sorted(output_names):
        gt_name = name.replace("output_", "gt_")
        if gt_name in gt_images:
            gt_to_output[name] = gt_name

    print(f"  Matched pairs: {len(gt_to_output)}")

    # Compute calibration
    print("Computing calibration curves...")
    level_coverages = {level: [] for level in args.levels}

    for name, gt_name in tqdm(gt_to_output.items(), desc="Calibration"):
        # Stack seeds: [K, H, W, C]
        stack = np.stack([si[name] for si in seed_images], axis=0)
        gt = gt_images[gt_name]

        for level in args.levels:
            alpha = 1.0 - level
            lo = np.quantile(stack, alpha / 2, axis=0)
            hi = np.quantile(stack, 1.0 - alpha / 2, axis=0)
            covered = (gt >= lo) & (gt <= hi)
            coverage = float(np.mean(covered))
            level_coverages[level].append(coverage)

    # Aggregate
    calibration_curve = {}
    print(f"\n{'Nominal':>10s} | {'Empirical':>10s} | {'Gap':>10s}")
    print("-" * 36)
    for level in args.levels:
        empirical = float(np.mean(level_coverages[level]))
        gap = empirical - level
        calibration_curve[str(level)] = {
            "nominal": level,
            "empirical": empirical,
            "gap": gap,
            "std": float(np.std(level_coverages[level])),
        }
        print(f"{level:>10.2f} | {empirical:>10.4f} | {gap:>+10.4f}")

    # Expected calibration error (ECE)
    ece = float(np.mean([abs(v["gap"]) for v in calibration_curve.values()]))

    results = {
        "num_seeds": len(args.seed_dirs),
        "num_images": len(gt_to_output),
        "expected_calibration_error": ece,
        "calibration_curve": calibration_curve,
    }

    with open(out / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
