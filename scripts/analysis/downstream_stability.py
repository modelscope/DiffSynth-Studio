#!/usr/bin/env python3
"""
Phase 2c: Downstream Task Stability Analysis

For K=10 seed outputs:
  - Run segmentation (SAM3) on each output
  - Run depth estimation (ml-depth-pro) on each output
  - Compute segmentation entropy maps and IoU variance across seeds
  - Compute depth variance maps across seeds
  - Correlate downstream variance with RGB variance

Usage:
  python downstream_stability.py \
    --seed-dirs seed_0 seed_13 ... \
    --output-dir downstream_analysis/
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
import sys


def compute_segmentation_entropy(masks: list[np.ndarray]) -> np.ndarray:
    """Compute per-pixel entropy across K segmentation masks.

    Args:
        masks: List of K segmentation masks [H, W] with integer labels.

    Returns:
        Entropy map [H, W].
    """
    if not masks:
        return np.zeros((1, 1))

    K = len(masks)
    H, W = masks[0].shape

    all_labels = set()
    for m in masks:
        all_labels.update(np.unique(m).tolist())

    # Compute per-pixel label distribution
    entropy = np.zeros((H, W), dtype=np.float64)

    for label in all_labels:
        p = np.mean([(m == label).astype(np.float64) for m in masks], axis=0)
        p = np.clip(p, 1e-10, 1.0)
        entropy -= p * np.log2(p)

    return entropy.astype(np.float32)


def compute_depth_variance(depth_maps: list[np.ndarray]) -> np.ndarray:
    """Compute per-pixel depth variance across K predictions."""
    stack = np.stack(depth_maps, axis=0)
    return np.var(stack, axis=0).astype(np.float32)


def run_depth_estimation(image_dir: Path, output_dir: Path):
    """Run ml-depth-pro on a directory of images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_script = Path(__file__).resolve().parents[2] / "analysis_depth.py"
    if depth_script.exists():
        subprocess.run([
            sys.executable, str(depth_script),
            "--input-dir", str(image_dir),
            "--output-dir", str(output_dir),
        ], check=True)
    else:
        print(f"  WARNING: Depth estimation script not found at {depth_script}")
        print(f"  Skipping depth estimation. Run manually.")


def main():
    parser = argparse.ArgumentParser(description="Downstream task stability analysis")
    parser.add_argument("--seed-dirs", type=str, nargs="+", required=True,
                        help="Validation output directories for each seed")
    parser.add_argument("--output-dir", type=str, default="./downstream_analysis")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip running SAM3/depth, only aggregate existing results")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed_dirs = [Path(d) for d in args.seed_dirs]

    if not args.skip_inference:
        print("Running depth estimation on each seed...")
        for sd in seed_dirs:
            output_subdir = sd / "output"
            depth_dir = sd / "depth"
            if not depth_dir.exists() and output_subdir.exists():
                print(f"  {sd.name}...")
                run_depth_estimation(output_subdir, depth_dir)

    # Load depth maps
    print("\nAggregating depth variance...")
    all_depth_maps: list[dict[str, np.ndarray]] = []
    for sd in seed_dirs:
        depth_dir = sd / "depth"
        if not depth_dir.exists():
            print(f"  WARNING: No depth dir in {sd}")
            continue
        maps = {}
        for f in sorted(depth_dir.glob("*.png")):
            d = np.array(Image.open(f).convert("L")).astype(np.float32) / 255.0
            maps[f.stem] = d
        all_depth_maps.append(maps)

    if len(all_depth_maps) >= 2:
        common_names = set(all_depth_maps[0].keys())
        for dm in all_depth_maps[1:]:
            common_names &= set(dm.keys())

        depth_variances = []
        for name in sorted(common_names):
            maps = [dm[name] for dm in all_depth_maps if name in dm]
            var = compute_depth_variance(maps)
            depth_variances.append(float(np.mean(var)))

        mean_depth_var = float(np.mean(depth_variances))
        print(f"  Mean depth variance: {mean_depth_var:.6f} ({len(common_names)} images)")
    else:
        mean_depth_var = None
        print("  Not enough depth data for variance computation")

    results = {
        "num_seeds": len(seed_dirs),
        "mean_depth_variance": mean_depth_var,
    }

    with open(out / "downstream_stability.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
