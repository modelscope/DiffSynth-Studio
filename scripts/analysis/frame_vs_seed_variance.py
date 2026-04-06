#!/usr/bin/env python3
"""
Phase 2b: Frame-vs-Seed Variance Decomposition

For selected validation views, decompose output variance into:
  - Measurement variance (across different binary frame realizations, same seed)
  - Seed variance (across different seeds, same frame)

Uses 7 frame folders (frame 0, 1000, 4000, 5000, 8000, 12000, 16000)
and K=10 seeds. Only uses views present in ALL frame folders (common subset).

Total variance = Var_frame[E_seed[X]] + E_frame[Var_seed[X]]
               = "measurement-driven"    + "seed-driven"

Usage:
  python frame_vs_seed_variance.py \
    --base-dir /path/to/multi_frame_seed_outputs \
    --output-dir variance_decomposition/
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


FRAME_FOLDERS = [
    "bits", "bits_frame_1000", "bits_frame_4000", "bits_frame_5000",
    "bits_frame_8000", "bits_frame_12000", "bits_frame_16000",
]

DEFAULT_SEEDS = [0, 13, 23, 42, 55, 67, 77, 88, 99, 123]


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Frame-vs-seed variance decomposition")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory with frame/seed subdirectories")
    parser.add_argument("--output-dir", type=str, default="./variance_decomposition")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Expected structure: base_dir/{frame_folder}/seed_{seed}/output/output_NNNN.png
    # Find common images across all frames and seeds
    print("Finding common images across frames and seeds...")

    common_names = None
    for frame in FRAME_FOLDERS:
        for seed in args.seeds:
            seed_dir = base / frame / f"seed_{seed}" / "output"
            if not seed_dir.exists():
                print(f"  WARNING: Missing {seed_dir}")
                continue
            names = {f.stem for f in seed_dir.glob("output_*.png")}
            if common_names is None:
                common_names = names
            else:
                common_names &= names

    if not common_names:
        print("ERROR: No common images found across all frames and seeds!")
        return

    common_names = sorted(common_names)
    if args.max_images:
        common_names = common_names[:args.max_images]

    print(f"  Common images: {len(common_names)}")
    print(f"  Frames: {len(FRAME_FOLDERS)}")
    print(f"  Seeds: {len(args.seeds)}")

    # Load all images: [frames, seeds, images, H, W, C]
    total_var_measurement = []
    total_var_seed = []
    total_var = []

    for name in tqdm(common_names, desc="Processing images"):
        # Stack: [F, K, H, W, C] where F=frames, K=seeds
        frame_seed_stack = []
        for frame in FRAME_FOLDERS:
            seed_stack = []
            for seed in args.seeds:
                img_path = base / frame / f"seed_{seed}" / "output" / f"{name}.png"
                if img_path.exists():
                    seed_stack.append(load_image(img_path))
                else:
                    seed_stack.append(np.zeros_like(seed_stack[0]) if seed_stack else np.zeros((512, 512, 3)))
            frame_seed_stack.append(np.stack(seed_stack, axis=0))

        data = np.stack(frame_seed_stack, axis=0)  # [F, K, H, W, C]

        # E_seed[X] for each frame: [F, H, W, C]
        mean_over_seeds = np.mean(data, axis=1)

        # Var_frame[E_seed[X]]: measurement-driven variance
        var_measurement = np.var(mean_over_seeds, axis=0)  # [H, W, C]

        # Var_seed[X] for each frame: [F, H, W, C]
        var_per_frame = np.var(data, axis=1)

        # E_frame[Var_seed[X]]: seed-driven variance
        var_seed = np.mean(var_per_frame, axis=0)  # [H, W, C]

        total_var_measurement.append(np.mean(var_measurement))
        total_var_seed.append(np.mean(var_seed))
        total_var.append(np.mean(var_measurement + var_seed))

    # Aggregate
    mean_var_measurement = float(np.mean(total_var_measurement))
    mean_var_seed = float(np.mean(total_var_seed))
    mean_total_var = float(np.mean(total_var))

    measurement_fraction = mean_var_measurement / (mean_total_var + 1e-10)
    seed_fraction = mean_var_seed / (mean_total_var + 1e-10)

    results = {
        "num_images": len(common_names),
        "num_frames": len(FRAME_FOLDERS),
        "num_seeds": len(args.seeds),
        "mean_measurement_variance": mean_var_measurement,
        "mean_seed_variance": mean_var_seed,
        "mean_total_variance": mean_total_var,
        "measurement_fraction": measurement_fraction,
        "seed_fraction": seed_fraction,
    }

    with open(out / "variance_decomposition.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Variance Decomposition ---")
    print(f"  Measurement-driven: {mean_var_measurement:.6f} ({measurement_fraction:.1%})")
    print(f"  Seed-driven:        {mean_var_seed:.6f} ({seed_fraction:.1%})")
    print(f"  Total:              {mean_total_var:.6f}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
