#!/usr/bin/env python3
"""
Compute cross-frame variance targets from multi-frame generation outputs.

For each scene, loads the 7 different SPAD frame realizations (same scene,
same seed, different binary conditioning) and computes per-pixel output
variance across frames. This variance measures how much the model's output
depends on which specific SPAD frame it sees.

Adds crossframe_variance (scalar) and spatial_crossframe_variance (32x32 map)
to the existing targets.json used by linear_probing.py.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


PATCH_H, PATCH_W = 32, 32  # Match linear_probing.py spatial resolution


def main():
    parser = argparse.ArgumentParser(description="Compute cross-frame variance probing targets")
    parser.add_argument("--crossframe_dir", type=str, default="./validation_outputs_crossframe/baseline",
                        help="Directory with per-frame subfolders (bits/, bits_frame_1000/, etc.)")
    parser.add_argument("--targets_json", type=str, nargs="+",
                        default=[
                            "./probing_results_allblocks/targets.json",
                            "./probing_results_control/targets.json",
                            "./probing_results_no_cn/targets.json",
                        ],
                        help="targets.json files to update")
    parser.add_argument("--n_images", type=int, default=776)
    args = parser.parse_args()

    crossframe_dir = Path(args.crossframe_dir)

    # Discover frame folders
    frame_folders = sorted([
        d for d in crossframe_dir.iterdir()
        if d.is_dir() and (d / "output").exists() and d.name != "ground_truth"
    ])
    print(f"Found {len(frame_folders)} frame folders: {[f.name for f in frame_folders]}")

    # Verify image counts
    for ff in frame_folders:
        n = len(list((ff / "output").glob("*.png")))
        print(f"  {ff.name}: {n} images")

    crossframe_var = []
    spatial_crossframe_var = []
    n_missing = 0

    for idx in tqdm(range(args.n_images), desc="Computing cross-frame variance"):
        fname = f"output_{idx:04d}.png"
        arrs = []
        for ff in frame_folders:
            fp = ff / "output" / fname
            if fp.exists():
                img = np.array(Image.open(str(fp)).convert("RGB"), dtype=np.float32) / 255.0
                arrs.append(img)

        if len(arrs) >= 2:
            stacked = np.stack(arrs, axis=0)  # [K, H, W, 3]
            var_map = stacked.var(axis=0).mean(axis=-1)  # [H, W] — mean over RGB channels
            crossframe_var.append(float(var_map.mean()))

            # Spatial target: resize to patch grid
            var_normalized = np.clip(var_map / (var_map.max() + 1e-8), 0, 1)
            spatial = np.array(
                Image.fromarray((var_normalized * 255).astype(np.uint8)).resize(
                    (PATCH_W, PATCH_H), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
            spatial_crossframe_var.append(spatial.tolist())
        else:
            n_missing += 1
            crossframe_var.append(0.0)
            spatial_crossframe_var.append(np.zeros((PATCH_H, PATCH_W)).tolist())

    # Statistics
    valid = [v for v in crossframe_var if v > 0]
    print(f"\nCross-frame variance computed:")
    print(f"  Valid images: {len(valid)}/{args.n_images}")
    print(f"  Missing: {n_missing}")
    if valid:
        print(f"  Range: [{min(valid):.6f}, {max(valid):.6f}]")
        print(f"  Mean: {np.mean(valid):.6f}")
        print(f"  Std: {np.std(valid):.6f}")

    # Update each targets.json
    for tf_path in args.targets_json:
        tf = Path(tf_path)
        if not tf.exists():
            print(f"  SKIP (not found): {tf}")
            continue

        with open(tf) as f:
            targets = json.load(f)

        targets["crossframe_variance"] = crossframe_var
        targets["spatial_crossframe_variance"] = spatial_crossframe_var

        with open(tf, "w") as f:
            json.dump(targets, f)
        print(f"  Updated: {tf}")

    print("\nDone. Run global probing with: python linear_probing.py --train --output-dir <dir>")


if __name__ == "__main__":
    main()
