#!/usr/bin/env python3
"""
Best-of-K Reranking: For each image, select the seed whose output best
explains the SPAD observation (lowest measurement NLL).

Uses the existing 10-seed generations in validation_outputs_multiseed/.
No GPU inference needed — just loads images and computes NLL.
"""

import argparse
import json
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from diffsynth.diffusion.spad_forward import SPADForwardModel, srgb_to_linear


def compute_nll(output_path, input_path, spad_model, to_tensor):
    """Compute per-pixel mean NLL for one image pair."""
    out = to_tensor(Image.open(output_path).convert("RGB")).unsqueeze(0)
    inp = to_tensor(Image.open(input_path).convert("RGB")).unsqueeze(0)

    linear = srgb_to_linear(out)
    intensity = linear.mean(dim=1, keepdim=True)
    inp_gray = inp.mean(dim=1, keepdim=True)

    H = spad_model.intensity_to_exposure(intensity)
    log_p = torch.log(-torch.expm1(-H))
    log_1mp = -H
    nll_map = -(inp_gray * log_p + (1.0 - inp_gray) * log_1mp)
    return nll_map.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Best-of-K reranking by measurement NLL")
    parser.add_argument("--multiseed_dir", type=str, default="./validation_outputs_multiseed")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs_best_of_k")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()

    multiseed_dir = Path(args.multiseed_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "output").mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth").mkdir(parents=True, exist_ok=True)

    # Find all seed directories
    seed_dirs = sorted([d for d in multiseed_dir.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")])
    print(f"Found {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")

    # Count images
    n_images = len(list((seed_dirs[0] / "output").glob("*.png")))
    print(f"Images per seed: {n_images}")

    spad_model = SPADForwardModel(alpha=args.alpha, beta=args.beta, num_frames=1)
    to_tensor = transforms.ToTensor()

    selections = {}
    nll_baseline = []  # NLL of seed_42
    nll_best = []      # NLL of best-of-K

    for idx in tqdm(range(n_images), desc="Best-of-K selection"):
        best_seed = None
        best_nll = float("inf")
        seed_nlls = {}

        for seed_dir in seed_dirs:
            out_path = seed_dir / "output" / f"output_{idx:04d}.png"
            inp_path = seed_dir / "input" / f"input_{idx:04d}.png"

            if not out_path.exists() or not inp_path.exists():
                continue

            with torch.no_grad():
                nll = compute_nll(out_path, inp_path, spad_model, to_tensor)

            seed_nlls[seed_dir.name] = nll

            if nll < best_nll:
                best_nll = nll
                best_seed = seed_dir

        if best_seed is None:
            continue

        selections[idx] = {"seed": best_seed.name, "nll": best_nll}
        nll_best.append(best_nll)

        # Track seed_42 NLL for comparison
        if "seed_42" in seed_nlls:
            nll_baseline.append(seed_nlls["seed_42"])

        # Copy/symlink best output
        src_out = best_seed / "output" / f"output_{idx:04d}.png"
        src_inp = best_seed / "input" / f"input_{idx:04d}.png"
        src_gt = best_seed / "ground_truth" / f"gt_{idx:04d}.png"

        shutil.copy2(src_out, output_dir / "output" / f"output_{idx:04d}.png")
        shutil.copy2(src_inp, output_dir / "input" / f"input_{idx:04d}.png")
        if src_gt.exists() and not (output_dir / "ground_truth" / f"gt_{idx:04d}.png").exists():
            shutil.copy2(src_gt, output_dir / "ground_truth" / f"gt_{idx:04d}.png")

    # Statistics
    seed_counts = {}
    for info in selections.values():
        s = info["seed"]
        seed_counts[s] = seed_counts.get(s, 0) + 1

    print(f"\n{'='*60}")
    print(f"Best-of-{len(seed_dirs)} Reranking Results")
    print(f"{'='*60}")
    print(f"  Images: {len(selections)}")
    print(f"  Mean NLL (seed_42 baseline): {np.mean(nll_baseline):.6f}")
    print(f"  Mean NLL (best-of-K):        {np.mean(nll_best):.6f}")
    print(f"  NLL improvement:             {np.mean(nll_baseline) - np.mean(nll_best):.6f}")
    print(f"\nSeed selection distribution:")
    for seed, count in sorted(seed_counts.items(), key=lambda x: -x[1]):
        print(f"  {seed}: {count} images ({100*count/len(selections):.1f}%)")

    # Save metadata
    meta = {
        "method": f"best_of_{len(seed_dirs)}_by_nll",
        "seeds": [d.name for d in seed_dirs],
        "n_images": len(selections),
        "mean_nll_baseline": float(np.mean(nll_baseline)),
        "mean_nll_best": float(np.mean(nll_best)),
        "seed_distribution": seed_counts,
        "per_image": {str(k): v for k, v in selections.items()},
    }
    with open(output_dir / "best_of_k_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput saved to {output_dir}")
    print(f"Run metrics: python run_metrics.py {output_dir} --save")


if __name__ == "__main__":
    main()
