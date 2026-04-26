#!/usr/bin/env python3
"""
Side-by-side comparison: AFHQ outputs from 3-level fallback vs full pyramid.

For each of the 10 cherry-picked AFHQ scenes, show:
  source sRGB | predicted mono | 3-level K=10 | pyramid K=10 | 3-level K=100 |
  pyramid K=100 | 3-level K=10000 | pyramid K=10000

The two simulators used identical seed=42, weights, exposure α=4, and source
images. The only difference is HOW the per-pixel p̂ was sampled from the LUT
(single-level lookup with binary fallback vs cascade walk with min_samples=20).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = Path("/scratch/ondemand30/jw954/afhq_simulation_3level")
NEW = Path("/scratch/ondemand30/jw954/afhq_simulation_pyramid")
OUT_PDF = NEW / "figures" / "comparison_3level_vs_pyramid.pdf"
OUT_PNG = NEW / "figures" / "comparison_3level_vs_pyramid.png"

K_TO_SHOW = [10, 100, 10000]


def load_p_png(p):
    img = Image.open(p)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32) / 255.0


def main():
    selected = sorted((NEW / "selected_images").glob("*.png"))
    n_scenes = len(selected)

    # Columns: source | predicted mono | (3-level Kx, pyramid Kx) for each K
    n_cols = 2 + 2 * len(K_TO_SHOW)
    fig, axes = plt.subplots(n_scenes, n_cols, figsize=(2 * n_cols, 2 * n_scenes))
    if n_scenes == 1:
        axes = axes[None, :]

    titles = ["AFHQ source", "predicted mono"]
    for K in K_TO_SHOW:
        titles.extend([f"3-level\nK={K}", f"pyramid\nK={K}"])

    # Load v4 weights
    wnpz = np.load("/nfs/horai.dgpsrv/ondemand30/jw954/calibration/rgb_to_mono_weights_v4.npz")
    w = (float(wnpz["w_r"]), float(wnpz["w_g"]), float(wnpz["w_b"]))

    for r, src in enumerate(selected):
        scene_id = src.stem
        srgb = np.array(Image.open(src).convert("RGB"))
        ax = axes[r, 0]
        ax.imshow(srgb)
        ax.axis("off")
        if r == 0:
            ax.set_title(titles[0], fontsize=10)
        ax.text(-0.18, 0.5, scene_id.replace("afhq_", "").replace("_", "\n"),
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8, family="monospace")

        # Predicted mono
        srgb_f = srgb.astype(np.float32) / 255.0
        a = 0.055
        lin = np.where(srgb_f <= 0.04045, srgb_f / 12.92,
                        ((srgb_f + a) / (1 + a)) ** 2.4)
        mono = lin[..., 0] * w[0] + lin[..., 1] * w[1] + lin[..., 2] * w[2]
        ax = axes[r, 1]
        ax.imshow(mono, cmap="gray", vmin=0, vmax=mono.max())
        ax.axis("off")
        if r == 0:
            ax.set_title(titles[1], fontsize=10)

        for ki, K in enumerate(K_TO_SHOW):
            for di, (root, label) in enumerate([(OLD, "3-level"), (NEW, "pyramid")]):
                col = 2 + 2 * ki + di
                png = root / f"extracts/frames_{K}" / f"{scene_id}_RAW_empty_frames0-{K-1}_p.png"
                ax = axes[r, col]
                if not png.exists():
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                    ax.axis("off")
                    continue
                p_arr = load_p_png(png)
                ax.imshow(p_arr, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if r == 0:
                    ax.set_title(titles[col], fontsize=10)

    fig.suptitle(
        "AFHQ → simulated SPAD: 3-level (super-pixel-only) fallback vs full dyadic pyramid cascade\n"
        "Same source images, same seed=42, same exposure α=4. Only the LUT sampling strategy differs.",
        fontsize=13,
    )
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
