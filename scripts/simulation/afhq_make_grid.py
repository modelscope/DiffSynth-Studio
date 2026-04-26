#!/usr/bin/env python3
"""
Build the AFHQ-simulation comparison grid.

Layout: rows = scenes, columns = [source sRGB, predicted mono, K=1, K=10, K=100, K=1000, K=10000].
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/ondemand30/jw954/afhq_simulation")
SEL = ROOT / "selected_images"
EXTRACTS = ROOT / "extracts"
OUT_PDF = ROOT / "figures" / "afhq_simulation_grid.pdf"
OUT_PNG = ROOT / "figures" / "afhq_simulation_grid.png"

K_VALUES = [1, 10, 100, 1000, 10000]


def load_p_png(p):
    """Load a 16-bit (or 8-bit) probability PNG, return float32 in [0,1]."""
    img = Image.open(p)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32) / 255.0


def main():
    weights_npz = np.load("/nfs/horai.dgpsrv/ondemand30/jw954/calibration/rgb_to_mono_weights_v4.npz")
    w = (float(weights_npz["w_r"]), float(weights_npz["w_g"]), float(weights_npz["w_b"]))
    print(f"Using weights: {w}")

    selected = sorted(SEL.glob("*.png"))
    n_scenes = len(selected)
    n_cols = 2 + len(K_VALUES)   # source + predicted + 5 accumulations

    fig, axes = plt.subplots(n_scenes, n_cols,
                              figsize=(2 * n_cols, 2 * n_scenes))
    if n_scenes == 1:
        axes = axes[None, :]

    col_titles = ["AFHQ source"] + ["predicted mono\n(w · linear_RGB)"] + \
                 [f"K={K} frames" for K in K_VALUES]

    for r, src in enumerate(selected):
        scene_id = src.stem
        # Source RGB
        srgb = np.array(Image.open(src).convert("RGB"))
        ax = axes[r, 0]
        ax.imshow(srgb)
        ax.axis("off")
        if r == 0:
            ax.set_title(col_titles[0], fontsize=10)
        ax.text(-0.15, 0.5, scene_id.replace("afhq_", "").replace("_", "\n"),
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8, family="monospace")

        # Predicted mono (gamma-decoded RGB · weights)
        srgb_f = srgb.astype(np.float32) / 255.0
        # sRGB inverse OETF
        a = 0.055
        lin = np.where(srgb_f <= 0.04045,
                        srgb_f / 12.92,
                        ((srgb_f + a) / (1 + a)) ** 2.4)
        mono = lin[..., 0] * w[0] + lin[..., 1] * w[1] + lin[..., 2] * w[2]
        ax = axes[r, 1]
        ax.imshow(mono, cmap="gray", vmin=0, vmax=mono.max())
        ax.axis("off")
        if r == 0:
            ax.set_title(col_titles[1], fontsize=10)

        # Accumulations at each K
        for ci, K in enumerate(K_VALUES):
            png = EXTRACTS / f"frames_{K}" / f"{scene_id}_RAW_empty_frames0-{K-1}_p.png"
            if not png.exists():
                ax = axes[r, 2 + ci]
                ax.text(0.5, 0.5, "missing",
                        ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            p_arr = load_p_png(png)
            ax = axes[r, 2 + ci]
            ax.imshow(p_arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[2 + ci], fontsize=10)

    fig.suptitle(
        "AFHQ → simulated SPAD: frame-accumulation series",
        fontsize=14,
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
