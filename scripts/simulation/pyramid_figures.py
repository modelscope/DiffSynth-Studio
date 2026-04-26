#!/usr/bin/env python3
"""
Pyramid diagnostic figures
==========================
Version: v1-pyramid

Two figures:
  1. Fallback heat-map at GT bin 77 (≈ p=0.30): each pixel coloured by the
     coarsest level it can be resolved at (min_samples=20).
  2. Bar chart of fraction of 1M queries resolved at each pyramid level.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from pyramid_sampler import PyramidSampler, LEVELS

H, W = 512, 512
N_BINS = 256
FIG_DIR = Path("/nfs/horai.dgpsrv/ondemand30/jw954/calibration/figures")
PYRAMID_PATH = "/nfs/horai.dgpsrv/ondemand30/jw954/calibration/lut_pyramid.npz"
USAGE_STATS_PATH = "/nfs/horai.dgpsrv/ondemand30/jw954/calibration/pyramid_usage_stats.npz"


def fig_fallback_map(min_samples=20, gt_bin=77):
    """Per-pixel heat-map of coarsest level needed at given GT bin."""
    sampler = PyramidSampler(PYRAMID_PATH, min_samples=min_samples)
    print(f"\nComputing fallback level map for bin {gt_bin} (p≈{(gt_bin+0.5)/N_BINS:.3f}) "
          f"with min_samples={min_samples} …")
    fmap = sampler.fallback_level_map(gt_bin, min_samples=min_samples)
    print(f"  Coverage: per-level pixel counts")
    for L in LEVELS:
        n = int((fmap == L).sum())
        pct = 100 * n / (H * W)
        print(f"    L={L:>3}: {n:>7,} pixels ({pct:5.2f}%)")
    n_un = int((fmap == -1).sum())
    if n_un:
        print(f"    Unresolved: {n_un} pixels ({100*n_un/(H*W):.4f}%)")

    # Map level -> log-scale color index for clean colormap
    level_to_idx = {L: i for i, L in enumerate(LEVELS)}
    img = np.full((H, W), -1, dtype=np.int8)
    for L in LEVELS:
        img[fmap == L] = level_to_idx[L]

    cmap = plt.cm.viridis
    bounds = np.arange(len(LEVELS) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(LEVELS)))
    cbar.set_ticklabels([f"L={L} ({L}×{L})" for L in LEVELS])
    cbar.set_label("Coarsest level satisfying ≥{} samples".format(min_samples), fontsize=11)
    ax.set_title(
        f"Per-pixel fallback level for GT bin {gt_bin} "
        f"(p ≈ {(gt_bin+0.5)/N_BINS:.3f})\n"
        f"Bright = needs coarser fallback (per-pixel data sparse here)",
        fontsize=12,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    out_pdf = FIG_DIR / "pyramid_fallback_map_p03.pdf"
    out_png = FIG_DIR / "pyramid_fallback_map_p03.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_pdf}")
    return fmap


def fig_usage_bar():
    """Bar chart of fraction of queries resolved at each pyramid level."""
    if not Path(USAGE_STATS_PATH).exists():
        print(f"  [WARN] {USAGE_STATS_PATH} missing — run pyramid_sampler.py first")
        return
    d = np.load(USAGE_STATS_PATH)
    levels = list(d["levels"])
    fracs = list(d["fractions"]) * 100  # to %
    fracs = list(np.array(d["fractions"]) * 100)
    n_q = int(d["n_queries"])
    min_s = int(d["min_samples"])
    gt_dist = str(d["gt_distribution"])

    fig, ax = plt.subplots(figsize=(11, 6))
    x_pos = np.arange(len(levels))
    bars = ax.bar(x_pos, fracs, color="steelblue", edgecolor="black")
    for bar, frac in zip(bars, fracs):
        if frac >= 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.5,
                    f"{frac:.1f}%", ha="center", fontsize=10)
        elif frac > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.5,
                    f"{frac:.2f}%", ha="center", fontsize=9, color="darkblue")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L={L}\n({L}×{L})" for L in levels], fontsize=9)
    ax.set_xlabel("Pyramid level (pool size)", fontsize=12)
    ax.set_ylabel("% of queries resolved at this level", fontsize=12)
    ax.set_title(
        f"Pyramid fallback usage  (1M queries from GT~{gt_dist}, min_samples={min_s})\n"
        f"Most queries resolved at finest level where data is dense; "
        f"sparse pixels fall back gracefully",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(fracs) * 1.15)
    fig.tight_layout()

    out_pdf = FIG_DIR / "pyramid_usage_bar.pdf"
    out_png = FIG_DIR / "pyramid_usage_bar.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_pdf}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_fallback_map(min_samples=20, gt_bin=77)
    fig_usage_bar()


if __name__ == "__main__":
    main()
