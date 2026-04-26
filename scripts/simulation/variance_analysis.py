#!/usr/bin/env python3
"""
Variance decomposition + representative pixel analysis
======================================================
Version: v1

Uses the per-pixel per-bin sufficient statistics (sum, sum_sq, count) saved
by build_three_luts.py to decompose:

    Var_total(b) = Var_between(b) + Var_within(b)

where b is the GT intensity bin (0–255).

Also generates the supervisor's per-pixel variance figure (3×3 grid).

Outputs:
    calibration/figures/variance_decomposition.pdf
    calibration/figures/per_pixel_variance_vs_gt.pdf
    calibration/figures/per_pixel_coverage_N{1,5,10,20,50}.pdf
    calibration/variance_decomposition.npz
"""

import sys
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Provenance helper
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from provenance import load as prov_load, summarize as prov_summarize  # noqa: E402

H, W = 512, 512
N_BINS = 256
K_SHORT = 64
VERSION = "v1"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cal-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--figures-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/figures")
    args = p.parse_args()

    cal = Path(args.cal_dir)
    fig_dir = Path(args.figures_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- Load variance stats with provenance ----------
    try:
        vs, vs_prov = prov_load(cal / "variance_stats.npz")
    except ValueError as e:
        sys.exit(f"\n[FATAL] variance_stats.npz lacks provenance metadata.\n  {e}\n"
                 f"  This file was likely built before the provenance fix. "
                 f"Rebuild with build_three_luts.py (post-audit2 version).")
    print(prov_summarize(vs_prov, "variance_stats.npz"))

    var_sum = vs["var_sum"]
    var_sum2 = vs["var_sum2"]
    var_count = vs["var_count"]
    k_short = int(vs_prov["k_short"])
    n_scenes_var = int(vs_prov["n_scenes_used"])

    # ---------- Load coverage counts ----------
    cc = np.load(cal / "coverage_counts.npz")
    counts_pp = cc["counts_per_pixel"]
    n_scenes_cc = int(cc["n_scenes"]) if "n_scenes" in cc else -1
    print(f"\n--- coverage_counts.npz ---")
    print(f"  n_scenes              {n_scenes_cc}")
    print(f"  counts_per_pixel.shape  {counts_pp.shape}")

    # ---------- Soft compatibility check ----------
    print()
    if n_scenes_cc < 0:
        print("[WARN] coverage_counts.npz has no scene-count metadata; cannot verify.")
    else:
        scene_diff = abs(n_scenes_var - n_scenes_cc)
        scene_pct = 100 * scene_diff / max(n_scenes_var, n_scenes_cc, 1)
        if scene_pct > 5:
            sys.exit(f"\n[FATAL] Scene-count mismatch >5%: variance_stats has "
                     f"{n_scenes_var} scenes, coverage_counts has {n_scenes_cc} "
                     f"({scene_pct:.1f}% diff). These artifacts are NOT compatible.\n"
                     f"  Both should be (re)built from the same scene universe.")
        elif scene_pct > 1:
            print(f"[WARN] Small scene-count mismatch: var={n_scenes_var}, cov={n_scenes_cc} "
                  f"({scene_pct:.1f}% diff). Proceeding but flagged.")
        else:
            print(f"[OK] Scene counts match: var={n_scenes_var}, cov={n_scenes_cc}")

    if var_sum.shape[:2] != counts_pp.shape[:2]:
        sys.exit(f"[FATAL] Spatial shape mismatch: var_sum {var_sum.shape}, "
                 f"counts_pp {counts_pp.shape}")
    if var_sum.shape[2] != counts_pp.shape[2]:
        sys.exit(f"[FATAL] Bin count mismatch: var_sum {var_sum.shape[2]}, "
                 f"counts_pp {counts_pp.shape[2]}")
    print()

    # =========================================================================
    # 1. Coverage maps (per-pixel spatial visualization)
    # =========================================================================
    print("\n[Coverage] Generating spatial coverage maps …")
    thresholds = [1, 5, 10, 20, 50]
    for N in thresholds:
        bins_above = (counts_pp >= N).sum(axis=2)  # (H, W): bins meeting threshold per pixel
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(bins_above, cmap="viridis", vmin=0, vmax=256)
        plt.colorbar(im, ax=ax, label="Bins with ≥N samples")
        ax.set_title(f"Per-pixel bin coverage (N≥{N})  [{VERSION}]\n"
                     f"bright = good coverage; dark = sparse", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        out = fig_dir / f"per_pixel_coverage_N{N}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  N≥{N}: {out}")

    # =========================================================================
    # 2. Variance decomposition
    # =========================================================================
    print("\n[Variance] Computing decomposition …")

    # Per-pixel per-bin mean: μ(x,y,b) = sum / count (where count > 0)
    mask = var_count >= 2  # need ≥2 samples for variance
    px_mean = np.where(mask, var_sum / np.maximum(var_count, 1), np.nan)
    px_var = np.where(mask,
                      (var_sum2 / np.maximum(var_count, 1)) - px_mean**2,
                      np.nan)
    # Clip tiny negative variances from floating-point error
    px_var = np.maximum(px_var, 0.0)

    var_total = np.zeros(N_BINS, dtype=np.float64)
    var_between = np.zeros(N_BINS, dtype=np.float64)
    var_within = np.zeros(N_BINS, dtype=np.float64)
    n_pixels_per_bin = np.zeros(N_BINS, dtype=np.int64)

    for b in range(N_BINS):
        pixel_mask = mask[:, :, b]
        n_px = pixel_mask.sum()
        if n_px < 10:
            var_total[b] = var_between[b] = var_within[b] = np.nan
            n_pixels_per_bin[b] = n_px
            continue

        # Gather all p̂ stats for this bin across all pixels
        means = px_mean[:, :, b][pixel_mask]   # per-pixel means for bin b
        variances = px_var[:, :, b][pixel_mask]  # per-pixel variances for bin b
        counts = var_count[:, :, b][pixel_mask].astype(np.float64)

        # Total variance: Var of all samples pooled
        # Using law of total variance:
        #   Var_total = E[Var_within] + Var[E_within]
        #   Var_total = mean(px_var) + Var(px_mean)
        # Weighted by count per pixel:
        total_count = counts.sum()
        weighted_mean = (means * counts).sum() / total_count
        var_between[b] = ((means - weighted_mean)**2 * counts).sum() / total_count
        var_within[b] = (variances * counts).sum() / total_count
        var_total[b] = var_between[b] + var_within[b]
        n_pixels_per_bin[b] = n_px

    # Binomial prediction: Var(p̂) = p(1-p)/K for i.i.d. Bernoulli
    p_centers = (np.arange(N_BINS) + 0.5) / N_BINS
    var_binomial = p_centers * (1 - p_centers) / k_short

    # Ratio
    ratio = np.where(var_total > 0, var_between / var_total, np.nan)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    valid = ~np.isnan(var_total)
    ax.plot(p_centers[valid], var_total[valid], "k-", lw=2, label="Var_total", zorder=3)
    ax.plot(p_centers[valid], var_between[valid], "r-", lw=2, label="Var_between (fixed-pattern)")
    ax.plot(p_centers[valid], var_within[valid], "b-", lw=2, label="Var_within (shot + afterpulsing)")
    ax.plot(p_centers, var_binomial, "g--", lw=1.5, alpha=0.7,
            label=f"Binomial p(1-p)/K  (K={k_short})")
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title(f"Variance decomposition vs. GT intensity bin  [{VERSION}]", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(p_centers[valid], ratio[valid] * 100, "r-", lw=2)
    ax.axhline(10, color="gray", ls="--", alpha=0.5, label="10% threshold")
    ax.axhline(30, color="gray", ls=":", alpha=0.5, label="30% threshold")
    ax.set_xlabel("GT intensity (p_GT)", fontsize=12)
    ax.set_ylabel("Var_between / Var_total  (%)", fontsize=12)
    ax.set_title("Fixed-pattern noise fraction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out_var = fig_dir / "variance_decomposition.pdf"
    fig.savefig(out_var, bbox_inches="tight")
    out_png = fig_dir / "variance_decomposition.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_var}")

    # Save numbers
    out_npz = cal / "variance_decomposition.npz"
    np.savez(out_npz,
             var_total=var_total, var_between=var_between, var_within=var_within,
             var_binomial=var_binomial, ratio_between_total=ratio,
             p_centers=p_centers, n_pixels_per_bin=n_pixels_per_bin,
             k_short=k_short, version=VERSION)
    print(f"  Data → {out_npz}")

    # Print summary
    valid_ratio = ratio[valid & (n_pixels_per_bin >= 100)]
    print(f"\n  Var_between / Var_total (for bins with ≥100 pixels):")
    print(f"    mean  = {np.nanmean(valid_ratio)*100:.1f}%")
    print(f"    median = {np.nanmedian(valid_ratio)*100:.1f}%")
    print(f"    min   = {np.nanmin(valid_ratio)*100:.1f}%")
    print(f"    max   = {np.nanmax(valid_ratio)*100:.1f}%")

    if np.nanmean(valid_ratio) < 0.10:
        print("  → Fixed-pattern noise is SMALL (<10%); global LUT may suffice")
    elif np.nanmean(valid_ratio) < 0.30:
        print("  → Fixed-pattern noise is MODERATE; super-pixel captures most of it")
    else:
        print("  → Fixed-pattern noise is LARGE (>30%); per-pixel/super-pixel is needed")

    # =========================================================================
    # 3. Representative pixel variance figure (supervisor's question)
    # =========================================================================
    print("\n[Pixels] Generating per-pixel variance figure …")

    # Compute per-pixel mean rate across all bins (as a summary statistic)
    total_per_pixel_count = var_count.sum(axis=2).astype(np.float64)
    total_per_pixel_sum = var_sum.sum(axis=2)
    mean_rate_per_pixel = np.where(total_per_pixel_count > 0,
                                    total_per_pixel_sum / total_per_pixel_count, 0)

    # Low-GT mean p̂ (dark count proxy): average p̂ in bins 0–10
    low_gt_mask = var_count[:, :, :10].sum(axis=2) > 0
    low_gt_mean = np.where(low_gt_mask,
                           var_sum[:, :, :10].sum(axis=2) /
                           np.maximum(var_count[:, :, :10].sum(axis=2), 1), 0)

    # High-GT mean p̂ (PDE proxy): average p̂ in bins 200–255
    high_gt_mask = var_count[:, :, 200:].sum(axis=2) > 0
    high_gt_mean = np.where(high_gt_mask,
                            var_sum[:, :, 200:].sum(axis=2) /
                            np.maximum(var_count[:, :, 200:].sum(axis=2), 1), 0)

    rng = np.random.RandomState(42)

    def pick_pixels(arr, quantile_lo, quantile_hi, valid_mask, n=3):
        """Pick n pixels from the quantile range of arr where valid_mask is True."""
        vals = arr[valid_mask]
        lo = np.percentile(vals, quantile_lo)
        hi = np.percentile(vals, quantile_hi)
        candidates = np.argwhere(valid_mask & (arr >= lo) & (arr <= hi))
        if len(candidates) < n:
            return candidates
        idx = rng.choice(len(candidates), n, replace=False)
        return candidates[idx]

    # Categories
    categories = {
        "Typical (median DCR)": pick_pixels(low_gt_mean, 45, 55, low_gt_mask, 3),
        "High DCR (top 1%)": pick_pixels(low_gt_mean, 99, 100, low_gt_mask, 3),
        "Low PDE (bottom 1%)": pick_pixels(high_gt_mean, 0, 1, high_gt_mask, 3),
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row, (cat_name, coords) in enumerate(categories.items()):
        for col in range(min(3, len(coords))):
            ax = axes[row, col]
            y, x = coords[col]

            # Per-bin variance for this pixel
            c = var_count[y, x, :]
            valid_bins = c >= 2
            bins_valid = np.where(valid_bins)[0]
            if len(bins_valid) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            means_b = np.where(valid_bins, var_sum[y, x, :] / np.maximum(c, 1), 0)
            vars_b = np.where(valid_bins,
                              var_sum2[y, x, :] / np.maximum(c, 1) - means_b**2, 0)
            vars_b = np.maximum(vars_b, 0)

            ax.scatter(p_centers[valid_bins], vars_b[valid_bins],
                       s=8, alpha=0.6, color="steelblue", label="Observed Var(p̂|GT)")
            ax.plot(p_centers, var_binomial, "g--", lw=1.5, alpha=0.7,
                    label=f"Binomial p(1-p)/{k_short}")

            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            ax.set_xlabel("GT bin center" if row == 2 else "")
            ax.set_ylabel("Var(p̂)" if col == 0 else "")
            ax.set_title(f"{cat_name}\npx ({x},{y}), mean_rate={mean_rate_per_pixel[y,x]:.4f}",
                         fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-pixel variance vs. GT bin  [{VERSION}]\n"
                 f"Blue = observed variance;  Green dashed = Binomial floor (K={k_short})\n"
                 f"Excess above green = non-ideality (afterpulsing, FPN, etc.)",
                 fontsize=12)
    fig.tight_layout()
    out_pv = fig_dir / "per_pixel_variance_vs_gt.pdf"
    fig.savefig(out_pv, bbox_inches="tight")
    out_pv_png = fig_dir / "per_pixel_variance_vs_gt.png"
    fig.savefig(out_pv_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_pv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
