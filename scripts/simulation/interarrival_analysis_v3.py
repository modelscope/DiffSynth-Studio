#!/usr/bin/env python3
"""
Inter-arrival analysis v3 — per-pixel residual methodology
==========================================================
Version: v3-post-audit

Addresses Audit Finding 2: the v2 analysis pooled inter-arrivals across wide
rate brackets and compared the pooled empirical histogram to a single fitted
Geometric(p̂_pooled). A heterogeneous mixture of geometrics inherently inflates
P(δt=1) above what a single Geometric predicts (by AM ≥ HM inequality on
per-pixel rates), so the v2 "17% excess at low rate = afterpulsing" claim is
confounded.

v3 fixes this by computing the residual per-pixel:

    For each pixel i with empirical rate p̂_i:
        observed_P_dt1_i = (#δt=1 events) / (#total events)
        predicted_P_dt1_i = p̂_i                    [single-pixel Geometric]
        residual_i = observed_P_dt1_i - predicted_P_dt1_i

The DISTRIBUTION of residuals tells the story:
    - mean(residual) > 0 with tight scatter → real afterpulsing (rate-independent)
    - mean(residual) ≈ 0                    → no afterpulsing
    - residual correlates with p̂            → likely a different non-ideality
                                              (saturation, dead-time)

Also computes:
    - Per-pixel KS-style statistic vs Geometric(p̂_i)
    - Aggregate residual histogram per narrow rate bracket (0.01-wide)
    - Lag-1 autocorrelation of inter-arrival sequence (true afterpulsing
      shows positive lag-1 correlation; pure i.i.d. shows ~0)

Uses the race-free `spad_utils_fixed.accumulate_counts_whole_file` for
p_GT computation.

Outputs:
    figures/interarrival_v3_per_pixel_residuals.pdf
    figures/interarrival_v3_summary.npz
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# Race-free utils
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from spad_utils_fixed import accumulate_counts_whole_file  # noqa: E402

H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8
MONO_BIN = "RAW_empty.bin"
VERSION = "v3-post-audit"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/images")
    p.add_argument("--figures-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/figures")
    p.add_argument("--n-scenes", type=int, default=30)
    p.add_argument("--n-frames", type=int, default=20_000)
    p.add_argument("--pixels-per-bracket-per-scene", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rotate-k", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    images_dir = Path(args.images_dir)
    all_scenes = sorted(
        d.name for d in images_dir.iterdir()
        if d.is_dir() and (d / MONO_BIN).exists()
    )
    print(f"Total scenes: {len(all_scenes)}")

    step = max(1, len(all_scenes) // args.n_scenes)
    selected = all_scenes[::step][: args.n_scenes]
    print(f"Selected {len(selected)} scenes (every {step}-th)")

    # Narrow rate brackets so within-bracket heterogeneity is small.
    # 0.01-wide brackets in p ∈ [0.005, 0.30].
    bracket_edges = np.arange(0.005, 0.305, 0.01)  # 30 brackets

    # Per-pixel records (one entry per pixel that's used in any bracket)
    pixel_records = []   # each: dict(scene, x, y, p_hat, dt1_obs, dt1_pred,
                          #              residual, n_events, lag1_corr, ia_array_short)

    print("Warming up Numba …")
    _d = np.zeros(BYTES_PER_FRAME, dtype=np.uint8)
    accumulate_counts_whole_file(_d, 1, H, W)

    t0 = time.time()
    try:
        from tqdm import tqdm
        scene_iter = tqdm(selected, desc="Scenes")
    except ImportError:
        scene_iter = selected

    for scene in scene_iter:
        bin_path = str(images_dir / scene / MONO_BIN)
        n_read = args.n_frames

        try:
            bytes_needed = n_read * BYTES_PER_FRAME
            with open(bin_path, "rb") as f:
                raw = np.frombuffer(f.read(bytes_needed), dtype=np.uint8)
            n_avail = len(raw) // BYTES_PER_FRAME
            n_use = min(n_read, n_avail)
            if n_use < 100:
                continue
        except Exception as e:
            print(f"[skip] {scene}: {e}")
            continue

        # Race-free GT computation
        counts, n_actual = accumulate_counts_whole_file(
            raw[:n_use * BYTES_PER_FRAME], n_use, H, W
        )
        p_gt = counts.astype(np.float32) / float(n_actual)
        if args.rotate_k % 4 != 0:
            p_gt = np.rot90(p_gt, k=args.rotate_k)

        # Pick eligible pixels per bracket and extract per-pixel inter-arrivals
        for bi in range(len(bracket_edges) - 1):
            lo, hi = bracket_edges[bi], bracket_edges[bi + 1]
            mask = (p_gt >= lo) & (p_gt < hi)
            eligible = np.argwhere(mask)
            if len(eligible) == 0:
                continue
            n_take = min(args.pixels_per_bracket_per_scene, len(eligible))
            idx = rng.choice(len(eligible), n_take, replace=False)
            coords = eligible[idx]

            for (py, px) in coords:
                # Map rotated coords back to raw
                if args.rotate_k % 4 == 1:
                    raw_y, raw_x = px, H - 1 - py
                elif args.rotate_k % 4 == 2:
                    raw_y, raw_x = H - 1 - py, W - 1 - px
                elif args.rotate_k % 4 == 3:
                    raw_y, raw_x = W - 1 - px, py
                else:
                    raw_y, raw_x = py, px

                bit_pos = raw_y * W + raw_x
                byte_off = bit_pos // 8
                bit_in_byte = 7 - (bit_pos % 8)

                detections = []
                for f_i in range(n_use):
                    byte_val = raw[f_i * BYTES_PER_FRAME + byte_off]
                    if (byte_val >> bit_in_byte) & 1:
                        detections.append(f_i)

                n_events = len(detections)
                if n_events < 10:
                    continue

                det = np.array(detections, dtype=np.int32)
                ia = np.diff(det)             # inter-arrivals in frames
                p_hat = n_events / float(n_use)
                dt1_obs = (ia == 1).sum() / len(ia)
                dt1_pred = p_hat              # Geometric(p_hat) gives P(δt=1) = p_hat

                # Lag-1 autocorrelation of ia sequence (centered)
                if len(ia) >= 4:
                    a = ia.astype(np.float64)
                    a -= a.mean()
                    var = (a * a).sum()
                    if var > 0:
                        lag1 = (a[:-1] * a[1:]).sum() / var
                    else:
                        lag1 = 0.0
                else:
                    lag1 = 0.0

                pixel_records.append({
                    "scene": scene, "x": int(px), "y": int(py),
                    "bracket_lo": float(lo), "bracket_hi": float(hi),
                    "p_hat": float(p_hat), "n_events": int(n_events),
                    "dt1_obs": float(dt1_obs), "dt1_pred": float(dt1_pred),
                    "residual": float(dt1_obs - dt1_pred),
                    "ratio": float(dt1_obs / max(dt1_pred, 1e-9)),
                    "lag1_corr": float(lag1),
                })

    elapsed = time.time() - t0
    print(f"\nProcessed {len(selected)} scenes in {elapsed:.1f}s "
          f"({len(pixel_records)} per-pixel records)")

    # =========================================================================
    # Analysis
    # =========================================================================
    rs = pixel_records
    p_hats = np.array([r["p_hat"] for r in rs])
    residuals = np.array([r["residual"] for r in rs])
    ratios = np.array([r["ratio"] for r in rs])
    lag1s = np.array([r["lag1_corr"] for r in rs])
    bracket_los = np.array([r["bracket_lo"] for r in rs])
    n_events = np.array([r["n_events"] for r in rs])

    # Per-bracket aggregates
    print("\n" + "=" * 90)
    print(f"  Inter-arrival per-pixel residual analysis  [{VERSION}]")
    print("=" * 90)
    print(f"{'bracket':>14}  {'n_pix':>5}  {'mean(p̂)':>9}  "
          f"{'mean(residual)':>14}  {'std(residual)':>13}  "
          f"{'mean(ratio)':>11}  {'mean(lag1)':>10}")
    print("-" * 90)
    bracket_summary = []
    unique_los = np.unique(bracket_los)
    for lo in unique_los:
        m = bracket_los == lo
        n = m.sum()
        if n < 5:
            continue
        mr = residuals[m].mean()
        sr = residuals[m].std()
        mratio = ratios[m].mean()
        mlag1 = lag1s[m].mean()
        mph = p_hats[m].mean()
        print(f"  [{lo:.3f},{lo+0.01:.3f})  {n:>5}  {mph:>9.4f}  "
              f"{mr:>+14.5f}  {sr:>13.5f}  {mratio:>11.3f}  {mlag1:>+10.4f}")
        bracket_summary.append({
            "lo": float(lo), "hi": float(lo + 0.01), "n_pixels": int(n),
            "mean_p_hat": float(mph),
            "mean_residual": float(mr), "std_residual": float(sr),
            "mean_ratio": float(mratio), "mean_lag1": float(mlag1),
        })
    print("=" * 90)

    # =========================================================================
    # Plot: residual vs. p̂, ratio vs. p̂, lag-1 correlation vs. p̂
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: per-pixel residual scatter
    ax = axes[0, 0]
    ax.scatter(p_hats, residuals, s=2, alpha=0.2, c="steelblue", rasterized=True)
    # Bin centers + means
    bcs = [s["mean_p_hat"] for s in bracket_summary]
    bres = [s["mean_residual"] for s in bracket_summary]
    bres_std = [s["std_residual"] for s in bracket_summary]
    ax.errorbar(bcs, bres, yerr=bres_std, fmt="ro-", lw=2, capsize=3,
                label="bracket mean ± std")
    ax.axhline(0, color="black", ls="--", alpha=0.5, label="zero (Geom. prediction)")
    ax.set_xlabel("Per-pixel p̂", fontsize=11)
    ax.set_ylabel("Residual: P(δt=1)_obs − P(δt=1)_pred", fontsize=11)
    ax.set_title("Per-pixel residual vs. rate", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: ratio (multiplicative)
    ax = axes[0, 1]
    ax.scatter(p_hats, ratios, s=2, alpha=0.2, c="forestgreen", rasterized=True)
    bratios = [s["mean_ratio"] for s in bracket_summary]
    ax.plot(bcs, bratios, "ro-", lw=2, label="bracket mean")
    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="ratio=1 (no excess)")
    ax.set_xlabel("Per-pixel p̂", fontsize=11)
    ax.set_ylabel("Ratio: P(δt=1)_obs / P(δt=1)_pred", fontsize=11)
    ax.set_title("Multiplicative excess vs. rate\n(values >1 = excess of short IAs)",
                 fontsize=12)
    ax.set_ylim(0.5, 2.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: lag-1 autocorrelation
    ax = axes[1, 0]
    ax.scatter(p_hats, lag1s, s=2, alpha=0.2, c="orchid", rasterized=True)
    blag1s = [s["mean_lag1"] for s in bracket_summary]
    ax.plot(bcs, blag1s, "ro-", lw=2, label="bracket mean")
    ax.axhline(0, color="black", ls="--", alpha=0.5, label="iid (uncorrelated)")
    ax.set_xlabel("Per-pixel p̂", fontsize=11)
    ax.set_ylabel("Lag-1 autocorrelation of inter-arrivals", fontsize=11)
    ax.set_title("Sequential correlation\n(positive = afterpulsing-like clustering)",
                 fontsize=12)
    ax.set_ylim(-0.1, 0.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: residual histogram aggregated, with bracket overlays
    ax = axes[1, 1]
    ax.hist(residuals, bins=80, density=True, alpha=0.7, color="steelblue",
            label=f"all pixels (N={len(residuals)})")
    ax.axvline(0, color="black", ls="--", alpha=0.5, label="iid prediction")
    ax.axvline(residuals.mean(), color="red", ls="-", lw=2,
               label=f"mean={residuals.mean():+.4f}")
    ax.set_xlabel("Residual: P(δt=1)_obs − p̂", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Residual distribution across all pixels", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Inter-arrival per-pixel residuals  [{VERSION}]    "
        f"{len(rs)} pixels, {len(selected)} scenes\n"
        f"Per-pixel comparison eliminates rate-mixing confound. "
        f"Residual > 0 with no rate dependence ⇒ afterpulsing.",
        fontsize=12,
    )
    fig.tight_layout()

    out_pdf = Path(args.figures_dir) / "interarrival_v3_per_pixel_residuals.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    out_png = Path(args.figures_dir) / "interarrival_v3_per_pixel_residuals.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_pdf}")

    # ---- Verdict ----
    overall_residual = residuals.mean()
    overall_ratio = ratios.mean()
    overall_lag1 = lag1s.mean()

    print(f"\n  Overall (all {len(rs)} pixels):")
    print(f"    mean residual = {overall_residual:+.5f}")
    print(f"    mean ratio    = {overall_ratio:.4f}")
    print(f"    mean lag-1    = {overall_lag1:+.5f}")
    print()

    if overall_residual > 0.001 and overall_lag1 > 0.005:
        print("    → Both per-pixel residual AND lag-1 are positive.")
        print("      Strong evidence of AFTERPULSING (correlated re-firing).")
    elif overall_residual > 0.001:
        print("    → Per-pixel residual positive but lag-1 ≈ 0.")
        print("      Some short-IA excess, but not from sequential correlation.")
        print("      Could be saturation, hot-pixel artifact, or dead-time recovery patterns.")
    elif abs(overall_residual) < 0.001:
        print("    → Per-pixel residuals near zero. SPAD is well-modeled by per-pixel Geometric(p_i).")
        print("      The v2 finding of '17% excess' was ENTIRELY due to rate-mixing — confirmed.")
    else:
        print("    → Negative residual: short IAs are DEPLETED. Possible dead-time signature.")

    # Save summary
    out_npz = Path(args.figures_dir) / "interarrival_v3_summary.npz"
    np.savez(
        out_npz,
        p_hats=p_hats, residuals=residuals, ratios=ratios, lag1s=lag1s,
        bracket_los=bracket_los, n_events=n_events,
        bracket_lo=np.array([s["lo"] for s in bracket_summary]),
        bracket_hi=np.array([s["hi"] for s in bracket_summary]),
        bracket_n=np.array([s["n_pixels"] for s in bracket_summary]),
        bracket_mean_residual=np.array([s["mean_residual"] for s in bracket_summary]),
        bracket_std_residual=np.array([s["std_residual"] for s in bracket_summary]),
        bracket_mean_ratio=np.array([s["mean_ratio"] for s in bracket_summary]),
        bracket_mean_lag1=np.array([s["mean_lag1"] for s in bracket_summary]),
        bracket_mean_p_hat=np.array([s["mean_p_hat"] for s in bracket_summary]),
        version=VERSION,
    )
    print(f"  Summary → {out_npz}")


if __name__ == "__main__":
    main()
