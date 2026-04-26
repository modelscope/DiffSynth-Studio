#!/usr/bin/env python3
"""
RGB-to-Mono Calibration in λ-space  —  v3
======================================================
Version: v3

Fits a linear map from per-channel photon-flux proxies (λ = -ln(1-p)) to the
monochrome flux proxy, which is the space in which the RGB→mono relationship
is physically additive (integrals of spectral density × filter transmittance).

    λ_mono(x,y)  ≈  w_r·λ_r + w_g·λ_g + w_b·λ_b

Changes vs. v1:
  1. Fit in λ-space (not p-space). λ = -ln(1 - p), clipped away from 1.
     → True additive-across-spectrum model; removes the Bernoulli
       saturation nonlinearity that v1 absorbed into its weights.
  2. Apply median_hotpixel_fix on the RAW COUNTS before normalizing.
     → Sensor-defect pixels that fire regardless of incident photons
       violate the spectral model; this replaces them with their
       neighborhood median (same pixel in all 4 channels).
  3. Pixel mask applied on the post-fix Bernoulli rate, identical
     thresholds to v1 (0.01 < p < 0.95) for comparability.

Outputs (alongside v1 artifacts):
  calibration/rgb_to_mono_weights_v3.npz
  calibration/rgb_to_mono_diagnostic_v3.png
  calibration/calibration_log_v3.txt
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# NOTE (2026-04-16): switched to race-free spad_utils_fixed after audit.
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from spad_utils_fixed import accumulate_counts_whole_file, median_hotpixel_fix  # noqa: E402

# ---------------------------------------------------------------------------
H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8

CHANNEL_FILES = {
    "r": "RAW_col_r.bin",
    "g": "RAW_col_g.bin",
    "b": "RAW_col_b.bin",
    "mono": "RAW_empty.bin",
}

RATE_LO = 0.01
RATE_HI = 0.95
LOG_EPS = 1e-6  # keep λ finite when p → 1
VERSION = "v3"


def load_bernoulli_rate_with_hotpix_fix(bin_path, max_frames, rotate_k):
    """Read binary, accumulate counts, median-fix hot pixels on counts,
    then normalize to Bernoulli rate."""
    bytes_needed = max_frames * BYTES_PER_FRAME
    with open(bin_path, "rb") as f:
        raw = np.frombuffer(f.read(bytes_needed), dtype=np.uint8)
    counts, n_frames = accumulate_counts_whole_file(raw, max_frames, H, W)

    # Median hot-pixel fix on counts.  spike threshold defaults:
    # max(10, int(0.01 * N))  — replaces pixels whose count exceeds its 3×3
    # median neighbor by more than ~1% of N_frames.
    counts_fixed = median_hotpixel_fix(counts, n_frames, ksize=3,
                                       spike_thresh_abs=10, spike_thresh_rel=0.01)

    rate = counts_fixed.astype(np.float32) / float(n_frames)
    if rotate_k % 4 != 0:
        rate = np.rot90(rate, k=rotate_k)
    return rate


def rate_to_lambda(rate):
    """λ = -log(1 - p), numerically stable for p near 1."""
    p_safe = np.clip(rate, 0.0, 1.0 - LOG_EPS)
    return -np.log1p(-p_safe)  # log1p(-p) = log(1-p) with better precision


def compute_r2(XtX, Xty, yty, sumy, N, w):
    ss_res = yty - 2.0 * w @ Xty + w @ XtX @ w
    mean_y = sumy / N
    ss_tot = yty - N * mean_y ** 2
    return 1.0 - ss_res / ss_tot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=str,
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/images")
    p.add_argument("--output-dir", type=str,
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scatter-points", type=int, default=50_000)
    p.add_argument("--rotate-k", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = Path(args.images_dir)

    print("=" * 64)
    print(f"  RGB → Mono Calibration in λ-space  [{VERSION}]")
    print("=" * 64)
    print(f"  Hot pixel fix:     ON (median, ksize=3, spike>max(10, 0.01·N))")
    print(f"  Space:             λ = -ln(1 - p)")
    print(f"  Pixel filter:      {RATE_LO} < p < {RATE_HI} (applied to post-fix rate)")
    print()

    # Enumerate scenes (same as v1, for deterministic matching)
    all_scenes = sorted(
        d.name for d in images_dir.iterdir()
        if d.is_dir()
        and all((d / fn).exists() for fn in CHANNEL_FILES.values())
    )
    if args.max_scenes is not None:
        all_scenes = all_scenes[:args.max_scenes]
    n_total = len(all_scenes)
    print(f"Scenes available: {n_total}")
    if n_total == 0:
        sys.exit("ERROR: no scenes found")

    # Train/val split — same seed and permutation as v1 for apples-to-apples
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_fraction))
    val_set = set(perm[:n_val].tolist())
    n_train = n_total - n_val
    print(f"Train scenes: {n_train}   Val scenes: {n_val}")
    print(f"Max frames per file: {args.max_frames}")
    print()

    # Running sufficient statistics (separate for train/val)
    stats = {
        s: dict(
            XtX=np.zeros((3, 3), dtype=np.float64),
            Xty=np.zeros(3, dtype=np.float64),
            yty=np.float64(0.0),
            sumy=np.float64(0.0),
            N=0,
        ) for s in ("train", "val")
    }

    scatter_buf = []
    samples_per_scene = max(10, (args.scatter_points * 3) // n_total)

    skipped = 0
    t0 = time.time()

    print("Warming up Numba JIT …")
    _dummy = np.zeros(BYTES_PER_FRAME, dtype=np.uint8)
    accumulate_counts_whole_file(_dummy, 1, H, W)
    print("JIT ready.\n")

    try:
        from tqdm import tqdm
        scene_iter = tqdm(enumerate(all_scenes), total=n_total, desc="Scenes")
    except ImportError:
        scene_iter = enumerate(all_scenes)

    for idx, scene in scene_iter:
        split = "val" if idx in val_set else "train"
        scene_dir = images_dir / scene

        try:
            rates = {}
            for ch, fn in CHANNEL_FILES.items():
                rates[ch] = load_bernoulli_rate_with_hotpix_fix(
                    str(scene_dir / fn), args.max_frames, args.rotate_k
                )
        except Exception as e:
            print(f"[skip] {scene}: {e}")
            skipped += 1
            continue

        # Mask pixels where any channel rate is outside [RATE_LO, RATE_HI]
        # (evaluated on POST-hotpix-fix rates — so hot pixels are replaced
        # with their median neighbors and then tested, which is the right
        # thing to do).
        mask = np.ones((H, W), dtype=bool)
        for ch in ("r", "g", "b", "mono"):
            mask &= (rates[ch] > RATE_LO) & (rates[ch] < RATE_HI)
        n_valid = int(mask.sum())
        if n_valid < 10:
            skipped += 1
            continue

        # Transform to λ-space
        lam_r = rate_to_lambda(rates["r"][mask]).astype(np.float64)
        lam_g = rate_to_lambda(rates["g"][mask]).astype(np.float64)
        lam_b = rate_to_lambda(rates["b"][mask]).astype(np.float64)
        lam_mono = rate_to_lambda(rates["mono"][mask]).astype(np.float64)

        X_px = np.column_stack([lam_r, lam_g, lam_b])
        y_px = lam_mono

        s = stats[split]
        s["XtX"] += X_px.T @ X_px
        s["Xty"] += X_px.T @ y_px
        s["yty"] += y_px @ y_px
        s["sumy"] += y_px.sum()
        s["N"] += n_valid

        n_take = min(samples_per_scene, n_valid)
        pick = rng.choice(n_valid, n_take, replace=False)
        scatter_buf.append(np.column_stack([X_px[pick], y_px[pick]]))

    elapsed = time.time() - t0
    print(f"\nProcessed {n_total - skipped}/{n_total} scenes in {elapsed:.1f}s")
    print(f"Skipped {skipped} scenes")
    print(f"Train pixels: {stats['train']['N']:,}")
    print(f"Val   pixels: {stats['val']['N']:,}")

    if stats["train"]["N"] == 0:
        sys.exit("ERROR: no valid training pixels")

    # Solve
    w = np.linalg.solve(stats["train"]["XtX"], stats["train"]["Xty"])
    r2_train = compute_r2(stats["train"]["XtX"], stats["train"]["Xty"],
                          stats["train"]["yty"], stats["train"]["sumy"],
                          stats["train"]["N"], w)
    r2_val = compute_r2(stats["val"]["XtX"], stats["val"]["Xty"],
                        stats["val"]["yty"], stats["val"]["sumy"],
                        stats["val"]["N"], w) if stats["val"]["N"] > 0 else float("nan")

    print("\n" + "=" * 64)
    print(f"  RESULTS  [{VERSION}]    (weights are in λ-space)")
    print("=" * 64)
    print(f"  w_r  = {w[0]:.6f}")
    print(f"  w_g  = {w[1]:.6f}")
    print(f"  w_b  = {w[2]:.6f}")
    print(f"  sum  = {w.sum():.6f}")
    print()
    print(f"  Train R² = {r2_train:.6f}   ({stats['train']['N']:>12,} pixels)")
    print(f"  Val   R² = {r2_val:.6f}   ({stats['val']['N']:>12,} pixels)")
    print("=" * 64)

    weights_path = Path(args.output_dir) / "rgb_to_mono_weights_v3.npz"
    np.savez(
        weights_path,
        weights=w, w_r=w[0], w_g=w[1], w_b=w[2],
        train_r2=r2_train, val_r2=r2_val,
        train_N=stats["train"]["N"], val_N=stats["val"]["N"],
        max_frames=args.max_frames,
        rate_lo=RATE_LO, rate_hi=RATE_HI, log_eps=LOG_EPS,
        n_train_scenes=n_train, n_val_scenes=n_val,
        version=VERSION,
        space="lambda",
        hotpix_fix=True,
    )
    print(f"\nWeights saved → {weights_path}")

    # Diagnostic scatter (in λ-space, since that's the fit space)
    scatter_all = np.concatenate(scatter_buf, axis=0)
    if len(scatter_all) > args.scatter_points:
        pick = rng.choice(len(scatter_all), args.scatter_points, replace=False)
        scatter_all = scatter_all[pick]

    pred = scatter_all[:, :3] @ w
    actual = scatter_all[:, 3]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: λ-space (native fit space)
    ax = axes[0]
    ax.scatter(actual, pred, s=0.8, alpha=0.08, c="steelblue", rasterized=True)
    lam_max = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, lam_max], [0, lam_max], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Actual λ_mono  (= -ln(1 - p_mono))", fontsize=12)
    ax.set_ylabel("Predicted λ_mono  (w · λ_RGB)", fontsize=12)
    ax.set_title(f"λ-space fit  |  Val R² = {r2_val:.4f}", fontsize=12)
    ax.set_xlim(0, lam_max)
    ax.set_ylim(0, lam_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: transformed back to p-space for direct comparison to v1
    p_actual = 1.0 - np.exp(-actual)
    p_pred = 1.0 - np.exp(-pred)
    ax = axes[1]
    ax.scatter(p_actual, p_pred, s=0.8, alpha=0.08, c="darkorange", rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Actual p_mono", fontsize=12)
    ax.set_ylabel("Predicted p_mono  (1 - exp(-w·λ_RGB))", fontsize=12)
    ax.set_title("Transformed back to p-space (for comparison vs. v1)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"RGB → Mono Calibration  [{VERSION}]    "
        f"w = [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]  sum={w.sum():.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = Path(args.output_dir) / "rgb_to_mono_diagnostic_v3.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plot → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
