#!/usr/bin/env python3
"""
RGB-to-Mono Bernoulli Rate Calibration  —  v1
==========================================================
Version: v1

Calibrates a linear map from RGB color-filtered SPAD Bernoulli rates
to monochrome (unfiltered) Bernoulli rates:

    p_mono ≈ w_r · p_r  +  w_g · p_g  +  w_b · p_b

For each scene, loads raw binary captures for R, G, B, and mono channels,
computes Bernoulli rates (p = counts / num_frames), and solves ordinary
least-squares via the normal equations (equivalent to np.linalg.lstsq).

Memory-efficient: accumulates X^T X (3×3) and X^T y (3×1) incrementally.
Uses Numba-accelerated bit unpacking from the existing SPAD codebase.

Outputs:
  - calibration/rgb_to_mono_weights.npz   (weights + metadata)
  - calibration/rgb_to_mono_diagnostic.png (predicted-vs-actual scatter)
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Import Numba-accelerated SPAD utilities from existing codebase
# ---------------------------------------------------------------------------
# NOTE (2026-04-16): switched to race-free spad_utils_fixed after audit.
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from spad_utils_fixed import accumulate_counts_whole_file  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8  # 32 768

CHANNEL_FILES = {
    "r": "RAW_col_r.bin",
    "g": "RAW_col_g.bin",
    "b": "RAW_col_b.bin",
    "mono": "RAW_empty.bin",
}

RATE_LO = 0.01   # skip pixels below this
RATE_HI = 0.95   # skip pixels above this
VERSION = "v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_bernoulli_rate(bin_path: str, max_frames: int, rotate_k: int) -> np.ndarray:
    """
    Read a SPAD binary file and return the Bernoulli rate image (512×512 float32).

    Only the first *max_frames* frames are read from disk; remaining bytes
    are not touched.  Uses Numba-accelerated ``accumulate_counts_whole_file``.
    """
    bytes_needed = max_frames * BYTES_PER_FRAME
    with open(bin_path, "rb") as f:
        raw = np.frombuffer(f.read(bytes_needed), dtype=np.uint8)
    counts, n_frames = accumulate_counts_whole_file(raw, max_frames, H, W)
    rate = counts.astype(np.float32) / float(n_frames)
    if rotate_k % 4 != 0:
        rate = np.rot90(rate, k=rotate_k)
    return rate


def compute_r2(XtX, Xty, yty, sumy, N, w):
    """Compute R² from pre-accumulated sufficient statistics."""
    ss_res = yty - 2.0 * w @ Xty + w @ XtX @ w
    mean_y = sumy / N
    ss_tot = yty - N * mean_y ** 2
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate linear RGB→mono Bernoulli-rate map"
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default="/nfs/horai.dgpsrv/ondemand30/jw954/images",
        help="Root directory containing scene subdirectories",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration",
        help="Where to save weights and plot",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Number of binary frames to accumulate per file (default: 1000)",
    )
    p.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit number of scenes (for quick testing)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of scenes held out for validation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    p.add_argument(
        "--scatter-points",
        type=int,
        default=50_000,
        help="Number of points in the diagnostic scatter plot",
    )
    p.add_argument(
        "--rotate-k",
        type=int,
        default=1,
        help="np.rot90 rotations (1 = 90° CCW, matching existing pipeline)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = Path(args.images_dir)

    print(f"{'=' * 64}")
    print(f"  RGB → Mono Bernoulli Rate Calibration  [{VERSION}]")
    print(f"{'=' * 64}")

    # ------------------------------------------------------------------
    # 1. Enumerate scenes that have all four binary files
    # ------------------------------------------------------------------
    all_scenes = sorted(
        d.name
        for d in images_dir.iterdir()
        if d.is_dir()
        and all((d / fn).exists() for fn in CHANNEL_FILES.values())
    )
    if args.max_scenes is not None:
        all_scenes = all_scenes[: args.max_scenes]

    n_total = len(all_scenes)
    print(f"\nScenes with all 4 channels: {n_total}")
    if n_total == 0:
        print("ERROR: no valid scenes found. Check --images-dir.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Train / val split (scene-level, deterministic)
    # ------------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_fraction))
    val_set = set(perm[:n_val].tolist())

    n_train = n_total - n_val
    print(f"Train scenes: {n_train}   Val scenes: {n_val}")
    print(f"Max frames per file: {args.max_frames}")
    print(f"Pixel filter: {RATE_LO} < rate < {RATE_HI}")
    print()

    # ------------------------------------------------------------------
    # 3. Accumulate sufficient statistics (memory-efficient)
    #
    #    Normal equations:  w = (X^T X)^{-1} X^T y
    #    R²:  needs  y^T y,  sum(y),  N   for each split
    # ------------------------------------------------------------------
    stats = {
        split: dict(
            XtX=np.zeros((3, 3), dtype=np.float64),
            Xty=np.zeros(3, dtype=np.float64),
            yty=np.float64(0.0),
            sumy=np.float64(0.0),
            N=0,
        )
        for split in ("train", "val")
    }

    # Reservoir for scatter plot (over-collect, then trim)
    scatter_buf = []
    samples_per_scene = max(10, (args.scatter_points * 3) // n_total)

    skipped = 0
    t0 = time.time()

    # Warm-up Numba JIT on a tiny dummy array
    print("Warming up Numba JIT compiler …")
    _dummy = np.zeros(BYTES_PER_FRAME, dtype=np.uint8)
    accumulate_counts_whole_file(_dummy, 1, H, W)
    print("JIT ready.\n")

    # Use tqdm if available, else plain loop
    try:
        from tqdm import tqdm
        scene_iter = tqdm(enumerate(all_scenes), total=n_total, desc="Scenes")
    except ImportError:
        scene_iter = enumerate(all_scenes)

    for idx, scene in scene_iter:
        split = "val" if idx in val_set else "train"
        scene_dir = images_dir / scene

        # Load all four Bernoulli rate images
        try:
            rates = {}
            for ch, fn in CHANNEL_FILES.items():
                rates[ch] = load_bernoulli_rate(
                    str(scene_dir / fn), args.max_frames, args.rotate_k
                )
        except Exception as e:
            if not isinstance(scene_iter, enumerate):
                tqdm.write(f"[skip] {scene}: {e}")
            else:
                print(f"[skip] {scene}: {e}")
            skipped += 1
            continue

        # Pixel mask: exclude dark / saturated in ANY channel
        mask = np.ones((H, W), dtype=bool)
        for ch in ("r", "g", "b", "mono"):
            mask &= (rates[ch] > RATE_LO) & (rates[ch] < RATE_HI)

        n_valid = int(mask.sum())
        if n_valid < 10:
            skipped += 1
            continue

        # Extract valid pixels  →  X (N, 3), y (N,)
        X_px = np.column_stack(
            [rates["r"][mask], rates["g"][mask], rates["b"][mask]]
        ).astype(np.float64)
        y_px = rates["mono"][mask].astype(np.float64)

        # Update sufficient statistics
        s = stats[split]
        s["XtX"] += X_px.T @ X_px
        s["Xty"] += X_px.T @ y_px
        s["yty"] += y_px @ y_px
        s["sumy"] += y_px.sum()
        s["N"] += n_valid

        # Subsample for scatter plot
        n_take = min(samples_per_scene, n_valid)
        pick = rng.choice(n_valid, n_take, replace=False)
        scatter_buf.append(
            np.column_stack([X_px[pick], y_px[pick]])
        )  # shape (n_take, 4)

    elapsed = time.time() - t0
    print(f"\nProcessed {n_total - skipped}/{n_total} scenes in {elapsed:.1f}s")
    print(f"Skipped {skipped} scenes")
    print(f"Train pixels: {stats['train']['N']:,}")
    print(f"Val   pixels: {stats['val']['N']:,}")

    if stats["train"]["N"] == 0:
        print("ERROR: no valid training pixels. Check data or filters.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Solve normal equations  (equivalent to np.linalg.lstsq)
    # ------------------------------------------------------------------
    w = np.linalg.solve(stats["train"]["XtX"], stats["train"]["Xty"])

    # R² on each split
    r2_train = compute_r2(
        stats["train"]["XtX"], stats["train"]["Xty"],
        stats["train"]["yty"], stats["train"]["sumy"],
        stats["train"]["N"], w,
    )
    r2_val = compute_r2(
        stats["val"]["XtX"], stats["val"]["Xty"],
        stats["val"]["yty"], stats["val"]["sumy"],
        stats["val"]["N"], w,
    ) if stats["val"]["N"] > 0 else float("nan")

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  RESULTS  [{VERSION}]")
    print(f"{'=' * 64}")
    print(f"  w_r  = {w[0]:.6f}")
    print(f"  w_g  = {w[1]:.6f}")
    print(f"  w_b  = {w[2]:.6f}")
    print(f"  sum  = {w.sum():.6f}")
    print()
    print(f"  Train R² = {r2_train:.6f}   ({stats['train']['N']:>12,} pixels)")
    print(f"  Val   R² = {r2_val:.6f}   ({stats['val']['N']:>12,} pixels)")
    print(f"{'=' * 64}")

    # ------------------------------------------------------------------
    # 6. Save weights
    # ------------------------------------------------------------------
    weights_path = Path(args.output_dir) / "rgb_to_mono_weights.npz"
    np.savez(
        weights_path,
        weights=w,
        w_r=w[0],
        w_g=w[1],
        w_b=w[2],
        train_r2=r2_train,
        val_r2=r2_val,
        train_N=stats["train"]["N"],
        val_N=stats["val"]["N"],
        max_frames=args.max_frames,
        rate_lo=RATE_LO,
        rate_hi=RATE_HI,
        n_train_scenes=n_train,
        n_val_scenes=n_val,
        version=VERSION,
    )
    print(f"\nWeights saved → {weights_path}")

    # ------------------------------------------------------------------
    # 7. Diagnostic scatter plot
    # ------------------------------------------------------------------
    scatter_all = np.concatenate(scatter_buf, axis=0)
    if len(scatter_all) > args.scatter_points:
        pick = rng.choice(len(scatter_all), args.scatter_points, replace=False)
        scatter_all = scatter_all[pick]

    pred = scatter_all[:, :3] @ w
    actual = scatter_all[:, 3]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, pred, s=0.8, alpha=0.08, c="steelblue", rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Actual Mono Bernoulli Rate", fontsize=12)
    ax.set_ylabel("Predicted Mono Rate  (w·RGB)", fontsize=12)
    ax.set_title(
        f"RGB → Mono Calibration  [{VERSION}]\n"
        f"Val R² = {r2_val:.4f}    "
        f"w = [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = Path(args.output_dir) / "rgb_to_mono_diagnostic.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plot → {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
