#!/usr/bin/env python3
"""
RGB-to-Mono Calibration on pre-processed PNGs  —  v4
=================================================================
Version: v4

Fits a linear map directly on the pre-generated processed PNG pair:

    y  (mono λ, tone-mapped, uint16 → [0,1])
        from  spad_dataset/monochrome16/*_RAW_empty_frames0-19999_lambda.png
    X  (linear RGB, post-WB, post-tone-map, uint8 → [0,1])
        from  spad_dataset/RGB_linear16/*_frames0-19999_linear16.png

    y ≈ w_r·R + w_g·G + w_b·B

This is the "pragmatic" calibration that operates in the same space online
sRGB would decode into (after gamma inversion) — treating our processed
"linear" RGB as a stand-in.

CAVEATS (read before interpreting weights):
  - Both X and y have had **per-scene tone-mapping** applied (divide by
    99.5th percentile). The scale factors are different for RGB vs. mono
    AND differ across scenes. Pooling the fit across 2632 scenes averages
    this out to first order but will bias the absolute magnitudes of the
    weights vs. the physics-pure v3.
  - X has also had gray-world white balance applied per scene.
  - Therefore v4 weights are NOT directly physically meaningful; they are
    useful only as a comparison check against v3.

Outputs:
  calibration/rgb_to_mono_weights_v4.npz
  calibration/rgb_to_mono_diagnostic_v4.png
  calibration/calibration_log_v4.txt
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
H, W = 512, 512
RATE_LO = 0.01   # same bounds as v1/v3 for apples-to-apples on pixel exclusion
RATE_HI = 0.95
VERSION = "v4"


def load_png_normalized(path):
    """Load a PNG, auto-detect dtype, return float32 in [0,1]."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        return arr.astype(np.float32)


def scene_id_from_rgb(fname):
    # '0724-dgp-001_frames0-19999_linear16.png' → '0724-dgp-001'
    return fname.split("_frames")[0]


def scene_id_from_mono(fname):
    # '0724-dgp-001_RAW_empty_frames0-19999_lambda.png' → '0724-dgp-001'
    return fname.split("_RAW")[0]


def compute_r2(XtX, Xty, yty, sumy, N, w):
    ss_res = yty - 2.0 * w @ Xty + w @ XtX @ w
    mean_y = sumy / N
    ss_tot = yty - N * mean_y ** 2
    return 1.0 - ss_res / ss_tot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-dir", type=str,
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/spad_dataset/RGB_linear16")
    p.add_argument("--mono-dir", type=str,
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/spad_dataset/monochrome16")
    p.add_argument("--output-dir", type=str,
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scatter-points", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rgb_dir = Path(args.rgb_dir)
    mono_dir = Path(args.mono_dir)

    print("=" * 64)
    print(f"  RGB → Mono Calibration on processed PNGs  [{VERSION}]")
    print("=" * 64)
    print(f"  X source: {rgb_dir}")
    print(f"  y source: {mono_dir}")
    print(f"  Space:    RGB_linear16 ↔ monochrome16_lambda (both tone-mapped)")
    print()

    # Build scene → (rgb_path, mono_path) pairing
    rgb_by_scene = {}
    for f in rgb_dir.iterdir():
        if f.suffix == ".png":
            sid = scene_id_from_rgb(f.name)
            rgb_by_scene[sid] = f

    mono_by_scene = {}
    for f in mono_dir.iterdir():
        if f.suffix == ".png":
            sid = scene_id_from_mono(f.name)
            mono_by_scene[sid] = f

    common = sorted(set(rgb_by_scene.keys()) & set(mono_by_scene.keys()))
    if args.max_scenes is not None:
        common = common[:args.max_scenes]

    n_total = len(common)
    print(f"RGB files:   {len(rgb_by_scene)}")
    print(f"Mono files:  {len(mono_by_scene)}")
    print(f"Paired:      {n_total}")
    if n_total == 0:
        sys.exit("ERROR: no matched scenes")

    # Same seed + logic as v1/v3 so split is consistent
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * args.val_fraction))
    val_set = set(perm[:n_val].tolist())
    n_train = n_total - n_val
    print(f"Train scenes: {n_train}   Val scenes: {n_val}")
    print()

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

    try:
        from tqdm import tqdm
        scene_iter = tqdm(enumerate(common), total=n_total, desc="Scenes")
    except ImportError:
        scene_iter = enumerate(common)

    for idx, scene in scene_iter:
        split = "val" if idx in val_set else "train"
        try:
            rgb = load_png_normalized(str(rgb_by_scene[scene]))  # (H, W, 3), [0,1]
            mono = load_png_normalized(str(mono_by_scene[scene]))  # (H, W), [0,1]
        except Exception as e:
            print(f"[skip] {scene}: {e}")
            skipped += 1
            continue

        if rgb.shape[:2] != (H, W) or mono.shape != (H, W):
            print(f"[skip] {scene}: shape mismatch {rgb.shape}, {mono.shape}")
            skipped += 1
            continue
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            skipped += 1
            continue

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Mask: exclude pixels near 0 or 1 in ANY channel
        mask = np.ones((H, W), dtype=bool)
        for ch in (r, g, b, mono):
            mask &= (ch > RATE_LO) & (ch < RATE_HI)
        n_valid = int(mask.sum())
        if n_valid < 10:
            skipped += 1
            continue

        X_px = np.column_stack([r[mask], g[mask], b[mask]]).astype(np.float64)
        y_px = mono[mask].astype(np.float64)

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

    w = np.linalg.solve(stats["train"]["XtX"], stats["train"]["Xty"])
    r2_train = compute_r2(stats["train"]["XtX"], stats["train"]["Xty"],
                          stats["train"]["yty"], stats["train"]["sumy"],
                          stats["train"]["N"], w)
    r2_val = compute_r2(stats["val"]["XtX"], stats["val"]["Xty"],
                        stats["val"]["yty"], stats["val"]["sumy"],
                        stats["val"]["N"], w) if stats["val"]["N"] > 0 else float("nan")

    print("\n" + "=" * 64)
    print(f"  RESULTS  [{VERSION}]    (weights in processed-PNG space)")
    print("=" * 64)
    print(f"  w_r  = {w[0]:.6f}")
    print(f"  w_g  = {w[1]:.6f}")
    print(f"  w_b  = {w[2]:.6f}")
    print(f"  sum  = {w.sum():.6f}")
    print()
    print(f"  Train R² = {r2_train:.6f}   ({stats['train']['N']:>12,} pixels)")
    print(f"  Val   R² = {r2_val:.6f}   ({stats['val']['N']:>12,} pixels)")
    print("=" * 64)

    weights_path = Path(args.output_dir) / "rgb_to_mono_weights_v4.npz"
    np.savez(
        weights_path,
        weights=w, w_r=w[0], w_g=w[1], w_b=w[2],
        train_r2=r2_train, val_r2=r2_val,
        train_N=stats["train"]["N"], val_N=stats["val"]["N"],
        rate_lo=RATE_LO, rate_hi=RATE_HI,
        n_train_scenes=n_train, n_val_scenes=n_val,
        version=VERSION,
        space="processed_png",
    )
    print(f"\nWeights saved → {weights_path}")

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
    ax.scatter(actual, pred, s=0.8, alpha=0.08, c="forestgreen", rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y = x")
    ax.set_xlabel("Actual mono (monochrome16 PNG, normalized)", fontsize=12)
    ax.set_ylabel("Predicted mono  (w · RGB_linear16)", fontsize=12)
    ax.set_title(
        f"RGB_linear16 → monochrome16  [{VERSION}]\n"
        f"Val R² = {r2_val:.4f}    "
        f"w = [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = Path(args.output_dir) / "rgb_to_mono_diagnostic_v4.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plot → {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
