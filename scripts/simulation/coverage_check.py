#!/usr/bin/env python3
"""
Coverage feasibility check for per-pixel SPAD Bernoulli noise LUT
=================================================================
Version: v1

Fast pre-pass that counts how many (pixel, GT_bin) cells would receive
≥N samples for various N thresholds.  Uses pre-extracted 1000-frame
probability PNGs from bits_multi_1000/ to avoid re-reading raw binaries.

Reports:
  - Fraction of all 512×512×256 cells meeting each threshold
  - Per-pixel: fraction of pixels where ≥200/256 bins meet threshold
  - Also: super-pixel (64×64×256) and global (256) coverage for comparison
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

H, W = 512, 512
N_BINS = 256
THRESHOLDS = [1, 5, 10, 20, 50]


def load_p_png(path):
    """Load a 16-bit probability PNG and return float32 in [0,1]."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bits-dir",
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/spad_dataset/bits_multi_1000")
    p.add_argument("--output-dir",
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--max-scenes", type=int, default=None)
    args = p.parse_args()

    bits_dir = Path(args.bits_dir)
    pngs = sorted(bits_dir.glob("*_RAW_empty_*_p.png"))
    if args.max_scenes:
        pngs = pngs[:args.max_scenes]
    n_scenes = len(pngs)
    print(f"Coverage check: {n_scenes} probability PNGs from {bits_dir.name}")

    # Per-pixel counts: (H, W, N_BINS) uint16 — 134 MB
    counts_pp = np.zeros((H, W, N_BINS), dtype=np.uint16)
    t0 = time.time()

    try:
        from tqdm import tqdm
        png_iter = tqdm(pngs, desc="Scenes")
    except ImportError:
        png_iter = pngs

    for png_path in png_iter:
        p_gt = load_p_png(str(png_path))  # (512, 512)
        if p_gt.shape != (H, W):
            continue
        bin_idx = np.clip((p_gt * 255.0 + 0.5).astype(np.int32), 0, N_BINS - 1)
        # Vectorized increment — use advanced indexing
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        counts_pp[yy.ravel(), xx.ravel(), bin_idx.ravel()] += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/n_scenes:.2f}s/scene)")

    # =========================================================================
    # Analysis
    # =========================================================================
    total_cells_pp = H * W * N_BINS       # 67,108,864
    total_pixels = H * W                   # 262,144

    # Super-pixel counts: sum 8×8 blocks
    counts_sp = counts_pp.reshape(H // 8, 8, W // 8, 8, N_BINS).sum(axis=(1, 3))
    total_cells_sp = (H // 8) * (W // 8) * N_BINS  # 1,048,576

    # Global counts: sum over all pixels
    counts_gl = counts_pp.sum(axis=(0, 1))  # (256,)

    print("\n" + "=" * 80)
    print("  COVERAGE FEASIBILITY CHECK")
    print("=" * 80)
    print(f"  Scenes: {n_scenes}")
    print(f"  GT source: bits_multi_1000 (1000-frame probability, 16-bit)")
    print()

    # --- Global (256 bins) ---
    print("  GLOBAL LUT (256 bins):")
    for N in THRESHOLDS:
        frac = (counts_gl >= N).sum() / N_BINS
        print(f"    N≥{N:>3}: {100*frac:6.1f}% of bins ({(counts_gl >= N).sum()}/{N_BINS})")
    print(f"    min count = {counts_gl.min():,},  max = {counts_gl.max():,},  "
          f"median = {int(np.median(counts_gl)):,}")
    # Flag any bins with < 1000 samples
    sparse_gl = np.where(counts_gl < 1000)[0]
    if len(sparse_gl):
        print(f"    ⚠ {len(sparse_gl)} bins with <1000 samples: bins {sparse_gl[:10]}{'...' if len(sparse_gl)>10 else ''}")
    else:
        print(f"    ✓ All bins have ≥1000 samples")
    print()

    # --- Super-pixel (64×64×256) ---
    print("  SUPER-PIXEL LUT (64×64×256 = 1,048,576 cells):")
    counts_sp_flat = counts_sp.ravel()
    for N in THRESHOLDS:
        frac = (counts_sp_flat >= N).sum() / total_cells_sp
        print(f"    N≥{N:>3}: {100*frac:6.1f}% of cells  "
              f"({(counts_sp_flat >= N).sum():,}/{total_cells_sp:,})")
    print()

    # --- Per-pixel (512×512×256) ---
    print("  PER-PIXEL LUT (512×512×256 = 67,108,864 cells):")
    counts_pp_flat = counts_pp.ravel()
    for N in THRESHOLDS:
        frac = (counts_pp_flat >= N).sum() / total_cells_pp
        print(f"    N≥{N:>3}: {100*frac:6.1f}% of cells  "
              f"({(counts_pp_flat >= N).sum():,}/{total_cells_pp:,})")
    nz = counts_pp_flat[counts_pp_flat > 0]
    print(f"    non-empty cell counts: min={nz.min() if len(nz) else 0}, "
          f"median={int(np.median(nz)) if len(nz) else 0}, "
          f"mean={float(nz.mean()) if len(nz) else 0:.1f}, "
          f"max={nz.max() if len(nz) else 0}")
    print()

    # --- Per-pixel bin coverage (how many bins per pixel are populated?) ---
    print("  PER-PIXEL BIN COVERAGE (≥200/256 bins populated → good coverage):")
    for N in THRESHOLDS:
        bins_with_enough = (counts_pp >= N).sum(axis=2)  # (H, W) = bins meeting threshold per pixel
        good_pixels = (bins_with_enough >= 200).sum()
        any_pixels = (bins_with_enough >= 1).sum()
        median_bins = int(np.median(bins_with_enough))
        mean_bins = float(bins_with_enough.mean())
        print(f"    N≥{N:>3}: {100*good_pixels/total_pixels:5.1f}% pixels have ≥200/256 bins  "
              f"(median bins/pixel: {median_bins}, mean: {mean_bins:.1f})")
    print()

    # Summary verdict
    frac_nonempty_pp = (counts_pp_flat > 0).sum() / total_cells_pp
    frac_ge5_pp = (counts_pp_flat >= 5).sum() / total_cells_pp
    frac_ge20_pp = (counts_pp_flat >= 20).sum() / total_cells_pp

    print("  VERDICT:")
    if frac_nonempty_pp < 0.5:
        print("    ⚠ Per-pixel is FUNDAMENTALLY UNDERSAMPLED (>50% cells empty).")
        print("      Fallback hierarchy will do most of the work.")
    elif frac_ge20_pp > 0.8:
        print("    ✓ Per-pixel is VIABLE (>80% cells have ≥20 samples).")
    elif frac_ge5_pp > 0.5:
        print("    ~ Per-pixel is PARTIALLY viable. Many cells need fallback to super-pixel.")
    else:
        print("    ⚠ Per-pixel is SPARSE. Super-pixel or global fallback needed for most queries.")
    print("=" * 80)

    # Save counts for later use (with provenance for cross-artifact compatibility check)
    import hashlib, datetime
    scene_ids = sorted({fn.name.split("_RAW_empty_")[0] for fn in pngs})
    scene_list_hash = hashlib.sha256(("\n".join(scene_ids)).encode()).hexdigest()
    out_path = Path(args.output_dir) / "coverage_counts.npz"
    np.savez_compressed(
        out_path,
        counts_per_pixel=counts_pp,       # (512, 512, 256) uint16
        counts_super_pixel=counts_sp,     # (64, 64, 256) uint32
        counts_global=counts_gl,          # (256,) uint64
        n_scenes=n_scenes,
        thresholds=np.array(THRESHOLDS),
        # Provenance
        artifact_kind="coverage_counts",
        n_scenes_used=np.int32(n_scenes),
        n_scenes_requested=np.int32(n_scenes),
        scene_list_hash=scene_list_hash,
        scene_list=np.array(scene_ids),
        bits_dir=str(bits_dir),
        build_timestamp=datetime.datetime.now().isoformat(),
        version="v1-with-provenance",
    )
    print(f"\nCounts saved → {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
