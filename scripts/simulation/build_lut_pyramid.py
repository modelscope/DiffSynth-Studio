#!/usr/bin/env python3
"""
Build a dyadic spatial pyramid of LUTs from the per-pixel LUT
=============================================================
Version: v1-pyramid

Derives 10 levels at pool sizes [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:

  Pool   Grid       Pixels/cell
  ----   ----       -----------
   1     512×512×256        1
   2     256×256×256        4
   4     128×128×256       16
   8      64× 64×256       64    ← matches the existing super-pixel LUT
  16      32× 32×256      256
  32      16× 16×256    1,024
  64       8×  8×256    4,096
 128       4×  4×256   16,384
 256       2×  2×256   65,536
 512       1×  1×256  262,144   ← matches the existing global LUT

Each level is derived purely by re-grouping the per-pixel samples by coarser
cell index — no dataset re-scan needed.

Saved to a single uncompressed NPZ for fast random access at sampler time:

  level_{L}_flat_values  (float32, 690M)         — sorted by level-L cell
  level_{L}_offsets      (int64, n_cells_L + 1)
  level_{L}_cell_counts  (uint32, n_cells_L)
  level_{L}_shape        (int64[3])              — (H/L, W/L, N_BINS)
  levels                 (int64[10])

Plus full provenance inherited from the per-pixel LUT.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Provenance helper
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")

H, W = 512, 512
N_BINS = 256
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
VERSION = "v1-pyramid"


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-pixel-lut",
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/lut_per_pixel.npz")
    p.add_argument("--output",
                   default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/lut_pyramid.npz")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(f"  LUT Pyramid Build  [{VERSION}]")
    print("=" * 70)
    print(f"  Source: {args.per_pixel_lut}")
    print(f"  Output: {args.output}")
    print(f"  Levels: {LEVELS}")
    print()

    # --- Load per-pixel LUT and inherit provenance ---
    print("Loading per-pixel LUT …")
    pp = np.load(args.per_pixel_lut)
    flat_values = pp["flat_values"]   # 690M float32
    pp_offsets = pp["offsets"]
    pp_cell_counts = pp["cell_counts"]
    n_samples = len(flat_values)
    print(f"  flat_values: {n_samples:,} float32  ({flat_values.nbytes/1e9:.2f} GB)")
    print(f"  offsets:     {len(pp_offsets):,} int64")
    print()

    # --- Recover (y, x, b) for each sample via np.repeat on cell counts ---
    # cell_idx = y * W * N_BINS + x * N_BINS + b
    print("Recovering (y, x, b) per sample via np.repeat …")
    t0 = time.time()
    cell_idx_per_sample = np.repeat(
        np.arange(len(pp_cell_counts), dtype=np.uint32),
        pp_cell_counts,
    )
    assert len(cell_idx_per_sample) == n_samples, (
        f"Expected {n_samples} samples but np.repeat gave {len(cell_idx_per_sample)}"
    )
    y_per_sample = (cell_idx_per_sample // (W * N_BINS)).astype(np.uint16)
    x_per_sample = ((cell_idx_per_sample // N_BINS) % W).astype(np.uint16)
    b_per_sample = (cell_idx_per_sample % N_BINS).astype(np.uint8)
    del cell_idx_per_sample
    del pp_cell_counts, pp_offsets   # free
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  y_per_sample uint16: {y_per_sample.nbytes/1e9:.2f} GB")
    print()

    # --- Build each level ---
    save_dict = {"levels": np.array(LEVELS, dtype=np.int64)}

    # Inherit provenance from the per-pixel LUT
    INHERIT = ("artifact_kind", "n_scenes_used", "n_scenes_requested",
               "n_scenes_skipped", "total_samples", "version",
               "build_timestamp", "accumulator_module", "k_short",
               "n_gt_frames", "rotate_k", "hotpix_fix_enabled", "seed",
               "mono_bin", "images_dir", "scene_list_hash")
    for k in INHERIT:
        if k in pp.files:
            save_dict[f"source_{k}"] = pp[k]

    save_dict["artifact_kind"] = "spatial_pyramid_lut"
    save_dict["pyramid_version"] = VERSION
    import datetime
    save_dict["pyramid_build_timestamp"] = datetime.datetime.now().isoformat()

    for L in LEVELS:
        H_L = H // L
        W_L = W // L
        n_cells_L = H_L * W_L * N_BINS
        print(f"--- Level {L}  (grid {H_L}×{W_L}, {n_cells_L:,} cells) ---")
        t_lvl = time.time()

        # Compute level-L cell index per sample
        # cell_L = (y // L) * W_L * N_BINS + (x // L) * N_BINS + b
        cell_L = (
            (y_per_sample.astype(np.uint32) // L) * (W_L * N_BINS)
            + (x_per_sample.astype(np.uint32) // L) * N_BINS
            + b_per_sample.astype(np.uint32)
        )
        print(f"  cell_L computed in {time.time()-t_lvl:.1f}s")

        # Sort
        t = time.time()
        order = np.argsort(cell_L, kind="stable")
        cell_L_sorted = cell_L[order]
        flat_sorted = flat_values[order]
        del cell_L, order
        print(f"  argsort+gather in {time.time()-t:.1f}s")

        # Offsets via searchsorted
        t = time.time()
        offsets = np.searchsorted(
            cell_L_sorted,
            np.arange(n_cells_L + 1, dtype=np.uint32),
            side="left",
        ).astype(np.int64)
        cell_counts = np.diff(offsets).astype(np.uint32)
        del cell_L_sorted
        print(f"  offsets in {time.time()-t:.1f}s")

        # Stats
        nz = cell_counts[cell_counts > 0]
        empty = int((cell_counts == 0).sum())
        thresholds = [1, 5, 10, 20, 50]
        thr_fracs = [(int((cell_counts >= N).sum()), 100*float((cell_counts >= N).sum())/n_cells_L)
                     for N in thresholds]
        print(f"  {empty:,} empty ({100*empty/n_cells_L:.2f}%); "
              f"non-empty min={nz.min() if len(nz) else 0}, "
              f"median={int(np.median(nz)) if len(nz) else 0}, "
              f"mean={float(nz.mean()) if len(nz) else 0:.1f}, "
              f"max={nz.max() if len(nz) else 0}")
        for N, (count, pct) in zip(thresholds, thr_fracs):
            print(f"    fraction with ≥{N:>3}: {pct:6.2f}%")

        save_dict[f"level_{L}_flat_values"] = flat_sorted
        save_dict[f"level_{L}_offsets"] = offsets
        save_dict[f"level_{L}_cell_counts"] = cell_counts
        save_dict[f"level_{L}_shape"] = np.array([H_L, W_L, N_BINS], dtype=np.int64)
        # Per-level threshold fractions for quick reference
        save_dict[f"level_{L}_threshold_counts"] = np.array(
            [c for c, _ in thr_fracs], dtype=np.int64
        )
        save_dict[f"level_{L}_threshold_fractions"] = np.array(
            [p for _, p in thr_fracs], dtype=np.float64
        )

        del flat_sorted, offsets, cell_counts, nz
        print(f"  Level {L} done in {time.time()-t_lvl:.1f}s\n")

    print("Writing NPZ …")
    t = time.time()
    np.savez(args.output, **save_dict)
    print(f"  Wrote in {time.time()-t:.1f}s — {os.path.getsize(args.output)/1e9:.2f} GB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
