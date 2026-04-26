#!/usr/bin/env python3
"""
Build three parallel Bernoulli noise LUTs in a single pass
==========================================================
Version: v2  (post-audit, race-free)

All three index by GT intensity bin (256 bins) and store empirical p̂ samples
from K=64-frame short bursts. They differ only in spatial granularity:

  1. Global:      shape (256,)                 → lut_global.npz
  2. Super-pixel: shape (64, 64, 256)          → spad_bernoulli_lut_v2.npz
  3. Per-pixel:   shape (512, 512, 256)        → lut_per_pixel.npz

NUMPY AXIS ORDER (clarified post-audit Finding 6):
  The flat ragged offsets index cells in [sy, sx, bin] order — matching
  numpy's default row-major (y is row, x is col, bin is innermost).
  Caller code must compute cell_idx = sy * N_SUPER * N_BINS + sx * N_BINS + b.

Single pass through dataset to avoid 3× I/O cost.  For each scene, compute
p_GT and p̂_short once, then update all three simultaneously.

Hot pixel fix (--hotpix-fix, default ON) is applied to the raw counts of
both p_short and p_GT before normalizing to rates. This was inconsistent in
the original build_bernoulli_lut.py (audit Finding 4) — fixed here.

Also accumulates per-pixel per-bin sufficient statistics (sum, sum_sq, count)
for the variance decomposition analysis — no additional pass needed.
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# NOTE (2026-04-16): switched from utils.accumulate_counts_whole_file to the
# race-free spad_utils_fixed.accumulate_counts_whole_file after audit found
# nondeterministic undercounts in the original (Numba prange race on shared
# counts array). See archive_pre_audit_2026-04-16/README.md.
sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from spad_utils_fixed import accumulate_counts_whole_file, median_hotpixel_fix  # noqa: E402

H, W = 512, 512
BYTES_PER_FRAME = (H * W) // 8
N_BINS = 256
N_SUPER = 64
SUPER_PX = 8
MONO_BIN = "RAW_empty.bin"
VERSION = "v3-post-audit2"   # adds provenance metadata to all saved artifacts

TOTAL_CELLS_PP = H * W * N_BINS             # 67,108,864
TOTAL_CELLS_SP = N_SUPER * N_SUPER * N_BINS  # 1,048,576


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/images")
    p.add_argument("--output-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--k-short", type=int, default=64)
    p.add_argument("--n-gt-frames", type=int, default=20_000)
    p.add_argument("--rotate-k", type=int, default=1)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--hotpix-fix", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = Path(args.images_dir)

    scenes = sorted(
        d.name for d in images_dir.iterdir()
        if d.is_dir() and (d / MONO_BIN).exists()
    )
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    n_scenes = len(scenes)

    print("=" * 70)
    print(f"  Three-LUT Build  [{VERSION}]")
    print("=" * 70)
    print(f"  Scenes:    {n_scenes}")
    print(f"  K_short:   {args.k_short}")
    print(f"  N_GT:      {args.n_gt_frames}")
    print(f"  Rotate:    k={args.rotate_k}")
    print(f"  Hotpix:    {'ON' if args.hotpix_fix else 'OFF'}")
    print()

    # ==========================================================================
    # Preallocate
    # ==========================================================================
    N_max = n_scenes * H * W

    # Per-pixel LUT: store (cell_idx, p_hat) pairs; sort at end
    pp_cell = np.empty(N_max, dtype=np.uint32)
    pp_phat = np.empty(N_max, dtype=np.float32)

    # Super-pixel LUT: same approach
    sp_cell = np.empty(N_max, dtype=np.uint32)
    # (shares pp_phat — same values, different grouping)

    # Global LUT: just bin index (uint8)
    gl_bin = np.empty(N_max, dtype=np.uint8)

    # Variance decomposition stats: per-pixel per-bin (sum, sum_sq, count)
    # Using float64 accumulators for numerical stability
    # 512×512×256 × 8 bytes × 2 + uint16 count = 512 + 512 + 134 = 1.16 GB
    var_sum = np.zeros((H, W, N_BINS), dtype=np.float64)     # 512 MB
    var_sum2 = np.zeros((H, W, N_BINS), dtype=np.float64)    # 512 MB
    var_count = np.zeros((H, W, N_BINS), dtype=np.uint16)     # 134 MB

    fill = 0
    scenes_used = []      # provenance: which scenes actually contributed
    scenes_skipped = []   # provenance: which scenes were skipped and why

    # Precompute coordinate grids
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sy = (yy // SUPER_PX).astype(np.uint32)
    sx = (xx // SUPER_PX).astype(np.uint32)
    sp_base = (sy * N_SUPER + sx).ravel().astype(np.uint32)   # (H*W,) — super-pixel part
    pp_base = (yy * W + xx).ravel().astype(np.uint32)         # (H*W,) — pixel part
    pp_base_bins = pp_base * N_BINS                            # (H*W,) — multiply once

    # Warm up numba
    print("Warming up Numba …")
    _d = np.zeros(BYTES_PER_FRAME, dtype=np.uint8)
    accumulate_counts_whole_file(_d, 1, H, W)

    t0 = time.time()
    skipped = 0

    try:
        from tqdm import tqdm
        scene_iter = tqdm(enumerate(scenes), total=n_scenes, desc="Scenes")
    except ImportError:
        scene_iter = enumerate(scenes)

    for i, scene in scene_iter:
        bin_path = str(images_dir / scene / MONO_BIN)
        try:
            bytes_needed = args.n_gt_frames * BYTES_PER_FRAME
            with open(bin_path, "rb") as f:
                raw = np.frombuffer(f.read(bytes_needed), dtype=np.uint8)
            n_avail = len(raw) // BYTES_PER_FRAME
            if n_avail < args.k_short + 1:
                skipped += 1
                scenes_skipped.append((scene, f"only {n_avail} frames available"))
                continue

            # Short burst: first K frames
            raw_short = raw[: args.k_short * BYTES_PER_FRAME]
            counts_short, n_short = accumulate_counts_whole_file(raw_short, args.k_short, H, W)
            if args.hotpix_fix:
                counts_short = median_hotpixel_fix(counts_short, n_short)
            p_short = counts_short.astype(np.float32) / float(n_short)

            # GT: all frames
            counts_gt, n_gt = accumulate_counts_whole_file(raw[:n_avail * BYTES_PER_FRAME], n_avail, H, W)
            if args.hotpix_fix:
                counts_gt = median_hotpixel_fix(counts_gt, n_gt)
            p_gt = counts_gt.astype(np.float32) / float(n_gt)
        except Exception as e:
            print(f"\n[skip] {scene}: {e}")
            skipped += 1
            scenes_skipped.append((scene, str(e)))
            continue

        scenes_used.append(scene)

        # Rotate
        if args.rotate_k % 4 != 0:
            p_short = np.rot90(p_short, k=args.rotate_k)
            p_gt = np.rot90(p_gt, k=args.rotate_k)

        # Bin index
        bin_idx = np.clip((p_gt * 255.0 + 0.5).astype(np.int32), 0, N_BINS - 1).astype(np.uint32)
        bin_flat = bin_idx.ravel()
        phat_flat = p_short.ravel().astype(np.float32)

        # Cell indices for all three LUTs
        end = fill + H * W
        pp_cell[fill:end] = pp_base_bins + bin_flat      # per-pixel cell
        sp_cell[fill:end] = sp_base * N_BINS + bin_flat   # super-pixel cell
        gl_bin[fill:end] = bin_flat.astype(np.uint8)       # global bin
        pp_phat[fill:end] = phat_flat

        # Variance decomposition: update per-pixel per-bin stats
        yy_f, xx_f = yy.ravel(), xx.ravel()
        var_sum[yy_f, xx_f, bin_flat] += phat_flat.astype(np.float64)
        var_sum2[yy_f, xx_f, bin_flat] += (phat_flat.astype(np.float64)) ** 2
        var_count[yy_f, xx_f, bin_flat] += 1

        fill = end

    elapsed = time.time() - t0
    print(f"\nProcessed {n_scenes - skipped}/{n_scenes} scenes in {elapsed:.1f}s "
          f"({elapsed/max(1,n_scenes-skipped):.1f}s/scene)")
    print(f"Skipped: {skipped}")
    print(f"Total samples: {fill:,}")

    # Trim to actual fill
    pp_cell = pp_cell[:fill]
    sp_cell = sp_cell[:fill]
    gl_bin = gl_bin[:fill]
    pp_phat = pp_phat[:fill]

    # ==========================================================================
    # Provenance: build a metadata dict shared across all saved artifacts so
    # downstream code can verify scene set, preprocessing, and origin.
    # ==========================================================================
    import hashlib
    import datetime
    scenes_used_arr = np.array(scenes_used)
    scene_list_hash = hashlib.sha256(
        ("\n".join(scenes_used)).encode("utf-8")
    ).hexdigest()
    scenes_skipped_arr = np.array(
        [f"{s}|{r}" for s, r in scenes_skipped]
    ) if scenes_skipped else np.array([], dtype=str)

    provenance = dict(
        # Counts
        n_scenes_requested=np.int32(n_scenes),
        n_scenes_used=np.int32(len(scenes_used)),
        n_scenes_skipped=np.int32(len(scenes_skipped)),
        total_samples=np.int64(fill),
        # Build identity
        version=VERSION,
        build_timestamp=datetime.datetime.now().isoformat(),
        accumulator_module="spad_utils_fixed",
        # Build params
        k_short=np.int32(args.k_short),
        n_gt_frames=np.int32(args.n_gt_frames),
        rotate_k=np.int32(args.rotate_k),
        hotpix_fix_enabled=bool(args.hotpix_fix),
        seed=np.int32(args.seed),
        mono_bin=MONO_BIN,
        images_dir=str(images_dir),
        # Reproducibility
        scene_list_hash=scene_list_hash,
        scene_list=scenes_used_arr,
        scenes_skipped_with_reason=scenes_skipped_arr,
    )
    print(f"\n[Provenance] scene_list_hash = {scene_list_hash[:16]}…")
    print(f"[Provenance] n_scenes_used    = {len(scenes_used)} / {n_scenes} requested")

    # ==========================================================================
    # Build per-pixel LUT (ragged)
    # ==========================================================================
    print("\n[Per-pixel] Sorting …")
    order = np.argsort(pp_cell, kind="stable")
    pp_cell_sorted = pp_cell[order]
    pp_vals_sorted = pp_phat[order]
    del pp_cell, order

    print("[Per-pixel] Building offsets …")
    pp_offsets = np.searchsorted(
        pp_cell_sorted, np.arange(TOTAL_CELLS_PP + 1, dtype=np.uint32), side="left"
    ).astype(np.int64)
    pp_counts = np.diff(pp_offsets).astype(np.uint32)

    out_pp = Path(args.output_dir) / "lut_per_pixel.npz"
    np.savez(
        out_pp,
        flat_values=pp_vals_sorted,
        offsets=pp_offsets,
        cell_counts=pp_counts,
        shape=np.array([H, W, N_BINS]),
        artifact_kind="per_pixel_lut",
        **provenance,
    )
    print(f"[Per-pixel] Saved → {out_pp} ({os.path.getsize(out_pp)/1e9:.2f} GB)")

    # ==========================================================================
    # Build super-pixel LUT (ragged)
    # ==========================================================================
    print("\n[Super-pixel] Sorting …")
    order = np.argsort(sp_cell, kind="stable")
    sp_cell_sorted = sp_cell[order]
    sp_vals_sorted = pp_phat[order]  # same p_hat values, different grouping
    del sp_cell, order

    sp_offsets = np.searchsorted(
        sp_cell_sorted, np.arange(TOTAL_CELLS_SP + 1, dtype=np.uint32), side="left"
    ).astype(np.int64)
    sp_counts = np.diff(sp_offsets).astype(np.uint32)

    out_sp = Path(args.output_dir) / "spad_bernoulli_lut_v2.npz"
    np.savez(
        out_sp,
        flat_values=sp_vals_sorted,
        offsets=sp_offsets,
        cell_counts=sp_counts,
        shape=np.array([N_SUPER, N_SUPER, N_BINS]),
        artifact_kind="super_pixel_lut",
        **provenance,
    )
    print(f"[Super-pixel] Saved → {out_sp} ({os.path.getsize(out_sp)/1e9:.2f} GB)")

    # ==========================================================================
    # Build global LUT (ragged)
    # ==========================================================================
    print("\n[Global] Sorting …")
    order = np.argsort(gl_bin, kind="stable")
    gl_vals_sorted = pp_phat[order]
    gl_bin_sorted = gl_bin[order]
    del order

    gl_offsets = np.searchsorted(
        gl_bin_sorted, np.arange(N_BINS + 1, dtype=np.int32), side="left"
    ).astype(np.int64)
    gl_counts = np.diff(gl_offsets).astype(np.uint32)

    out_gl = Path(args.output_dir) / "lut_global.npz"
    np.savez(
        out_gl,
        flat_values=gl_vals_sorted,
        offsets=gl_offsets,
        cell_counts=gl_counts,
        shape=np.array([N_BINS]),
        artifact_kind="global_lut",
        **provenance,
    )
    print(f"[Global] Saved → {out_gl} ({os.path.getsize(out_gl)/1e9:.2f} GB)")

    # ==========================================================================
    # Save variance decomposition stats
    # ==========================================================================
    out_var = Path(args.output_dir) / "variance_stats.npz"
    np.savez_compressed(
        out_var,
        var_sum=var_sum,       # (512, 512, 256) float64
        var_sum2=var_sum2,     # (512, 512, 256) float64
        var_count=var_count,   # (512, 512, 256) uint16
        artifact_kind="per_pixel_per_bin_variance_stats",
        **provenance,
    )
    print(f"\n[Variance] Stats saved → {out_var} ({os.path.getsize(out_var)/1e6:.1f} MB)")

    # ==========================================================================
    # Print summary stats
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"  BUILD SUMMARY  [{VERSION}]")
    print("=" * 70)

    for name, counts, total in [
        ("Global", gl_counts, N_BINS),
        ("Super-pixel", sp_counts, TOTAL_CELLS_SP),
        ("Per-pixel", pp_counts, TOTAL_CELLS_PP),
    ]:
        nz = counts[counts > 0]
        empty = int((counts == 0).sum())
        under5 = int((counts < 5).sum())
        print(f"\n  {name} ({total:,} cells):")
        print(f"    Empty: {empty:,} ({100*empty/total:.1f}%)")
        print(f"    <5 samples: {under5:,} ({100*under5/total:.1f}%)")
        if len(nz):
            print(f"    Non-empty: min={nz.min()}, median={int(np.median(nz))}, "
                  f"mean={nz.mean():.1f}, max={nz.max()}, p95={int(np.percentile(nz,95))}")
        # Flag bins with < 1000 for global
        if name == "Global":
            sparse = np.where(gl_counts < 1000)[0]
            if len(sparse):
                print(f"    ⚠ {len(sparse)} bins with <1000 samples")
            else:
                print(f"    ✓ All bins ≥1000 samples")

    # Sanity: verify global = pooled per-pixel for 5 random bins
    print("\n  Sanity check: global == pooled per-pixel?")
    rng = np.random.RandomState(args.seed)
    test_bins = rng.choice(N_BINS, 5, replace=False)
    all_match = True
    for b in sorted(test_bins):
        gl_start, gl_end = gl_offsets[b], gl_offsets[b + 1]
        # Collect all per-pixel cells for this bin
        pp_total_for_bin = 0
        for pix in range(H * W):
            cell = pix * N_BINS + b
            pp_total_for_bin += int(pp_offsets[cell + 1] - pp_offsets[cell])
        match = (pp_total_for_bin == gl_end - gl_start)
        status = "✓" if match else "✗"
        print(f"    bin {b:3d}: global={gl_end-gl_start:,}  pooled_pp={pp_total_for_bin:,}  {status}")
        if not match:
            all_match = False
    if all_match:
        print("    ✓ All 5 test bins match exactly")
    else:
        print("    ⚠ Mismatch detected — investigate")

    # Memory
    pp_mb = os.path.getsize(out_pp) / 1e6
    sp_mb = os.path.getsize(out_sp) / 1e6
    gl_mb = os.path.getsize(out_gl) / 1e6
    print(f"\n  Disk footprint:")
    print(f"    Per-pixel:   {pp_mb:.1f} MB")
    print(f"    Super-pixel: {sp_mb:.1f} MB")
    print(f"    Global:      {gl_mb:.1f} MB")
    print(f"    Var stats:   {os.path.getsize(out_var)/1e6:.1f} MB")
    print(f"    Total:       {(pp_mb+sp_mb+gl_mb+os.path.getsize(out_var)/1e6):.1f} MB")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
