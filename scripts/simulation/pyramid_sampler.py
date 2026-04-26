#!/usr/bin/env python3
"""
Pyramid-fallback sampler + 1M-query usage validation
=====================================================
Version: v1-pyramid

Walks the dyadic LUT pyramid coarsest-to-finest (smallest pool first), then
finest-to-coarsest fallback at sampler time:

    For each query (x, y, gt_bin):
      for L in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
          sx, sy = x // L, y // L
          cell = pyramid[level_L][sy, sx, gt_bin]
          if len(cell) >= min_samples:
              return random_choice(cell), L
      raise (should never reach: level_512 always has thousands)

Validates by running 1M queries from a realistic GT distribution (β(2,10))
and reporting the fraction resolved at each level.

Saves: calibration/pyramid_usage_stats.npz
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

H, W = 512, 512
N_BINS = 256
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


class PyramidSampler:
    """Multi-resolution dyadic pyramid sampler with fallback."""

    def __init__(self, pyramid_path, min_samples=20, seed=42):
        print(f"Loading pyramid from {pyramid_path} …")
        t0 = time.time()
        npz = np.load(pyramid_path)
        self.levels = list(npz["levels"])
        self.min_samples = min_samples
        self.rng = np.random.default_rng(seed)

        # Pre-load every level into memory for fast access
        # Total ~30 GB — fits in our 113 GB RAM
        self.flat = {}
        self.offsets = {}
        self.shapes = {}
        for L in self.levels:
            self.flat[L] = npz[f"level_{L}_flat_values"]
            self.offsets[L] = npz[f"level_{L}_offsets"]
            self.shapes[L] = tuple(int(s) for s in npz[f"level_{L}_shape"])
        print(f"  Loaded {len(self.levels)} levels in {time.time()-t0:.1f}s")
        for L in self.levels:
            print(f"    L={L:>3}: shape={self.shapes[L]}, "
                  f"flat={len(self.flat[L]):,}, "
                  f"offsets={len(self.offsets[L]):,}")

    def sample_batch(self, xs, ys, gt_bins, min_samples=None):
        """Vectorised batch sampling.  Returns (p_hats, levels_used)."""
        if min_samples is None:
            min_samples = self.min_samples
        n = len(xs)
        p_hats = np.empty(n, dtype=np.float32)
        levels_used = np.full(n, -1, dtype=np.int16)
        resolved = np.zeros(n, dtype=bool)

        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        gt_bins = np.asarray(gt_bins, dtype=np.int64)

        for L in self.levels:
            if resolved.all():
                break
            idx_remaining = np.where(~resolved)[0]
            if len(idx_remaining) == 0:
                break

            x_r = xs[idx_remaining]
            y_r = ys[idx_remaining]
            b_r = gt_bins[idx_remaining]

            H_L, W_L, _ = self.shapes[L]
            sx = x_r // L
            sy = y_r // L
            cells = sy * (W_L * N_BINS) + sx * N_BINS + b_r

            offs = self.offsets[L]
            starts = offs[cells]
            ends = offs[cells + 1]
            counts = ends - starts

            ok = counts >= min_samples
            if not ok.any():
                continue

            # Sample within each ok cell
            ok_idx = idx_remaining[ok]
            ok_starts = starts[ok]
            ok_counts = counts[ok]
            rand_offs = self.rng.integers(0, ok_counts).astype(np.int64)
            sampled = self.flat[L][ok_starts + rand_offs]

            p_hats[ok_idx] = sampled
            levels_used[ok_idx] = L
            resolved[ok_idx] = True

        # Anything still unresolved? Use the global level (L=512) — should always
        # have samples. If genuinely empty, return GT bin midpoint as last resort.
        if not resolved.all():
            unresolved_idx = np.where(~resolved)[0]
            print(f"  [WARN] {len(unresolved_idx)} queries reached the global level "
                  f"and still had insufficient samples — falling back to bin midpoint.")
            for i in unresolved_idx:
                p_hats[i] = (gt_bins[i] + 0.5) / N_BINS
                levels_used[i] = 512
        return p_hats, levels_used

    def fallback_level_map(self, gt_bin, min_samples=None):
        """Return a (H, W) int array where each entry is the coarsest level
        index needed to satisfy `min_samples` for the given GT bin.

        Used for the diagnostic fallback heat-map.
        """
        if min_samples is None:
            min_samples = self.min_samples
        out = np.full((H, W), -1, dtype=np.int16)

        # Build (y, x) → cell mapping per level and check counts
        for L in self.levels:
            if (out != -1).all():
                break
            H_L, W_L, _ = self.shapes[L]
            offs = self.offsets[L]

            # For every pixel that hasn't been resolved yet
            yet_to_resolve = (out == -1)
            yy, xx = np.indices((H, W))
            sx_all = (xx // L).astype(np.int64)
            sy_all = (yy // L).astype(np.int64)
            cells = sy_all * (W_L * N_BINS) + sx_all * N_BINS + gt_bin
            counts = offs[cells + 1] - offs[cells]
            ok_mask = (counts >= min_samples) & yet_to_resolve
            out[ok_mask] = L
        return out


# ---------------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pyramid", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/lut_pyramid.npz")
    p.add_argument("--n-queries", type=int, default=1_000_000)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-stats", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration/pyramid_usage_stats.npz")
    p.add_argument("--gt-distribution", choices=["beta_2_10", "uniform"], default="beta_2_10")
    return p.parse_args()


def main():
    args = parse_args()
    sampler = PyramidSampler(args.pyramid, min_samples=args.min_samples, seed=args.seed)

    # Generate realistic queries
    print(f"\nGenerating {args.n_queries:,} random queries …")
    rng = np.random.default_rng(args.seed)
    if args.gt_distribution == "beta_2_10":
        gts = rng.beta(2, 10, size=args.n_queries).astype(np.float32)
    else:
        gts = rng.uniform(0, 1, size=args.n_queries).astype(np.float32)
    gt_bins = np.clip((gts * 255 + 0.5).astype(np.int32), 0, N_BINS - 1)
    xs = rng.integers(0, W, size=args.n_queries)
    ys = rng.integers(0, H, size=args.n_queries)
    print(f"  GT distribution: {args.gt_distribution}, mean={gts.mean():.3f}")

    # Run
    print(f"\nRunning {args.n_queries:,} pyramid-sampled queries …")
    t0 = time.time()
    p_hats, levels_used = sampler.sample_batch(xs, ys, gt_bins,
                                                min_samples=args.min_samples)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({args.n_queries/elapsed:.0f} queries/s)")

    # Report
    print()
    print("=" * 60)
    print(f"  Pyramid usage at min_samples={args.min_samples}")
    print("=" * 60)
    print(f"{'Level':>6}  {'Pool':>5}  {'Resolved':>10}  {'Fraction':>9}")
    print("-" * 60)
    counts_per_level = {}
    for L in LEVELS:
        c = int((levels_used == L).sum())
        f = 100.0 * c / args.n_queries
        counts_per_level[L] = c
        print(f"  L={L:>3}  {L:>5}  {c:>10,}  {f:>8.3f}%")
    unresolved = int((levels_used == -1).sum())
    print(f"  ----  -----  {unresolved:>10,}  {100*unresolved/args.n_queries:>8.3f}%  (unresolved!)")
    print()
    print(f"Mean p_hat sampled: {float(p_hats.mean()):.4f}  (GT mean: {float(gts.mean()):.4f})")

    # Save
    np.savez(args.output_stats,
             levels=np.array(LEVELS),
             counts_per_level=np.array([counts_per_level[L] for L in LEVELS]),
             fractions=np.array([counts_per_level[L]/args.n_queries for L in LEVELS]),
             n_queries=args.n_queries,
             min_samples=args.min_samples,
             gt_distribution=args.gt_distribution,
             unresolved=unresolved,
             p_hats_mean=float(p_hats.mean()),
             gts_mean=float(gts.mean()))
    print(f"\nStats saved → {args.output_stats}")


if __name__ == "__main__":
    main()
