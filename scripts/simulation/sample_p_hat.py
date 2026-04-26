#!/usr/bin/env python3
"""
Hierarchical p̂ sampler with three fallback modes
==================================================
Version: v1

Provides sample_p_hat(x, y, gt_intensity, hierarchy) that draws from the
non-parametric Bernoulli noise LUT with configurable fallback hierarchy:

    'per_pixel_first':   try per-pixel → super-pixel → global
    'super_pixel_first': try super-pixel → global  (skip per-pixel)
    'global_only':       always global

Each query returns (p_hat, level_used) where level_used ∈ {0='per_pixel',
1='super_pixel', 2='global'}.

Also validates the hierarchy by sampling 1M random queries and reporting
the fraction from each level.
"""

import numpy as np
from pathlib import Path

H, W = 512, 512
N_BINS = 256
N_SUPER = 64
SUPER_PX = 8

MIN_SAMPLES_PP = 20   # per-pixel fallback threshold
MIN_SAMPLES_SP = 5    # super-pixel fallback threshold


class SPADNoiseSampler:
    """Non-parametric SPAD Bernoulli noise sampler with fallback hierarchy."""

    def __init__(self, cal_dir="calibration",
                 pp_file="lut_per_pixel.npz",
                 sp_file="spad_bernoulli_lut_v2.npz",
                 gl_file="lut_global.npz",
                 min_pp=MIN_SAMPLES_PP,
                 min_sp=MIN_SAMPLES_SP):
        cal = Path(cal_dir)
        self.min_pp = min_pp
        self.min_sp = min_sp
        self.rng = np.random.RandomState(42)

        # Load per-pixel
        d = np.load(cal / pp_file)
        self.pp_vals = d["flat_values"]
        self.pp_off = d["offsets"]
        self.pp_shape = tuple(d["shape"])  # (512, 512, 256)

        # Load super-pixel
        d = np.load(cal / sp_file)
        self.sp_vals = d["flat_values"]
        self.sp_off = d["offsets"]

        # Load global
        d = np.load(cal / gl_file)
        self.gl_vals = d["flat_values"]
        self.gl_off = d["offsets"]

        print(f"Loaded LUTs: pp={len(self.pp_vals):,}, "
              f"sp={len(self.sp_vals):,}, gl={len(self.gl_vals):,}")

    def _sample_from_cell(self, vals, offsets, cell_idx):
        """Sample one value from a ragged cell. Returns (value, count)."""
        start = offsets[cell_idx]
        end = offsets[cell_idx + 1]
        count = end - start
        if count == 0:
            return None, 0
        idx = self.rng.randint(0, count)
        return float(vals[start + idx]), int(count)

    def sample(self, x, y, gt_intensity, hierarchy="per_pixel_first"):
        """
        Sample p̂ from the LUT hierarchy.

        Args:
            x, y: pixel coordinates (0-indexed, in rotated image space)
            gt_intensity: ground-truth intensity in [0, 1]
            hierarchy: 'per_pixel_first', 'super_pixel_first', or 'global_only'

        Returns:
            (p_hat, level_used)
            level_used: 0=per_pixel, 1=super_pixel, 2=global
        """
        b = min(int(gt_intensity * 255 + 0.5), 255)

        if hierarchy == "per_pixel_first":
            # Try per-pixel
            pp_cell = y * W * N_BINS + x * N_BINS + b
            val, count = self._sample_from_cell(self.pp_vals, self.pp_off, pp_cell)
            if count >= self.min_pp:
                return val, 0

            # Fallback to super-pixel
            sy, sx = y // SUPER_PX, x // SUPER_PX
            sp_cell = sy * N_SUPER * N_BINS + sx * N_BINS + b
            val, count = self._sample_from_cell(self.sp_vals, self.sp_off, sp_cell)
            if count >= self.min_sp:
                return val, 1

            # Fallback to global
            val, count = self._sample_from_cell(self.gl_vals, self.gl_off, b)
            if val is not None:
                return val, 2
            return gt_intensity, 2  # ultimate fallback: use GT as-is

        elif hierarchy == "super_pixel_first":
            sy, sx = y // SUPER_PX, x // SUPER_PX
            sp_cell = sy * N_SUPER * N_BINS + sx * N_BINS + b
            val, count = self._sample_from_cell(self.sp_vals, self.sp_off, sp_cell)
            if count >= self.min_sp:
                return val, 1

            val, count = self._sample_from_cell(self.gl_vals, self.gl_off, b)
            if val is not None:
                return val, 2
            return gt_intensity, 2

        elif hierarchy == "global_only":
            val, count = self._sample_from_cell(self.gl_vals, self.gl_off, b)
            if val is not None:
                return val, 2
            return gt_intensity, 2

        else:
            raise ValueError(f"Unknown hierarchy: {hierarchy}")

    def sample_batch(self, xs, ys, gt_intensities, hierarchy="per_pixel_first"):
        """Vectorized batch sampling. Returns (p_hats, levels)."""
        n = len(xs)
        p_hats = np.empty(n, dtype=np.float32)
        levels = np.empty(n, dtype=np.int8)
        for i in range(n):
            p_hats[i], levels[i] = self.sample(
                int(xs[i]), int(ys[i]), float(gt_intensities[i]), hierarchy
            )
        return p_hats, levels


def validate_hierarchy(sampler, n_queries=1_000_000, seed=42):
    """Sample random queries and report level usage per hierarchy mode."""
    rng = np.random.RandomState(seed)

    # Generate realistic GT distribution (use beta distribution centered around 0.1)
    gt_intensities = rng.beta(2, 10, size=n_queries).astype(np.float32)
    xs = rng.randint(0, W, size=n_queries)
    ys = rng.randint(0, H, size=n_queries)

    print(f"\nValidating hierarchy with {n_queries:,} queries …")
    print(f"  GT distribution: beta(2,10), mean={gt_intensities.mean():.3f}")
    level_names = {0: "per_pixel", 1: "super_pixel", 2: "global"}

    for mode in ["per_pixel_first", "super_pixel_first", "global_only"]:
        sampler.rng = np.random.RandomState(seed)
        _, levels = sampler.sample_batch(xs, ys, gt_intensities, hierarchy=mode)

        print(f"\n  '{mode}':")
        for lev in [0, 1, 2]:
            frac = (levels == lev).sum() / n_queries
            print(f"    {level_names[lev]:>12s}: {100*frac:6.2f}%  "
                  f"({(levels == lev).sum():,})")


def sanity_check(sampler, p_target=0.3, n_frames=1000):
    """
    Synthetic test: uniform GT = p_target across sensor.
    Verify output has spatial mean ≈ p_target.
    """
    print(f"\n[Sanity] Uniform GT = {p_target} across full sensor …")
    gt_map = np.full((H, W), p_target, dtype=np.float32)

    # Sample p_hat for every pixel using each hierarchy
    for mode in ["per_pixel_first", "super_pixel_first", "global_only"]:
        p_hat_map = np.empty((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                p_hat_map[y, x], _ = sampler.sample(x, y, p_target, mode)

        mean_p = p_hat_map.mean()
        std_p = p_hat_map.std()
        # Simulate binary frames
        frames = np.random.binomial(1, np.clip(p_hat_map, 0, 1),
                                    size=(n_frames, H, W))
        obs_rate = frames.mean()
        print(f"  {mode:>20s}: mean(p̂)={mean_p:.4f}, std(p̂)={std_p:.4f}, "
              f"observed_rate={obs_rate:.4f}  "
              f"({'✓' if abs(obs_rate - p_target) < 0.02 else '⚠'})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cal-dir", default="/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
    p.add_argument("--validate", action="store_true", default=True)
    p.add_argument("--sanity", action="store_true", default=True)
    p.add_argument("--n-queries", type=int, default=100_000,
                   help="Queries for validation (default 100K; 1M is slow)")
    args = p.parse_args()

    sampler = SPADNoiseSampler(cal_dir=args.cal_dir)

    if args.validate:
        validate_hierarchy(sampler, n_queries=args.n_queries)

    if args.sanity:
        sanity_check(sampler)
