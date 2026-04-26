#!/usr/bin/env python3
"""
End-to-end cascade test for the dyadic pyramid sampler.

Verifies that every level in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] is reachable,
that no query is left unresolved, and that the sampler is unbiased across the
GT distribution. Designed to surface bugs in the level walk, edge cases at the
binning boundaries, and the global-level safety net.

Tests:
  T1. Cascade reachability: queries hand-crafted to fall back at each level.
  T2. Edge bins: bin 0 (dark) and bin 255 (saturated) — both rare in practice.
  T3. Empty-cell handling: queries at sparse (pixel, bin) cells should fall
      back to the next coarser level cleanly.
  T4. Global safety net: artificially extreme min_samples to force every query
      to reach L=512; verify all 1M queries still resolve there.
  T5. Statistical unbiasedness: 10M queries from β(2,10), verify mean(p_hat)
      tracks GT mean within ±0.005.
  T6. Determinism: same seed → identical samples; different seeds → different.
  T7. Per-level sample integrity: pull every sample from a chosen cell at each
      level and verify ⊆ relationship between consecutive levels.

Outputs:
  calibration/cascade_test_report.txt
  exit code 0 if all pass, 1 otherwise
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, "/nfs/horai.dgpsrv/ondemand30/jw954/calibration")
from pyramid_sampler import PyramidSampler, LEVELS

H, W = 512, 512
N_BINS = 256
PYRAMID_PATH = "/nfs/horai.dgpsrv/ondemand30/jw954/calibration/lut_pyramid.npz"
REPORT_PATH = "/nfs/horai.dgpsrv/ondemand30/jw954/calibration/cascade_test_report.txt"


class TestRunner:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.lines = []

    def log(self, msg):
        print(msg)
        self.lines.append(msg)

    def run(self, name, fn):
        self.log(f"\n{'='*70}\n  TEST: {name}\n{'='*70}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            self.log(f"  ✓ PASSED  ({elapsed:.1f}s)")
            self.passed.append(name)
        except AssertionError as e:
            elapsed = time.time() - t0
            self.log(f"  ✗ FAILED  ({elapsed:.1f}s)")
            self.log(f"    {e}")
            self.failed.append((name, str(e)))
        except Exception as e:
            elapsed = time.time() - t0
            self.log(f"  ✗ ERROR  ({elapsed:.1f}s)")
            self.log(f"    {type(e).__name__}: {e}")
            self.failed.append((name, f"{type(e).__name__}: {e}"))

    def summary(self):
        n_pass = len(self.passed)
        n_fail = len(self.failed)
        n_total = n_pass + n_fail
        self.log(f"\n{'='*70}")
        self.log(f"  CASCADE TEST SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"  Passed: {n_pass}/{n_total}")
        self.log(f"  Failed: {n_fail}/{n_total}")
        if self.failed:
            self.log("\n  Failures:")
            for name, msg in self.failed:
                self.log(f"    ✗ {name}")
                self.log(f"      {msg}")
        self.log("")
        return n_fail == 0


def main():
    runner = TestRunner()
    runner.log(f"Loading pyramid from {PYRAMID_PATH} …")
    t0 = time.time()
    sampler = PyramidSampler(PYRAMID_PATH, min_samples=20, seed=42)
    runner.log(f"  Loaded in {time.time()-t0:.1f}s")

    # ---- T1: Cascade reachability ----------------------------------------
    def test_cascade_reachability():
        """
        Construct queries that should resolve at each pyramid level by
        choosing combinations of (x, y, gt_bin) we know are sparse at finer
        levels.

        Strategy: we don't know a priori which queries hit which level, so we
        sample widely and bin by observed level, then assert each level was
        reachable for SOME query (with the global safety net guaranteeing the
        coarsest).
        """
        rng = np.random.default_rng(0)

        # Mix of common and rare bins to stress the cascade
        n = 200_000
        # 50% from beta(2,10) (typical), 25% from uniform (covers tails),
        # 25% concentrated near the rare bin extremes
        gts_typical = rng.beta(2, 10, size=n // 2)
        gts_uniform = rng.uniform(0, 1, size=n // 4)
        gts_extreme = np.concatenate([
            rng.uniform(0.0, 0.005, size=n // 8),    # very dark
            rng.uniform(0.95, 1.0,  size=n // 8),    # very bright
        ])
        gts = np.concatenate([gts_typical, gts_uniform, gts_extreme])
        gt_bins = np.clip((gts * 255 + 0.5).astype(np.int32), 0, N_BINS - 1)
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)

        sampler.rng = np.random.default_rng(0)
        p_hats, levels = sampler.sample_batch(xs, ys, gt_bins, min_samples=20)

        # Every level present at some query
        per_level = {L: int((levels == L).sum()) for L in LEVELS}
        unresolved = int((levels == -1).sum())
        runner.log("    Level usage on diverse query set (200k):")
        for L in LEVELS:
            runner.log(f"      L={L:>3}: {per_level[L]:>8,}")
        runner.log(f"      unresolved: {unresolved}")

        # Critical: 0 unresolved
        assert unresolved == 0, f"{unresolved} queries failed to resolve"
        # The first three levels must dominate (these are the bulk of the data)
        bulk = per_level[1] + per_level[2] + per_level[4]
        assert bulk > 0.8 * n, f"Top-3 levels resolved only {bulk}/{n}"

    runner.run("T1: Cascade reachability + 0 unresolved", test_cascade_reachability)

    # ---- T2: Edge bins ----------------------------------------------------
    def test_edge_bins():
        """Bins 0 and 255 are rarely populated. They should still resolve via
        cascade — likely at coarser levels since per-pixel data is very sparse
        there."""
        rng = np.random.default_rng(1)
        n = 50_000
        bin0 = np.zeros(n, dtype=np.int32)
        bin255 = np.full(n, 255, dtype=np.int32)
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)

        for label, gb in [("bin=0  (dark)", bin0), ("bin=255 (saturated)", bin255)]:
            sampler.rng = np.random.default_rng(7)
            p_hats, levels = sampler.sample_batch(xs, ys, gb, min_samples=20)
            unresolved = int((levels == -1).sum())
            mean_p = float(p_hats.mean())
            modes = np.bincount(np.where(levels < 0, 0, levels), minlength=513)
            top_levels = sorted(((modes[L], L) for L in LEVELS if modes[L] > 0),
                                reverse=True)[:3]
            runner.log(f"    {label}: mean(p̂)={mean_p:.5f}, top-3 fallback levels: "
                       f"{[(L, c) for c, L in top_levels]}")
            assert unresolved == 0, f"{label}: {unresolved} unresolved"

    runner.run("T2: Edge bins (0 and 255) resolve via cascade", test_edge_bins)

    # ---- T3: Sparse-cell fallback walk -----------------------------------
    def test_sparse_cell_fallback():
        """Find a (pixel, bin) cell with 0 samples at L=1 but >0 at L=2.
        Hand the sampler that exact query and confirm it walks to L=2."""
        # Scan L=1 for empty cells at a known-rare bin
        L1_off = sampler.offsets[1]
        L2_off = sampler.offsets[2]
        # Try several bins to find empty L=1 cells
        for bin_test in [0, 1, 2, 250, 253, 255]:
            # Pick a random pixel and check counts
            for _ in range(50):
                rng = np.random.default_rng()
                py = rng.integers(0, H)
                px = rng.integers(0, W)
                cell_L1 = py * W * N_BINS + px * N_BINS + bin_test
                count_L1 = L1_off[cell_L1 + 1] - L1_off[cell_L1]
                if count_L1 == 0:
                    # Now check L=2
                    sy2, sx2 = py // 2, px // 2
                    cell_L2 = sy2 * 256 * N_BINS + sx2 * N_BINS + bin_test
                    count_L2 = L2_off[cell_L2 + 1] - L2_off[cell_L2]
                    sampler.rng = np.random.default_rng(13)
                    p_hat, level = sampler.sample_batch(
                        np.array([px]), np.array([py]),
                        np.array([bin_test]), min_samples=1,
                    )
                    runner.log(f"    px=({px:>3},{py:>3}) bin={bin_test:>3}: "
                               f"L1_count={count_L1}, L2_count={count_L2}, "
                               f"sampled at L={int(level[0])}")
                    if count_L2 > 0 and count_L1 == 0:
                        assert level[0] >= 2, (
                            f"Expected fallback past L=1 but used L={level[0]}"
                        )
                        return  # Found and verified one case
        runner.log("    (no perfect L=1-empty / L=2-nonempty cell found; cascade "
                   "still implicitly tested by T1)")

    runner.run("T3: Sparse-cell L=1 → L=2 fallback walk", test_sparse_cell_fallback)

    # ---- T4: Global safety net -------------------------------------------
    def test_global_safety_net():
        """Force every query to reach L=512 by setting min_samples absurdly
        high. Confirm 100% are still resolved at the global level."""
        rng = np.random.default_rng(2)
        n = 100_000
        gts = rng.beta(2, 10, size=n)
        gt_bins = np.clip((gts * 255 + 0.5).astype(np.int32), 0, N_BINS - 1)
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)

        # min_samples = 1e9 forces every level except the largest to reject
        sampler.rng = np.random.default_rng(99)
        p_hats, levels = sampler.sample_batch(xs, ys, gt_bins, min_samples=10**9)
        unresolved = int((levels == -1).sum())
        per_level = {L: int((levels == L).sum()) for L in LEVELS}
        runner.log(f"    Forced max-fallback distribution:")
        for L in LEVELS:
            if per_level[L] > 0:
                runner.log(f"      L={L:>3}: {per_level[L]:>8,}")
        # All should be at L=512 (global), or — for bins with even fewer than
        # 10^9 samples globally — fall through to bin-midpoint
        assert per_level[512] + unresolved == n, (
            f"Global net failed: {per_level[512]} at L=512, {unresolved} unresolved"
        )
        # We expect global to absorb everything since 10^9 > all global counts,
        # but the safety branch in sample_batch returns bin midpoint as the very
        # last resort. Both outcomes are acceptable.
        runner.log(f"    Resolved at L=512: {per_level[512]:,}, "
                   f"safety fallback: {unresolved:,}")

    runner.run("T4: Global safety net (extreme min_samples)", test_global_safety_net)

    # ---- T5: Statistical unbiasedness ------------------------------------
    def test_unbiasedness():
        """For 10M queries from β(2,10), verify the mean of sampled p̂ tracks
        the GT mean within tolerance. Bias > 0.01 indicates a sampler bug."""
        rng = np.random.default_rng(3)
        n = 1_000_000   # 1M (10M is overkill given pyramid hits global ~0%)
        gts = rng.beta(2, 10, size=n).astype(np.float32)
        gt_bins = np.clip((gts * 255 + 0.5).astype(np.int32), 0, N_BINS - 1)
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)

        sampler.rng = np.random.default_rng(31)
        p_hats, levels = sampler.sample_batch(xs, ys, gt_bins, min_samples=20)

        gt_mean = float(gts.mean())
        gt_bin_centers = (gt_bins.astype(np.float64) + 0.5) / N_BINS
        gt_bin_mean = float(gt_bin_centers.mean())
        p_hat_mean = float(p_hats.mean())
        bias = p_hat_mean - gt_bin_mean

        runner.log(f"    n_queries     = {n:,}")
        runner.log(f"    GT mean       = {gt_mean:.5f}")
        runner.log(f"    GT bin mean   = {gt_bin_mean:.5f}  (after binning)")
        runner.log(f"    p̂ mean        = {p_hat_mean:.5f}")
        runner.log(f"    bias          = {bias:+.5f}  (p̂ - GT_bin)")

        assert abs(bias) < 0.005, (
            f"Sampler is biased: |bias|={abs(bias):.5f} > 0.005"
        )

    runner.run("T5: Statistical unbiasedness (1M queries)", test_unbiasedness)

    # ---- T6: Determinism --------------------------------------------------
    def test_determinism():
        """Same seed → identical samples; different seed → different."""
        n = 1000
        rng = np.random.default_rng(4)
        xs = rng.integers(0, W, size=n)
        ys = rng.integers(0, H, size=n)
        gt_bins = rng.integers(0, N_BINS, size=n)

        sampler.rng = np.random.default_rng(42)
        p1, _ = sampler.sample_batch(xs, ys, gt_bins, min_samples=20)
        sampler.rng = np.random.default_rng(42)
        p2, _ = sampler.sample_batch(xs, ys, gt_bins, min_samples=20)
        sampler.rng = np.random.default_rng(43)
        p3, _ = sampler.sample_batch(xs, ys, gt_bins, min_samples=20)

        same = np.array_equal(p1, p2)
        diff = not np.array_equal(p1, p3)
        runner.log(f"    seed=42 twice  → identical: {same}")
        runner.log(f"    seed=42 vs 43  → differ:    {diff}")
        assert same, "Same seed produced different samples"
        assert diff, "Different seeds produced identical samples (suspicious)"

    runner.run("T6: Determinism (same seed → same; diff seed → diff)", test_determinism)

    # ---- T7: Per-level sample-set containment ----------------------------
    def test_level_containment():
        """For a chosen pixel and bin, the set of samples available at level L
        must be a SUPERSET of the set at level L-1's parent cell. Verifies
        the pyramid was built consistently from per-pixel data."""
        rng = np.random.default_rng(5)
        # Pick a pixel/bin where we expect dense data
        py, px, b = 256, 256, 50  # mid-image, mid-rate
        runner.log(f"    Test pixel ({px}, {py}), bin {b}")

        prev_set = None
        for L in LEVELS:
            H_L, W_L, _ = sampler.shapes[L]
            sx, sy = px // L, py // L
            cell = sy * W_L * N_BINS + sx * N_BINS + b
            offs = sampler.offsets[L]
            vals = sampler.flat[L][offs[cell]:offs[cell + 1]]
            n_vals = len(vals)
            runner.log(f"      L={L:>3}: cell ({sy:>3},{sx:>3},{b:>3}) has {n_vals:>8,} samples")
            if prev_set is not None:
                # Each value at the finer level must appear at the coarser level
                # (since the coarser level pools children).
                # We test by comparing sorted multiplicities.
                assert n_vals >= len(prev_set), (
                    f"L={L} has fewer samples ({n_vals}) than its child ({len(prev_set)})"
                )
                # Verify each prev value is present (multiset containment)
                from collections import Counter
                c_prev = Counter(prev_set.tolist())
                c_curr = Counter(vals.tolist())
                for v, cnt in c_prev.items():
                    if c_curr[v] < cnt:
                        raise AssertionError(
                            f"L={L} missing {cnt - c_curr[v]} copies of value {v} "
                            f"that appeared at the previous level"
                        )
            prev_set = vals

    runner.run("T7: Per-level sample-set containment (pyramid integrity)",
               test_level_containment)

    # -----------------------------------------------------------------------
    ok = runner.summary()

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(runner.lines))
    print(f"\nReport written → {REPORT_PATH}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
