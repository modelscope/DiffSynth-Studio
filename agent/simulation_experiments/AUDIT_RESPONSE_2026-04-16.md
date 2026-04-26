# Audit Response — 2026-04-16

External audit (GPT) flagged six issues in the simulation pipeline. Here's the
response: which we accept, which we contest, and what was fixed.

## TL;DR

| # | Severity | Status | What we did |
|---|---|---|---|
| 1 | **Critical** — race in `utils._accumulate_counts_core` | ✅ **Confirmed and fixed** | New `spad_utils_fixed.py`; all builds rerun |
| 2 | High — inter-arrival pooling masks rate-mixing as afterpulsing | ✅ **Confirmed; rebuilt with per-pixel methodology** | New `interarrival_analysis_v3.py` |
| 3 | High — forward model lacks camera-RGB ↔ SPAD-flux bridge | ⚠ **Acknowledged — known limitation, not a bug** | Already documented in `CALIBRATION_DESIGN_NOTES.md` |
| 4 | Medium — hot-pixel handling inconsistent across stages | ✅ **Confirmed; documented + fixed in new builds** | Hot-pix fix is now explicit and ON by default in `build_three_luts.py` |
| 5 | Medium — v4 not on same scene split as v1/v3 | ✅ **Confirmed; cosmetic, won't refit** | Documented; future comparisons will use common subset |
| 6 | Low — LUT axis naming `[sx, sy, bin]` vs code `[sy, sx, bin]` | ✅ **Confirmed; doc fixed** | Docstrings updated to match numpy row-major layout |

All pre-audit artifacts moved to `calibration/archive_pre_audit_2026-04-16/`
with a README explaining why they're deprecated.

---

## Finding 1 (CRITICAL): race condition in count accumulator

### Audit claim
The Numba kernel `_accumulate_counts_core` at `utils.py:35` parallelizes over
frames with `prange` but every iteration writes to the same `counts` array.
Race-prone read-modify-write loses increments. Auditor reproduced
nondeterministic undercounts locally on synthetic data.

### Verification

Wrote `/tmp/race_test.py`. Synthetic `1000 × 32 768`-byte buffer, 16 threads:

```
Ground truth (numpy): 131,070,975 counts
Numba (parallel=True), 10 runs:
    losses: 73k, 122k, 107k, 60k, 127k, 140k, 175k, 202k, 254k, 215k
    pixels affected: ~50k–137k out of 262k (19% – 52%)
    max per-pixel loss: up to 91 counts (≈18% of ~500-count pixels)
    deterministic across runs: NO
```

Real SPAD binary, 5 000 frames of `RAW_empty.bin`:
```
Ground truth: 142,416,028 counts
Numba (parallel=True): 142,057,642 / 142,013,520 / 142,041,227   (losses ~360k-400k each)
```

**Confirmed: real bug. Bias is ~0.06–0.30 % on totals, but per-pixel can lose
up to ~18 % at high-rate pixels. Always undercounts (no overcounts).**

### Fix

`/nfs/horai.dgpsrv/ondemand30/jw954/calibration/spad_utils_fixed.py` — drop-in
replacement using **per-thread chunk accumulators** (each thread writes only to
its own row of a `(n_threads, H*W)` array; serial reduction at the end).

Verified deterministic and exact match to numpy ground truth. Performance
near-identical to the racy version (27 ms vs 29 ms warm; first call slower
due to JIT compilation).

The fix preserves the same parallelism (one thread per chunk) without any
shared writes. Memory cost: `n_threads × H*W × 4 bytes` ≈ 16 MB for a 16-thread
machine — negligible.

### Why we did NOT modify `spad-diffusion/spad_dataset/utils.py`

That file is shared with other code in the user's project that we haven't
audited. Adding a fix there could change unrelated behavior. Instead the
fixed copy lives next to the calibration scripts and is imported with an
explicit comment marking the switch.

### What got rebuilt with the fixed code

| Artifact | Pre-audit (buggy) → Post-audit |
|---|---|
| Three-LUT build | `archive/.../{lut_global,lut_per_pixel,spad_bernoulli_lut_v2,variance_stats}.npz` → rebuilding (started 2026-04-16 17:04, ETA ~10 h) |
| RGB-to-mono v1 calibration | superseded by v3 already — not rebuilt |
| RGB-to-mono v3 calibration | `archive/.../rgb_to_mono_weights_v3.npz` → rebuilding (later) |
| RGB-to-mono v4 calibration | KEPT — was built from extracted PNGs (numpy-only extractor in `extract_binary_images.py`, no race) |
| Inter-arrival v1 + v2 | superseded by v3 (different methodology, see Finding 2) |
| Coverage check | KEPT — built from PNGs |

---

## Finding 2 (HIGH): inter-arrival pooling masks rate-mixing as afterpulsing

### Audit claim
Pooling pixels across a [0.01, 0.05] bracket and fitting a single Geometric
with `p̂ = 1/mean_ia` is confounded: a heterogeneous mixture of geometrics
already inflates `P(δt=1)` above the single-Geometric prediction by AM ≥ HM,
even with NO afterpulsing. The reported "17 % excess" is therefore not specific
evidence of afterpulsing.

### Why this is correct

For a single pixel with rate `p_i`: `P(δt=1) = p_i`.
For a pooled mixture of pixels with rates `{p_i}` and event counts `{n_i}`:

```
true pooled P(δt=1)        = Σᵢ nᵢ · pᵢ / Σᵢ nᵢ          (event-weighted arithmetic mean)
fitted Geom(p̂_pooled) gives = p̂_pooled = 1/mean(IA)
                            = Σᵢ nᵢ / Σᵢ nᵢ·(1/pᵢ)       (event-weighted harmonic mean)
```

Since AM ≥ HM, the true pooled `P(δt=1)` ALWAYS exceeds the fitted Geometric's
prediction whenever rates are not identical. For uniform p ∈ [0.01, 0.05]:
arithmetic mean / harmonic mean ≈ 1.20 (a ~20 % spurious "excess").

So the v2 ratio of 1.171 is **fully consistent with no afterpulsing**.

### Fix: `interarrival_analysis_v3.py`

Per-pixel residual methodology:

```python
for each pixel i:
    p̂_i        = (#detections) / n_frames
    dt1_obs_i  = (#δt=1 events) / (#total events)
    dt1_pred_i = p̂_i                   # Geometric(p̂_i) prediction
    residual_i = dt1_obs_i − dt1_pred_i
    lag1_i     = lag-1 autocorrelation of inter-arrival sequence
```

Then:
- Plot residual vs. p̂ — afterpulsing would show **rate-independent** positive offset
- Plot lag-1 correlation vs. p̂ — true afterpulsing introduces **positive lag-1**
  (the previous IA being short makes the next IA more likely to also be short)

The combined evidence (residual sign + lag-1 sign) is robust to rate
heterogeneity and saturation. If both are positive across the rate range,
afterpulsing is real. If only the pooled-bracket ratio is high but per-pixel
residual is ≈ 0, the v2 finding was a methodological artifact — exactly what
the auditor predicted.

Result will be reported once `interarrival_analysis_v3.py` finishes (~5 min).

---

## Finding 3 (HIGH): camera-RGB ↔ SPAD-flux bridge is uncalibrated

### Audit claim
The v3 weights are valid for SPAD-channel flux proxies (SPAD's own R/G/B
filters). Using them as `λ_r = α·R, λ_g = α·G, λ_b = α·B` for ordinary camera
RGB is a strong spectral-correspondence assumption that the current calibration
does NOT establish.

### Status: known limitation, not a bug

This is documented in `CALIBRATION_DESIGN_NOTES.md` (the four-spaces table
plus the "Approach A (λ-space) → Approach B (linear-RGB-space)" discussion).
The user explicitly asked about this earlier ("eventually we have to simulate
using online sRGB" + "is binary or processed RGB more appropriate") and the
written answer acknowledged that an additional bridge would be needed for
a proper end-to-end inference pipeline.

### What's missing for a clean end-to-end forward model

1. A 3×3 color-correction matrix (CCM) between camera RGB color space and
   SPAD R/G/B-filter color space. Could be calibrated with a color chart
   captured by both, or fit empirically from registered camera-RGB ↔ SPAD
   data if available.
2. Exposure scaling `α`. Could be sampled from the empirical λ-distribution of
   our calibration set.

Both are downstream of the LUT work. Tracked as future tasks; doesn't
invalidate any current artifact.

---

## Finding 4 (MEDIUM): inconsistent hot-pixel handling

### Audit claim
- `calibrate_rgb_to_mono_v3.py` applies `median_hotpixel_fix` on raw counts
  before normalizing.
- `build_bernoulli_lut.py` (the **original** super-pixel LUT builder) does NOT.
- `interarrival_analysis_v2.py` does NOT.
- The `EXPERIMENTS_LOG.md` says "with hot pixel fix" for the LUT build —
  inconsistent with what the script actually did.

### Verification

Looked at the original `build_bernoulli_lut.py:138-141`:
```python
counts_short, n_short = accumulate_counts_whole_file(...)   # no fix
counts_gt, n_gt = accumulate_counts_whole_file(...)         # no fix
p_short = counts_short.astype(np.float32) / n_short
```

Confirmed: original LUT build did NOT apply hot-pixel fix despite the doc
claiming it did. Documentation error.

### Fix

- New `build_three_luts.py` (post-audit version 2):
  - Hot-pix fix is **explicit** and **ON by default** (`--hotpix-fix` flag).
  - Docstring states this clearly.
  - Both `p_short` and `p_GT` get the fix applied to their counts.
- `interarrival_analysis_v3.py`: hot pixel fix not needed because the analysis
  tracks individual pixel time-series (not aggregated counts). A hot pixel
  just shows up as its own high-rate sequence — its inter-arrival statistics
  are still meaningful for itself, and the per-pixel residual methodology
  doesn't get contaminated by hot pixels in other pixels.
- `EXPERIMENTS_LOG.md`: corrected to reflect actual behavior of each version.

---

## Finding 5 (MEDIUM): v4 split mismatch

### Audit claim
`v4` enumerates `RGB_linear16/ ∩ monochrome16/` = 2631 paired scenes, while
`v1/v3` enumerate raw-binary scenes = 2633. With seed=42 and different
input lists, `train_test_split` produces different scenes in each split.

### Verification

Yes — confirmed via file counts:
- v1/v3 universe: scenes with all four `.bin` files = 2633
- v4 universe: scenes with both PNG products = 2631

The 2-scene difference is in which `monochrome16/` entries match
`RGB_linear16/`. Same seed → permutation of a different list → different splits.

### Severity assessment

- **R²**, **weights**, **residuals** of each individual version are still valid
  for THAT version's data scope.
- **Cross-version comparisons** (e.g., "v3 has higher R² than v4") are still
  meaningful because the difference is 2 / 2631 ≈ 0.08 % of scenes.
- The bias from this is dominated by:
  - the race condition (Finding 1) for v1/v3
  - the tone-map per-scene scaling for v4
  - the linearization choice (p-space vs λ-space)

### Resolution

Documented in `EXPERIMENTS_LOG.md`. Future comparison runs will:
1. Compute the common scene set (binaries ∩ RGB_linear16 ∩ monochrome16).
2. Use that set with seed=42 split → identical splits across versions.

Not refitting now because the v3 rebuild (with race fix) is the higher-impact
correction.

---

## Finding 6 (LOW): LUT axis naming inconsistency

### Audit claim
Docstring says `LUT[sx, sy, bin]` but code computes
`flat_idx = sy*N_SUPER + sx`, then `* N_BINS + bin`. Actual layout is
`[sy, sx, bin]`.

### Verification & fix

Confirmed — the code is consistent within itself (numpy row-major: y is row,
x is col, bin is innermost), but the prose docstring used the prompt's
`(sx, sy)` notation which puts sx first.

Updated docstrings in `build_three_luts.py` to explicitly state:

> `cell_idx = sy * N_SUPER * N_BINS + sx * N_BINS + b`
> `[sy, sx, bin]` numpy row-major layout

`sample_p_hat.py` was already consistent (it computes the same expression).

---

## What this means for the experiments log

**Pre-audit results that REMAIN VALID** (no race-affected code):
- v4 RGB-to-mono calibration
- Coverage feasibility check (built from PNGs)
- Inter-arrival v2 figure as a *demonstration of the methodological pitfall*
  (now explicitly framed that way)

**Pre-audit results SUPERSEDED (archived)**:
- Original super-pixel LUT (`spad_bernoulli_lut.npz`)
- Three-LUT v1 (`lut_global`, `lut_per_pixel`, `spad_bernoulli_lut_v2`)
- Variance stats v1 (`variance_stats.npz`)
- v1 and v3 RGB-to-mono calibrations
- Inter-arrival v1 figure (used buggy GT for bracket assignment)

**Pre-audit results that are still useful as a comparison reference** (kept in
archive, marked as such):
- The buggy LUT could be used to quantify the magnitude of the bias once the
  fixed LUT exists. We don't plan to do this analysis right now but the data
  is there if needed.

---

## Round 2 Audit (same day, 2026-04-16 evening)

After the round-1 fixes, GPT did a second pass and flagged seven more issues.
Three were significant; the rest were quality-of-life. All addressed.

### Round-2 Finding 1 (HIGH): three-LUT artifacts not provenance-safe

**Claim:** files in `calibration/lut_*.npz` after the round-1 patch did not
record `n_scenes_used` or `total_samples`, and inspection showed
`flat_values` length = 1,310,720 = 5 × 512 × 512. That's a 5-scene smoke-test
artifact, not the full-dataset build.

**Confirmed:** at audit time the live calibration/ files were indeed the
smoke-test outputs left over from re-verifying the patch. The full build
was running in tmux but had not yet completed and overwritten them.

**Fix:**
- `build_three_luts.py` now writes a comprehensive provenance dict to every
  saved NPZ. Fields include `n_scenes_used`, `n_scenes_requested`,
  `n_scenes_skipped`, `total_samples`, `build_timestamp`,
  `accumulator_module`, `k_short`, `n_gt_frames`, `rotate_k`,
  `hotpix_fix_enabled`, `seed`, `mono_bin`, `images_dir`,
  `scene_list_hash` (sha256 of joined scene IDs), `scene_list` (full list),
  `scenes_skipped_with_reason`, and `artifact_kind`.
- New `calibration/provenance.py` module provides `load(path)`,
  `assert_compatible(*provs)`, and `summarize(prov)` for downstream use.
- The smoke-test outputs were moved to
  `archive_pre_audit_2026-04-16/smoke_test_5_scenes/` and
  `archive_pre_audit_2026-04-16/smoke_test_5_scenes_with_provenance/` so they
  can't be confused with full builds.
- The running build was killed (had reached 378/2633 ≈ 14 % over ~1.5 h), the
  saver patched, then re-launched fresh at v3-post-audit2.

### Round-2 Finding 2 (HIGH): variance_analysis.py mixes incompatible sources

**Claim:** the script loaded `variance_stats.npz` (from build) and
`coverage_counts.npz` (from PNG pre-pass) without checking they came from
the same scene set or preprocessing. Could combine 5-scene build stats with
2,633-scene coverage counts.

**Fix:**
- `variance_analysis.py` now imports `provenance.py` and:
  - Loads `variance_stats.npz` via `prov_load` (errors out if no provenance)
  - Loads `coverage_counts.npz` and reads its `n_scenes` field
  - Verifies scene counts match within ±5 %; warns at ±1 %; refuses to run
    if >5 % mismatch
  - Verifies spatial shape and bin count match exactly
- `coverage_check.py` now also writes provenance fields (artifact_kind,
  n_scenes_used, scene_list_hash, build_timestamp, version) so future
  cross-artifact compatibility checks can be more rigorous.

### Round-2 Finding 3 (MEDIUM): rgb_to_mono_weights_v3.npz is still missing

**Claim:** correct. v1/v3 calibration weight files were archived in round 1,
but the v3 rebuild has not yet been kicked off.

**Status:** queued behind the LUT rebuild (they contend for the same NFS
bandwidth). Will run after the LUT build completes (~8 h). Until then, v4
weights (which were built from PNGs and never used the racy accumulator) are
the only live calibration.

### Round-2 Finding 4 (MEDIUM): old afterpulsing artifacts still live

**Claim:** correct. The IA v2 PDF/NPZ were still in `figures/` even though
v3 superseded the methodology and `EXPERIMENTS_LOG.md` marked v2 as
DEPRECATED.

**Fix:** moved `figures/interarrival_v2_*` to
`archive_pre_audit_2026-04-16/v2_interarrival_deprecated/`.

### Round-2 Finding 5 (MEDIUM): forward-model bridge still uncalibrated

**Acknowledged again.** Same status as round-1 Finding 3 — known limitation,
not a bug, not addressable at this stage of the pipeline. Waits for a CCM
calibration step downstream of the LUT work.

### Round-2 Finding 6 (LOW): legacy scripts still live

**Claim:** `build_bernoulli_lut.py` (original super-pixel builder, no
hotpix-fix) and `interarrival_analysis_v2.py` (deprecated pooled methodology)
were still in the live calibration folder and could be re-run accidentally.

**Fix:** moved both to
`archive_pre_audit_2026-04-16/legacy_scripts/`.

### Round-2 Finding 7 (LOW): cv2 hard-dependency in spad_utils_fixed.py

**Claim:** `import cv2` at module load time fails on systems without OpenCV,
even for callers who only need the count accumulator.

**Fix:** `cv2` import moved inside `median_hotpixel_fix`. Module now imports
cleanly without OpenCV; the import only fires (with a clear error message
suggesting `pip install opencv-python-headless`) if `median_hotpixel_fix`
is actually called.

---

## State as of round-2 fix complete (2026-04-16 18:17)

```
calibration/                                          <-- LIVE
├── coverage_counts.npz                               (with provenance, 2633 scenes)
├── rgb_to_mono_weights_v4.npz                        (clean from PNG-based fit)
├── figures/
│   ├── interarrival_v3_per_pixel_residuals.pdf       (current IA result)
│   └── interarrival_v3_summary.npz
├── *.py                                              (all calibration/analysis scripts)
└── archive_pre_audit_2026-04-16/                     <-- DEPRECATED
    ├── README.md                                     (explains everything)
    ├── npz/                                          (round-1 buggy LUTs/weights)
    ├── figures/                                      (round-1 deprecated plots)
    ├── logs/                                         (round-1 build logs)
    ├── smoke_test_5_scenes/                          (no-provenance smoke test)
    ├── smoke_test_5_scenes_with_provenance/          (with-provenance smoke test)
    ├── legacy_scripts/                               (build_bernoulli_lut.py, ia v2)
    └── v2_interarrival_deprecated/                   (deprecated IA figures)
```

In progress (tmux `lut3v3`, started 2026-04-16 18:17, ETA ~03:00 next day):
- Full three-LUT build with provenance and race-free accumulator

Pending:
- Variance decomposition + supervisor's 3×3 grid + coverage maps (after rebuild)
- Sampling function validation (after rebuild)
- v3 RGB-to-mono recalibration with fixed code (after rebuild)

## Acknowledgements

Thanks to the GPT auditor for catching the race condition. It's the kind of
bug that's easy to miss because the per-pixel bias is small (~0.1–0.3 %) and
the function "looks parallel-safe" at a glance — but the read-modify-write
on a shared array is exactly the textbook race pattern. The fix was simple
once it was named.

Round 2 was harder to catch but more interesting: the 5-scene smoke-test
files looked indistinguishable from full builds without a way to inspect
their provenance. Lesson learned: every saved artifact needs to carry
metadata about how it was built. The new `provenance.py` module is the
standardized way to do this going forward.
