# SPAD Simulation Calibration & Sampling Pipeline

This directory holds the calibration scripts, LUTs, and sampler that turn an
ordinary sRGB image into a plausible 1-bit SPAD binary frame stack.

For the full project narrative, audit responses, and cross-references to the
DiffSynth project, see
`/nfs/horai.dgpsrv/ondemand30/jw954/DiffSynth-Studio-SPAD/agent/simulation_experiments/`.

---

## Pipeline at a glance

```
   sRGB image (any 512×512)
         │
         │  inverse sRGB OETF (gamma decode)
         ▼
   linear RGB
         │
         │  RGB → mono via calibration weights (v3 = λ-space, or v4 = PNG-space)
         ▼
   predicted mono intensity
         │
         │  exposure scaling α:  p_true = 1 - exp(-α · M)
         ▼
   per-pixel Bernoulli rate p_true(x, y)
         │
         │  pyramid LUT: for each pixel, sample p̂ from the cascade
         │  L=1 → L=2 → L=4 → … → L=512 (fixed-pattern noise injection)
         ▼
   per-pixel rate map p_simulated(x, y)
         │
         │  Bernoulli sample × N frames
         ▼
   binary frame stack
         │
         │  MSB-first bit-pack
         ▼
   scene/RAW_empty.bin   (byte-compatible with real SPAD captures)
```

All output `.bin` files are processed by `extract_binary_images.py` exactly
like real SPAD captures.

---

## Run order (cold start)

1. `coverage_check.py` — fast pre-pass over `bits_multi_1000/` PNGs to estimate per-pixel sample density. (~3 min)
2. `build_three_luts.py` — for each scene, read 20 000 frames of `RAW_empty.bin`, compute K=64 short-burst p̂ and full-frame p_GT, build per-pixel/super-pixel/global LUTs in one pass with provenance. (~8 h, run in tmux)
3. `build_lut_pyramid.py` — derive 10 dyadic levels (1, 2, 4, …, 512) from the per-pixel LUT. (~15 min)
4. `variance_analysis.py` — Var_total / Var_between / Var_within decomposition + supervisor's per-pixel grid. (~3 min)
5. `pyramid_sampler.py` — 1M-query usage validation across the cascade. (~5 min)
6. `pyramid_figures.py` — fallback-map heat-map and usage bar chart. (~2 min)
7. `test_cascade.py` — 7 integrity tests on the pyramid sampler in isolation (reachability, edge bins, safety net, unbiasedness, determinism, level containment). (~5 min including pyramid load)
   `test_afhq_simulation.py` — 8 end-to-end tests on the simulated AFHQ `.bin` outputs (file integrity, frame independence, SNR convergence, LUT-FPN visibility, extract-PNG match). (~1 min, run after step 11)
8. `interarrival_analysis_v3.py` — per-pixel residual + lag-1 autocorrelation against Geometric prediction. (~22 min)
9. `calibrate_rgb_to_mono_v3.py` — fit RGB→mono in λ-space using all four binary channels. (~2 h, run in tmux)
10. `afhq_simulate.py` — end-to-end demonstration: AFHQ → simulated SPAD .bin. (~5 min for 10 scenes × 10 000 frames)
11. `afhq_make_grid.py` — comparison figure across frame-accumulation levels.

The intermediate `extract_binary_images.py` step (run between 10 and 11) lives
in `/scratch/ondemand30/jw954/spad-diffusion/spad_dataset/`.

---

## File inventory

### Live scripts

| File | Purpose |
|---|---|
| `spad_utils_fixed.py` | Race-free count accumulator (per-thread chunk reduction). Replaces the racy `utils.py:_accumulate_counts_core`. |
| `provenance.py` | Provenance helper: load, summarize, and assert compatibility across saved NPZ artifacts. |
| `coverage_check.py` | Per-cell sample-count statistics from pre-extracted probability PNGs. |
| `build_three_luts.py` | Single-pass builder for global / super-pixel / per-pixel LUTs + variance stats. |
| `build_lut_pyramid.py` | Derives 10-level dyadic pyramid from per-pixel LUT. |
| `pyramid_sampler.py` | `PyramidSampler` class with batch fallback walk + 1M-query validation. |
| `pyramid_figures.py` | Per-pixel fallback map and usage bar chart. |
| `variance_analysis.py` | Variance decomposition + supervisor's 3×3 per-pixel figure. |
| `test_cascade.py` | 7-test integrity check on the pyramid sampler in isolation. |
| `test_afhq_simulation.py` | 8-test end-to-end validation of the simulated `.bin` outputs (file integrity, frame independence, SNR scaling, LUT-FPN visible, extract-PNG roundtrip). |
| `interarrival_analysis_v3.py` | Per-pixel residual + lag-1 methodology for SPAD photon-statistics validation. |
| `calibrate_rgb_to_mono.py` | v1: raw p-space RGB→mono fit (superseded). |
| `calibrate_rgb_to_mono_v3.py` | v3: λ-space + hot-pixel fix RGB→mono fit (primary). |
| `calibrate_rgb_to_mono_v4.py` | v4: fit on pre-processed PNGs (sanity comparison). |
| `compare_versions.py` | Side-by-side comparison of v1 / v3 / v4 weight sets. |
| `sample_p_hat.py` | Legacy 3-level sampler (per-pixel → super-pixel → global). Kept for back-compat; `pyramid_sampler.py` is now canonical. |
| `afhq_simulate.py` | sRGB → simulated SPAD `.bin`. |
| `afhq_make_grid.py` | Comparison grid figure for AFHQ outputs. |

### Live artifacts

| File | Size | What it is |
|---|---|---|
| `lut_pyramid.npz` | 28.7 GB | 10-level dyadic pyramid (canonical sampler input) |
| `lut_per_pixel.npz` | 3.6 GB | L=1 of pyramid, kept for back-compat |
| `spad_bernoulli_lut_v2.npz` | 2.8 GB | L=8 of pyramid, kept for back-compat |
| `lut_global.npz` | 2.8 GB | L=512 of pyramid, kept for back-compat |
| `variance_stats.npz` | 264 MB | per-pixel per-bin sufficient statistics |
| `coverage_counts.npz` | 44 MB | coverage pre-pass (for sanity-check joins) |
| `pyramid_usage_stats.npz` | small | 1M-query level breakdown |
| `variance_decomposition.npz` | small | Var_total/Var_between/Var_within per bin |
| `cascade_test_report.txt` | small | output of `test_cascade.py` |
| `afhq_simulation_test_report.txt` | small | output of `test_afhq_simulation.py` |
| `rgb_to_mono_weights_v3.npz` | small | primary RGB→mono weights (λ-space) |
| `rgb_to_mono_weights_v4.npz` | small | secondary weights (PNG-space) |

### Live figures (`figures/`)

- `variance_decomposition.pdf` — Var_total vs Var_between vs Var_within
- `per_pixel_variance_vs_gt.pdf` — supervisor's 3×3 grid (Typical / High DCR / Low PDE)
- `per_pixel_coverage_N{1,5,10,20,50}.pdf` — spatial coverage heat-maps
- `pyramid_usage_bar.pdf` — fraction of 1M queries resolved at each level
- `pyramid_fallback_map_p03.pdf` — per-pixel fallback level for bin 77 (p≈0.30)
- `interarrival_v3_per_pixel_residuals.pdf` — Poisson validation (no afterpulsing detected)

### Outputs of the AFHQ demo

`/scratch/ondemand30/jw954/afhq_simulation/`:
- `binaries/` — 10 scenes × `RAW_empty.bin` (10 000 frames each, 327 MB each)
- `extracts/frames_{1,10,100,1000,10000}/` — 50 accumulated PNGs from extract
- `figures/afhq_simulation_grid.pdf` — comparison figure
- `simulation_summary.csv` — per-scene p_true vs observed stats

### Archive

`archive_pre_audit_2026-04-16/` — pre-audit artifacts (race-buggy LUTs,
deprecated v1/v2 inter-arrival, smoke-test outputs). README.md inside
explains why each is deprecated.

---

## Calibrations and conventions

- **Image dimensions**: 512×512 throughout (matches the SPAD sensor).
- **Bit order**: MSB-first within each packed byte (`np.unpackbits(bitorder='big')`).
- **Rotation**: `np.rot90(k=1)` (counterclockwise 90°) is the convention in `extract_binary_images.py`. The simulator's `--inverse-rotate-k -1` cancels this so the output PNGs match input orientation.
- **GT bin**: `int(p_GT * 255 + 0.5)` clipped to `[0, 255]`. Both writer and reader must use the same convention; verified.
- **Cell flatten order**: `[sy, sx, bin]` numpy row-major. `flat_idx = sy * W_L * N_BINS + sx * N_BINS + bin`.
- **Hot pixel fix**: median filter (`utils.median_hotpixel_fix`, ksize=3, spike thresh = `max(10, 0.01·N_frames)`). Applied to counts before normalizing in `build_three_luts.py` and `calibrate_rgb_to_mono_v3.py`. NOT applied in inter-arrival analysis (per-pixel time series, not aggregate statistics).
- **Filter file**: `RAW_empty.bin` for monochrome (the primary calibration target); `RAW_col_{r,g,b}.bin` for color-filtered captures used by RGB→mono calibration.
- **Provenance**: every saved NPZ carries `artifact_kind, n_scenes_used, total_samples, scene_list_hash, accumulator_module, k_short, n_gt_frames, rotate_k, hotpix_fix_enabled, build_timestamp`. Use `provenance.load(path)` and `provenance.assert_compatible(*provs)` to safely combine artifacts.

---

## Key results

- **RGB→mono v3 calibration** (λ-space, race-free):
  `w = (1.033, 1.178, 0.866)`, sum ≈ 3.08, Val R² = 0.971.
  Sum ≈ 3 matches the spectral-partition identity `F_r + F_g + F_b ≈ 1`.
- **Inter-arrival v3** (per-pixel residual): mean residual = +0.0002,
  mean lag-1 = +0.003. **No afterpulsing detected.** SPAD is well-modelled
  by per-pixel Geometric. The earlier v2 finding of "17 % afterpulsing" was a
  rate-mixing artifact — see `AUDIT_RESPONSE_2026-04-16.md`.
- **Variance decomposition**: `Var_between / Var_total` ≈ 28 % across most
  bins. Fixed-pattern noise is moderate; super-pixel grain captures most of it.
- **Pyramid sampler** (1M queries, β(2,10) GT, min_samples=20):
  - 48.5 % resolved at L=1 (per-pixel)
  - 39.9 % resolved at L=2 (4-pixel pool)
  - 11.1 % resolved at L=4 (16-pixel pool)
  -  0.5 % resolved at L=8
  - <0.001 % at any coarser level; 0 unresolved.
- **Pyramid cascade test**: 7/7 pass. See `cascade_test_report.txt`.
- **AFHQ end-to-end test**: 8/8 pass. See `afhq_simulation_test_report.txt`. Confirms `.bin` files have correct size, MSB bit-packing roundtrips, accumulated rates exactly match the simulator's claimed `obs_rate`, frames are i.i.d. (frame-0 vs frame-9999 correlation = +0.00063), forward model agrees within 0.44%, SNR scales as theory predicts (std(K=10)/std(K=1000) = 10.53 vs theoretical 10.54), LUT injection visible at 22× the binomial floor, and extract-PNG matches re-accumulated `.bin` to within 7.6e-6 (PNG quantization).
- **AFHQ demo**: 10 sRGB images → simulated SPAD binary stacks. K=10 000 accumulations recover clean grayscale photos visually matching predicted mono GT; K=1 single-frame outputs indistinguishable from real SPAD single-frame captures of similar-rate scenes.

---

## Known limitations

1. **Camera-RGB ↔ SPAD-flux bridge is uncalibrated.** v3 weights describe SPAD-channel flux integrals, not arbitrary camera channels. For physically-correct sRGB simulation, a 3×3 CCM should be calibrated between camera RGB and SPAD R/G/B-filter color spaces. Currently we treat them as proportional with a single scalar exposure α.
2. **Exposure scaling α is hand-picked.** A future refinement would sample α from the empirical λ-distribution observed in the calibration set.
3. **Per-scene tone mapping in v4 weights.** v4 was fit on PNGs that had per-scene 99.5-percentile normalization, which biases the weights toward sum ≈ 1 (an artifact, not physics). Use v3 unless you specifically need PNG-space coefficients.
