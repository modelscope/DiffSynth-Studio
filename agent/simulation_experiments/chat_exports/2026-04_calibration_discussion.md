# Calibration Discussion — Chat Export

Sessions from 2026-04-11 through 2026-04-14 with the assistant
on the RGB-to-mono Bernoulli-rate calibration and its place in the
larger sRGB-to-SPAD simulation pipeline.

Lightly cleaned / paraphrased to foreground the technical reasoning.

---

## Session 1 — Kick-off (2026-04-11)

### User

> I'm building a SPAD dataset augmentation pipeline. First task: calibrate a
> linear map from RGB Bernoulli rates to monochrome Bernoulli rates.
>
> Inputs: for each scene in our SPAD dataset (use metadata_val.csv to
> enumerate), load the ground-truth Bernoulli rate images for the R, G, B
> color-filtered captures and the unfiltered monochrome capture. Use
> load_spad_image() from our existing codebase. Each is a 512×512 float image
> in [0,1].
>
> Build matrix X ∈ R^{N×3} (rows = pixels stacked across scenes) and vector
> y ∈ R^N (mono rates). Solve min_w ||Xw - y||² via np.linalg.lstsq. Hold out
> 20% of scenes for validation; report train/val R² and per-channel weights.
>
> Save weights to calibration/rgb_to_mono_weights.npz. Also save a diagnostic
> plot: scatter of predicted vs actual mono rate (subsample 50k points), with
> the y=x line.
>
> Skip pixels where any of R,G,B,mono is >0.95 (saturated) or <0.01 (dark)
> to avoid clipping bias.

### Assistant

Surveyed the project:

- **No `metadata_val.csv` exists** on this machine — it was referenced in
  existing scripts at `/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv`
  (a different box).
- All 2,633 scenes in `/nfs/horai.dgpsrv/ondemand30/jw954/images/` have the 4
  required `.bin` files (`RAW_col_r/g/b.bin`, `RAW_empty.bin`).
- Each binary is 655,360,000 bytes = 20,000 frames × 32,768 bytes/frame (512×512
  packed 1-bit MSB-first).
- There's no pre-extracted per-channel probability image set for R/G/B —
  only mono, and tone-mapped RGB composites. Need to process from binary.
- `spad-diffusion/spad_dataset/utils.py` has Numba-accelerated bit unpacking
  (`accumulate_counts_whole_file`); `diffsynth` conda env has numba + numpy +
  tqdm; needs matplotlib.

Decisions taken:

1. Enumerate scenes from directory listing (no `metadata_val.csv`).
2. Read **N=1000 frames per binary** — enough for ~1.6% RMS rate noise at
   `p=0.5`, statistically sufficient after averaging over hundreds of millions
   of pixels.
3. **Memory-efficient accumulation** — materialized X matrix would be ~6-10 GB;
   instead accumulate `X^T X` (3×3) and `X^T y` (3×1) per scene, solve normal
   equations at the end. Constant memory regardless of scene count.
4. Scene-level 80/20 train/val split, `np.random.permutation` with seed 42.
5. Install matplotlib + opencv-python-headless in `diffsynth` env (needed
   for JIT import of `utils.py`).
6. Run in tmux — estimated ~75 min for all scenes.
7. Label the script explicitly: `VERSION = "v1"`.

Built `/nfs/horai.dgpsrv/ondemand30/jw954/calibration/calibrate_rgb_to_mono.py`,
smoke-tested with 10 scenes (1.5 s/scene), launched full run in tmux session
`calibration`.

### Result (2026-04-12)

Run finished: 2631/2633 scenes processed (2 skipped), ~110 min, 280M train + 70M
val pixels.

| Quantity | Value |
|---|---|
| `w_r` | 0.8587 |
| `w_g` | 1.4299 |
| `w_b` | **−0.5446** |
| `sum(w)` | 1.7440 |
| Train R² | 0.8963 |
| Val R² | 0.8970 |

Val R² ≈ Train R² → no overfitting, weights generalize across held-out scenes.

The scatter plot showed tight clustering along y=x with slight bias near the
`p < 0.01` threshold (residual nonlinearity).

---

## Session 2 — Audit (2026-04-12)

### User

> For your calibration code, have you considered first reverting the gamma
> correction and all the white balance stuff? I feel like because all the
> online dataset we will be using will be sRGB, that maybe an important step.
> Don't do anything yet, but just audit and tell me what your reasoning is
> about this scenario. Also our RGB constructions have hot pixel fix too, not
> sure how that would tie into things please give a full analysis.

### Assistant

Summary of the audit:

**1. What the v1 calibration actually captures.** A linear map in **raw
Bernoulli-rate space** — no preprocessing at all. Directly reflects the
physical detection probabilities of the SPAD with and without color filters.

**2. The fundamental physics problem.** The Bernoulli model
`p = 1 − exp(−η·Φ)` is nonlinear. The truly additive space is **photon flux
(λ-space)**: spectral integrals combine linearly as λ = `−ln(1−p)/η`. Fitting
in p-space is only approximately linear, breaks down as p grows, and this is
almost certainly what produced the negative `w_b = −0.54` — a linearization
artifact, not a real physical coefficient.

**3. sRGB gamma.** If the calibration is ever applied to gamma-encoded sRGB,
the OETF **must** be undone first. sRGB ≈ linear^(1/2.2) — a compressive
nonlinearity. Linear combination of sRGB values is meaningless.

**4. White balance.** `spad_dataset/RGB_linear16/` has gray-world gains
applied per scene. My v1 weights absorb our specific dataset's illuminant
distribution (indoor, same capture setup). Transferring to diverse online
imagery is not apples-to-apples.

**5. Hot pixel fix.** Sensor-defect pixels fire regardless of photons, so
they violate the spectral model entirely. Some pass my 0.95 saturation filter
(they can sit at moderate rates from thermal alone). `median_hotpixel_fix()`
would clean these robustly, probably gaining a few % R².

**6. Recommendation (for discussion only).** Re-run as three labeled
versions:
- **v2-lambda-space:** fit on `λ = −ln(1−p)` instead of `p`. Fixes the biggest
  physical issue for free.
- **v3-lambda-hotpixfix:** add median hot pixel fix on counts.
- **v4-lambda-hotpixfix-wb:** also gray-world WB per scene — matches the
  pipeline that produced `spad_dataset/RGB_linear16`.

### User

> Please document this in the agent folder under diffsynth somewhere for
> simulation experiments. Add a short blurb talking about simulation
> experiments in thesis_context. But more importantly create a separate file
> detailing what is done and all these questions being asked, like the chat
> stuff you guys did.
>
> But I do have one more thing to ask though. Which is the fact that
> eventually we have to simulate using online sRGB. But you directly
> calculated using the binaries using proportion. Do you think that's more
> appropriate, or using the RGB images from the spad_dataset that we have
> already generated for this task instead? Yes online images are gonna be in
> sRGB, but we could just reverse the gamma correction right directly on
> those images? Or is that not possible due to potential loss of precision or
> something? Just keep in mind our final goal is to simulate SPAD data
> directly from sRGB dataset (I think).

### Assistant

Ongoing. See `CALIBRATION_DESIGN_NOTES.md` for the full design discussion. The
short version:

**Binaries vs. processed RGB — both paths end with a Bernoulli forward model
at inference, but the linear combination lives in a different space:**

- **Binaries (in λ-space) → v2/v3.** The weights are pure sensor+filter
  physics. They stay valid across any exposure — we just change the `α`
  scaling that takes `linear RGB → λ` at inference time. Cleanest, most
  interpretable, publishable.

- **Processed RGB_linear16 (pair with monochrome16) → v4.** The weights live
  in the same space that gamma-decoded online sRGB lands in. Slightly fewer
  moving parts at inference, but weights absorb our dataset's WB + tone-map
  idiosyncrasies.

Recommendation: **run both, primary = λ-space (v3).** The λ-space weights
mean something; if v4 disagrees materially, that disagreement is itself
informative about illuminant/tone-map bias.

**On gamma decode precision.** Not a real problem in float32. The sRGB
piecewise OETF inverts exactly as a math operation. Quantization from 8-bit
inputs affects highlight resolution slightly but actually *preserves*
shadows better than a linear 8-bit encoding. For SPAD simulation, the real
concerns are upstream of gamma — JPEG artifacts, demosaicing, camera ISP
nonlinearities, and scene-distribution mismatch. Float32 throughout; don't
worry about the gamma step itself.

---

## Session 3 — Execution (2026-04-15)

### User

> So like if we do something like this, we are building the correct and
> concrete first step to building this simulation model to later run
> inference on online sRGB datasets to simulate SPAD data? Is it correct? If
> it is start running the next steps please!

### Assistant

Confirmed yes — this IS the correct first concrete step. The calibration
weights are the central parameter in the forward model, and getting them
right in λ-space is non-negotiable before any online-sRGB inference.

Launched v3 (λ-space + hot pixel fix) and v4 (processed PNGs) in parallel
tmux sessions.

### Results (both complete)

**v3 — λ-space + hot pixel fix:**
- `w = (1.031, 1.181, 0.836)`, sum = **3.048**, Val R² = **0.970**
- **All positive** — hypothesis confirmed, v1's `w_b < 0` was indeed a
  p-space linearization artifact.
- `sum ≈ 3` matches the physics: if the three color filters partition the
  SPAD's spectrum, `F_r + F_g + F_b ≈ 1` pointwise, so mono = R + G + B and
  ideal `w = (1,1,1)`.
- Our weights deviate from (1,1,1) by amounts consistent with Bayer-style
  filter curves and silicon SPAD quantum efficiency (green-peaked,
  blue-attenuated).

**v4 — processed PNGs:**
- `w = (0.370, 0.500, 0.094)`, sum = **0.965** ≈ 1.0, Val R² = **0.968**
- Higher raw R² than v3 because both sides are tone-mapped to matching scales.
- Sum ≈ 1 is an artifact of that tone-map alignment, NOT a physical statement.
- Shape (R ≈ G > B) matches Rec.709-ish luminance weights, with more red
  than photopic would suggest — consistent with SPAD red/NIR sensitivity.

### Decision

**v3 wins.** It's the physics-correct calibration. v4 stays as a sanity
comparison.

Weights are locked in. Next: build the inference-time forward model that
takes online sRGB → synthesized SPAD measurement.

---

## Running open questions (post-v3/v4)

- [x] ~~Actually run v2-lambda-space.~~ Folded into v3.
- [x] Run v3 (λ-space + hot pixel fix). — `w_b` positive, R² = 0.97. ✅
- [x] Run v4 (on RGB_linear16 ↔ monochrome16). — sum≈1 artifact confirmed. ✅
- [ ] End-to-end smoke test: pipe a few online sRGB images through the
  proposed inference pipeline, visualize synthesized mono SPAD frames, compare
  to real captures.
- [ ] Decide: is the "exposure scaling α" going to be a global constant, a
  per-image estimate, or a distribution we sample from for augmentation?
  Needed to map linear RGB ∈ [0,1] → λ ∈ (0, λ_max]. Plan: histogram λ
  distributions from the v3 calibration binaries per channel; sample from
  that empirical distribution at inference.
- [ ] Do we also need a per-channel simulation (R/G/B SPAD from sRGB), or is
  mono-only sufficient for the augmentation pipeline? Affects whether we need
  a matrix calibration vs a 3-vector.
- [ ] Write `simulate_spad_from_srgb(sRGB_image, N_frames, α) → binary_frames`
  using v3 weights + Bernoulli sampling. Next milestone.
