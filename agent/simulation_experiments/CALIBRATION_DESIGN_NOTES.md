# Calibration Design Notes

Deep-dive design notes on what space to calibrate in, and how the calibration
step slots into the eventual sRGB-to-SPAD simulation pipeline.

---

## The big picture

End goal:

```
online sRGB image  ──(simulate)──▶  plausible SPAD measurement
                                     (either RGB-filtered or mono binary frames)
```

Calibration is one piece of the simulator. It maps between different
"measurement views" of the same scene (e.g. per-channel filtered SPAD → mono
SPAD) so we can use sRGB image as a stand-in radiance estimate.

## The four relevant "spaces"

| Space | Definition | Additive in this space? | Source in our data |
|---|---|---|---|
| **sRGB** | Gamma-encoded display values, [0,1] | **No** — non-linear OETF | `spad_dataset/RGB/`, online datasets |
| **Linear RGB** | `sRGB⁻¹(sRGB)`, [0,1], proportional to display radiance | Approximately yes at low flux | `spad_dataset/RGB_linear16/` (post-WB post-tone-map, pre-gamma) |
| **λ (flux)** | `−ln(1−p)/η`, ∝ photon rate | **Yes** (exact — flux is additive across filters) | Computable from any binary `.bin`; exported to `spad_dataset/monochrome16/…_lambda.png` |
| **p (Bernoulli)** | `counts / N_frames`, [0,1] | **Only approximately**, diverges as `p → 1` | Raw binaries directly; our v1 calibration lives here |

The linear-combination assumption `mono = w_r·R + w_g·G + w_b·B` is
**physically exact in λ-space** and **approximately true** in the other two
linear-ish spaces (Linear RGB and small-p Bernoulli). It is **false** in sRGB
space.

## Why v1 used raw binaries (p-space)

- Direct from physics — Bernoulli rates are what the SPAD actually measures.
- No intermediate processing to introduce errors.
- Minimal dependencies — just bit unpacking + summation.
- Matches the user's explicit spec: "linear map from RGB Bernoulli rates to monochrome Bernoulli rates."

## Audit — issues with the v1 approach for the ultimate sRGB goal

### 1. p-space is not where linearity truly holds

`p = 1 − exp(−η·Φ)` is concave in Φ. The true linear relationship is:

```
Φ_mono = w_r·Φ_r + w_g·Φ_g + w_b·Φ_b      ✅ physical identity
p_mono ≈ w_r·p_r + w_g·p_g + w_b·p_b      ⚠  only linearizes near p≈0
```

Fitting in p-space biases weights and can even flip signs at moderate p — which
is almost certainly what produced the negative `w_b = −0.54` in v1.

### 2. sRGB gamma must be undone before any linear combination

Online data arrives gamma-encoded:
```
sRGB ≈ linear^(1/2.2)   (heavy compression)
```

Linearly combining sRGB-encoded values has no physical meaning. **Gamma decode
is non-negotiable** for any sRGB-fed pipeline. This is a cheap, lossless (in
float32) inversion of the piecewise OETF; modest precision loss only if we
started from 8-bit.

### 3. White balance

`spad_dataset/RGB_linear16/` has had gray-world WB gains applied per scene.
Online images have camera-specific WB already baked in. The v1 calibration
(raw binaries, no WB) absorbs our dataset's natural illuminants. Sign of
contamination depends on inference-time data distribution.

### 4. Tone mapping

`SpadCompositeProcessor` divides by a per-scene 99.5th-percentile before gamma
encoding. This is a scene-dependent scalar — it rescales the intensity range
differently for bright vs. dark scenes. A linear fit across scenes averages
over these arbitrary rescalings.

### 5. Hot pixels

A small number of SPAD pixels fire at anomalously high rates regardless of
incident photons. They pass the `p < 0.95` filter (can sit at 0.3–0.7 from
thermal alone). They violate the spectral model entirely — their rate in mono
is not a weighted sum of filtered-channel rates. `median_hotpixel_fix` in
`utils.py` handles this robustly.

---

## Binaries vs. processed RGB — which input format for calibration?

Two defensible approaches, each with a different downstream contract:

### Approach A: calibrate on raw binaries (current v1, would become v2/v3 in λ-space)

```
binary .bin → counts → p → (hot-pix fix?) → λ = −ln(1−p)
           ↓
  fit: λ_mono = w·λ_RGB
```

**What the weights mean:** pure sensor-and-filter spectral integrals. Independent of
illuminant, independent of any display conventions. They capture the physics
of our specific SPAD sensor + its color filter array.

**How to use at inference from sRGB:**
```
online sRGB
  → inverse sRGB OETF   (gamma decode)
  → linear RGB
  → SCALE: λ_r = α·R_linear, λ_g = α·G_linear, λ_b = α·B_linear
    (α encodes the chosen "exposure" / light level)
  → apply weights: λ_mono = w_r·λ_r + w_g·λ_g + w_b·λ_b
  → p_mono = 1 − exp(−λ_mono)
  → sample Binomial(N_frames, p_mono) for each pixel
```

The `α` scaling is the critical unknown — it's where we inject our exposure
assumption.

**Pros:** physically clean, one calibration covers any exposure level.
**Cons:** need an explicit forward model at inference, more moving parts.

### Approach B: calibrate on processed RGB (would become v4)

```
spad_dataset/RGB_linear16  (linear, post-WB, post-tone-map)
spad_dataset/monochrome16  (λ-encoded mono from same pipeline)  ← for y target

  fit: mono_target = w_r·R + w_g·G + w_b·B  in whichever compatible space
```

**What the weights mean:** whatever mapping exists between the processed-RGB
representation and the processed-mono representation. Absorbs WB, absorbs tone
mapping, absorbs the specific way our pipeline produces RGB images.

**How to use at inference from sRGB:**
```
online sRGB
  → inverse sRGB OETF
  → linear RGB (now in roughly the same space as spad_dataset/RGB_linear16)
  → apply weights: pred_linear_mono = w_r·R + w_g·G + w_b·B
  → (forward model: linear → λ → p_mono, still needed)
  → sample
```

**Pros:** calibration lives in the same representation that online sRGB
decodes into. Fewer inference-time transformations. Matches the "look" of our
existing RGB data.

**Cons:** weights include scene-dependent WB and tone-map effects; mixing with
diverse online datasets may amplify domain shift; still need a
linear-→-Bernoulli forward model at the tail.

### The key insight

**Both approaches end up with "linear RGB + forward model → Bernoulli" at
inference time.** The difference is *where in the pipeline the linear
combination lives*:

- **A (λ-space):** linear combine the per-channel **fluxes** (right before the Bernoulli nonlinearity). The weights are pure physics; α (exposure) is a separate knob.
- **B (linear RGB-space):** linear combine the per-channel **linear intensities** (right after gamma decode). The weights absorb everything else.

For a clean, interpretable, publishable pipeline → **A is preferable** (v2/v3).
For the shortest path to plugging in online data → **B is slightly simpler**
(v4), but you lose separability and physical interpretation.

### Recommendation

Run both. **A (λ-space) is the primary** because:

1. The weights mean something and can be reported.
2. You can change exposure / integration time without refitting.
3. It decouples "what does the sensor see" from "what does the downstream image look like."

But **also** do B as a comparison to check whether the real-world sRGB-to-mono
mapping matches what physics predicts. If they disagree significantly, that
flags an issue in either the calibration set's illuminant distribution, the
processed-RGB pipeline, or the physical forward model — and is itself a useful
result.

---

## On gamma decode precision

The user asked whether inverting sRGB on online images loses precision.

The sRGB OETF in float32 is lossless as a mathematical operation. What can
degrade is quantization:

- **8-bit sRGB → decoded float.** sRGB was designed to put more code values in
  the shadows (where the eye is sensitive). Decoding redistributes 256 levels
  across a true linear range, compressing highlights to fewer distinct values.
  Shadows are actually *better* resolved than in a linear 8-bit encoding
  would be. In practice, for SPAD simulation — where we mostly care about
  low-flux regions — this is **not a problem**.
- **16-bit sRGB → decoded float.** Effectively lossless for our purposes.
- **JPEG artifacts.** A bigger issue than gamma-decode precision. DCT
  quantization adds blocky noise that looks nothing like SPAD noise. Avoid
  heavily-compressed JPEG sources when possible.
- **Demosaicing, denoise, sharpening.** Many online sRGB images have been
  through aggressive ISPs. That introduces spatial correlations foreign to
  raw SPAD physics. This matters more than the gamma step.

**Bottom line:** undoing gamma is cheap and correct. The real precision and
domain concerns are upstream of gamma — JPEG compression, camera ISP, scene
diversity. Float32 throughout; don't worry about the gamma step itself.

---

## Proposed next experiments

1. **v2-lambda-space** — rerun v1 with `y = −ln(1−p_mono)`, `X = column_stack(−ln(1−p_R/G/B))`. Compare to v1 weights and R². Expect all-positive weights, higher R².
2. **v3-lambda-hotpixfix** — v2 + `median_hotpixel_fix` on the raw counts. Marginal but clean.
3. **v4-linearRGB** — fit directly on decoded `spad_dataset/RGB_linear16` with decoded `spad_dataset/monochrome16` lambda images as target. Compare to v3 weights. See whether the WB + tone-map pipeline is neutral or not.
4. **End-to-end smoke test** — take a few online sRGB images, run the v3 or v4 pipeline, visualize the synthesized mono SPAD frames side-by-side with real mono SPAD captures of similar scenes. Qualitative sanity check.

Deferred unless needed:
- Calibrating a full `[3×3]` RGB-to-RGB matrix (e.g. for color-filter-array-aware simulation). Not clear the SPAD forward model benefits yet.
- Per-region / per-luminance-bin calibration to account for residual nonlinearity.
