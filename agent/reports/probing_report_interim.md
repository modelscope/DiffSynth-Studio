# Linear Probing Interim Report — SPAD-FLUX DiT Analysis
**Date: 2026-03-24 | Status: Interim (control baseline in progress)**

---

## 1. What Is This & Why Does It Matter?

We apply **AC3D-inspired linear probing** to our FLUX.1-dev DiT to answer:

1. **Where** in the 57-block transformer is SPAD measurement information encoded?
2. **When** during 28-step denoising does scene understanding (depth, uncertainty) emerge?
3. **What** does LoRA fine-tuning add beyond the pretrained model + ControlNet?

A **linear probe** (ridge regression) tests whether a target property is **linearly decodable** from internal activations. High R² means the information is explicitly represented; low R² means it's either absent or encoded nonlinearly.

---

## 2. Experimental Setup

### Architecture
- **FLUX.1-dev**: 12B rectified-flow transformer
  - 19 **joint** transformer blocks (cross-attend to text tokens) → J0–J18
  - 38 **single** transformer blocks (self-attention only) → S0–S37
  - Hidden dimension: **3072** per image token
  - Image token count: **1024** tokens (32×32 patch grid for 512×512 images)

### Model Configurations

| Config | Model | Blocks Probed | Samples | Probing Mode | Status |
|--------|-------|--------------|---------|--------------|--------|
| **Experiment 1** (sparse+spatial) | FLUX + ControlNet + LoRA | 10 sparse | 100 | Global + Spatial | **Complete** |
| **Experiment 2** (all-blocks) | FLUX + ControlNet + LoRA | All 57 | 776 | Global only | **Complete** |
| **Experiment 3** (control) | FLUX + ControlNet, **NO LoRA** | All 57 | 776 | Global only | **In progress (22%)** |

### Probe Targets
| Target | Description | Source | Why It Matters |
|--------|-------------|--------|----------------|
| **Bit density** | Mean of binary SPAD frame (photon evidence) | Input conditioning | Does the network preserve measurement statistics? |
| **Depth** | Monocular depth from Depth Anything V2 | Pseudo-GT | Does the network infer 3D structure from 1-bit data? |
| **Cross-seed variance** | Pixel-wise std across 10 seeds | Multi-seed generation | Does the network know where it's uncertain? |

### Probing Algorithm
- **Activation extraction**: Forward hooks on DiT blocks capture image-token features at 7 denoising timesteps (t ∈ {0, 4, 9, 14, 19, 24, 27} out of 28 steps)
- **Global mode**: Mean-pool 1024 tokens → 1 vector (dim 3072) per image. Ridge regression predicts scalar target.
- **Spatial mode**: Keep all 1024 tokens. Ridge regression predicts per-patch target (32×32 map). Uses streaming XᵀX/Xᵀy accumulation to avoid OOM.
- **Ridge regression**: Closed-form solve with:
  - Feature standardization (zero mean, unit variance)
  - **y-centering** (subtract mean of training targets before fitting, add back at prediction)
  - Trace-normalized regularization: λ_scaled = λ · tr(XᵀX) / D
  - λ = 0.1 (default)
- **Metric**: R² (coefficient of determination). R²=1 is perfect; R²=0 is mean-prediction; R²<0 is worse than predicting the mean.
- **Train/test split**: 80/20 random split

---

## 3. Critical Bug Fix: y-Centering

### The Problem
The original ridge regression implementation standardized features (X) but **did not center the target variable (y)**. This caused:
- Predictions with **correct correlation direction** (Pearson r up to 0.99)
- But **catastrophically wrong magnitude** (R² as low as -50)

### Why This Happens
Without y-centering, ridge regression has no intercept term. The model is forced to predict y through the origin in the feature-transformed space. When y has a non-zero mean (e.g., bit density ≈ 0.3), predictions are systematically offset, inflating residuals.

### The Fix
```python
# Center targets before fitting
mu_y = y_train.mean()
yn = y_train - mu_y
# ... solve ridge with centered targets ...
# Predict with intercept
yp = (X_test @ w).squeeze() + mu_y
```

### Impact
| Target | Before Fix (R²) | After Fix (R²) | Notes |
|--------|-----------------|----------------|-------|
| Bit density | -8.83 (best) | **+0.998** (best) | Same Pearson r ≈ 0.99 before and after |
| Depth | -0.02 (best) | **+0.437** (best) | Was masked by magnitude error |
| Variance | -0.86 (best) | **+0.424** (best) | Now properly positive |

---

## 4. Results — Experiment 2: All-Blocks Global Probing (776 samples, 57 blocks)

This is our **primary experiment** — full validation set, every block, LoRA-enabled model.

### Headline Numbers

| Target | Best R² | Best Block | Best Timestep | Pearson r | Positive R² blocks |
|--------|---------|------------|---------------|-----------|-------------------|
| **Bit density** | **0.998** | J8 | t=0 | 0.999 | 398/399 (99.7%) |
| **Depth** | **0.437** | S4 | t=4 | 0.692 | 235/399 (58.9%) |
| **Variance** | **0.424** | J1 | t=14 | 0.662 | 333/399 (83.5%) |

### 4.1 Bit Density — Where the Network Preserves Measurement Evidence

**Key finding**: Bit density is **almost perfectly linearly decodable** across the entire network.

- **Peak zone**: Joint blocks J5–J12 at t=0, all with R² > 0.995
- **Best single entry**: J8 at t=0 with R² = 0.998
- **Temporal pattern**: Strongest at t=0 (right after ControlNet injection), monotonically decaying toward t=27
- **Single blocks**: Still high (R² ≈ 0.93–0.97) but lower than joint blocks
- **Exception**: J0 is weak (R² = 0.76 max) — the very first block before information has been processed

**Top 10 blocks for bit density:**

| Rank | Block | Timestep | R² | Pearson r |
|------|-------|----------|----|-----------|
| 1 | J8 | t=0 | 0.998 | 0.999 |
| 2 | J7 | t=0 | 0.998 | 0.999 |
| 3 | J6 | t=0 | 0.998 | 0.999 |
| 4 | J9 | t=0 | 0.997 | 0.999 |
| 5 | J5 | t=9 | 0.997 | 1.000 |
| 6 | J10 | t=0 | 0.997 | 0.999 |
| 7 | J2 | t=9 | 0.996 | 1.000 |
| 8 | J11 | t=0 | 0.996 | 0.999 |
| 9 | J1 | t=9 | 0.996 | 1.000 |
| 10 | J12 | t=0 | 0.995 | 0.999 |

### 4.2 Depth — Where 3D Scene Understanding Emerges

**Key finding**: Depth is **moderately decodable** and concentrated in **early single blocks at early-to-mid timesteps**.

- **Peak zone**: S1–S7 at t=4 to t=9, with R² in the range 0.25–0.44
- **Best entry**: S4 at t=4 with R² = 0.437
- **Joint block peak**: J11 at t=19 with R² = 0.295 — depth understanding builds through the joint blocks
- **Temporal pattern**: Peaks at t=4–t=9 (early-to-mid denoising), NOT at t=0 — depth must be *inferred* from the SPAD data, not directly present in the conditioning
- **Late single blocks**: Depth signal weakens (R² ≈ 0.15–0.20), as the network shifts to texture generation

**Top 10 blocks for depth:**

| Rank | Block | Timestep | R² | Pearson r |
|------|-------|----------|----|-----------|
| 1 | S4 | t=4 | 0.437 | 0.692 |
| 2 | S1 | t=4 | 0.420 | 0.681 |
| 3 | S7 | t=9 | 0.409 | 0.652 |
| 4 | S3 | t=4 | 0.382 | 0.668 |
| 5 | S4 | t=9 | 0.372 | 0.635 |
| 6 | S5 | t=9 | 0.371 | 0.631 |
| 7 | S6 | t=9 | 0.315 | 0.581 |
| 8 | S18 | t=4 | 0.300 | 0.598 |
| 9 | S22 | t=14 | 0.297 | 0.558 |
| 10 | J11 | t=19 | 0.295 | 0.617 |

### 4.3 Variance — Where the Network Encodes Uncertainty

**Key finding**: Uncertainty is **moderately decodable** and peaks in **joint blocks at mid-to-late timesteps**.

- **Peak zone**: Joint blocks J0–J12 at t=14–t=27, with R² in the range 0.30–0.42
- **Best entry**: J1 at t=14 with R² = 0.424
- **Temporal pattern**: Peaks at t=14–t=27 (late denoising) — the network becomes aware of its uncertainty as it commits to details
- **Single blocks**: Weaker but nonzero (R² ≈ 0.10–0.33), with a secondary peak at S7 t=4 (R² = 0.326)
- **Interpretation**: Joint blocks maintain stronger uncertainty awareness, possibly because cross-attention to text tokens provides context for ambiguity

**Top 10 blocks for variance:**

| Rank | Block | Timestep | R² | Pearson r |
|------|-------|----------|----|-----------|
| 1 | J1 | t=14 | 0.424 | 0.662 |
| 2 | J1 | t=24 | 0.422 | 0.661 |
| 3 | J12 | t=24 | 0.418 | 0.700 |
| 4 | J1 | t=19 | 0.418 | 0.666 |
| 5 | J1 | t=27 | 0.412 | 0.657 |
| 6 | J7 | t=27 | 0.403 | 0.653 |
| 7 | J6 | t=27 | 0.392 | 0.658 |
| 8 | J2 | t=24 | 0.384 | 0.657 |
| 9 | J0 | t=19 | 0.380 | 0.629 |
| 10 | J11 | t=24 | 0.370 | 0.660 |

---

## 5. Results — Experiment 1: Sparse Spatial Probing (100 samples, 10 blocks)

This experiment tests **per-pixel spatial resolution** of the probes — can we decode a 32×32 map of each target from the 1024 image tokens?

### Headline Numbers

| Target | Global R² (best) | Spatial R² (best) | Spatial Improvement |
|--------|-----------------|------------------|-------------------|
| Bit density | 0.985 (S37, t=0) | **0.990** (J9, t=4) | Spatial ≈ global |
| Depth | 0.168 (S9, t=27) | **0.648** (S9, t=14) | **+0.48 absolute** |
| Variance | -0.862 (S37, t=9)* | **0.433** (S9, t=14) | **Dramatic** |

*Note: The global results in Experiment 1 used only 100 samples (vs 776 in Experiment 2) with only 10 sparse blocks. With the full dataset (Experiment 2), global depth reaches R²=0.437 and variance reaches R²=0.424.

### Why Spatial Probing Is Much Better for Depth and Variance

Depth and variance are **spatially heterogeneous** — they vary across the image. Global mean-pooling collapses 1024 tokens into 1 vector, losing the spatial pattern. Spatial probing preserves the full per-patch structure:

- Each of 1024 tokens "sees" its local receptive field
- The probe can use local geometry (near objects = high depth gradient) instead of averaging over the whole scene
- For depth: a single global scalar (mean depth) is uninformative; a 32×32 depth map IS the signal
- For variance: uncertainty is localized (high in textureless regions, low in structured areas)

### Spatial Probing — Selected Results

**Spatial Bit Density** (best blocks):

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 |
|-------|-----|-----|-----|------|------|------|------|
| J9 | 0.988 | **0.990** | 0.987 | 0.982 | 0.972 | 0.943 | 0.904 |
| J14 | **0.990** | 0.987 | 0.983 | 0.977 | 0.960 | 0.926 | 0.893 |
| S37 | 0.986 | 0.971 | 0.967 | 0.955 | 0.933 | 0.880 | 0.806 |

**Spatial Depth** (best blocks):

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 |
|-------|-----|-----|-----|------|------|------|------|
| S9 | 0.359 | 0.567 | 0.631 | **0.648** | 0.619 | 0.605 | 0.526 |
| S0 | 0.229 | 0.482 | 0.565 | **0.610** | 0.601 | 0.588 | 0.476 |
| J18 | 0.217 | 0.424 | 0.531 | **0.606** | 0.603 | 0.589 | 0.476 |

**Spatial Variance** (best blocks):

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 |
|-------|-----|-----|-----|------|------|------|------|
| S9 | 0.132 | 0.364 | 0.424 | **0.433** | 0.385 | 0.380 | 0.403 |
| J18 | 0.179 | 0.362 | 0.366 | **0.409** | 0.372 | 0.384 | 0.376 |
| S19 | 0.235 | 0.373 | 0.408 | 0.397 | 0.398 | 0.370 | 0.365 |

---

## 6. Interpretation — Information Flow Through the DiT

### Layer-by-Layer Knowledge Profile

```
Block:  J0  J1 ····· J8  J9 ····· J18 | S0  S1 ····· S7 ····· S37
        ─────────────────────────────────────────────────────────────
Bit     ▁▇  ▇▇      ▇▇  ▇▇      ▇▇  | ▇▇  ▇▆      ▆▅      ▆▆
Depth   ▁▁  ▁▁      ▁▂  ▁▂      ▂▃  | ▂▃  ▃▅      ▅▃      ▂▂
Var     ▁▃  ▅▅      ▄▃  ▃▃      ▃▃  | ▃▂  ▂▂      ▃▂      ▂▂
```

1. **J0 (input)**: Noisy — neither measurements nor structure well-formed yet
2. **J1–J8 (early joint)**: Bit density saturates to R²≈0.998. ControlNet signal is fully absorbed. Variance awareness begins building (R²≈0.30–0.40)
3. **J9–J18 (late joint)**: Bit density remains near-perfect. Depth starts emerging (R²≈0.10–0.30 in joint blocks). This is the **measurement → geometry conversion zone**
4. **S0–S7 (early single)**: **Peak depth** (R²=0.44). The model has converted SPAD photon statistics into 3D scene understanding
5. **S8–S37 (late single)**: Depth signal gradually weakens. The model shifts focus from geometry to **appearance/texture generation**

### Temporal Dynamics (Denoising Steps)

| Phase | Steps | Bit Density | Depth | Variance |
|-------|-------|-------------|-------|----------|
| **ControlNet injection** | t=0 | Peak (0.998) | Low (< 0.1) | Low |
| **Early denoising** | t=4–9 | High (0.99) | **Peak** (0.44) | Building (0.30) |
| **Mid denoising** | t=14 | High (0.98) | Moderate (0.30) | **Peak** (0.42) |
| **Late denoising** | t=19–27 | Decaying (0.93) | Decaying (0.15) | High (0.30–0.40) |

Key insight: **Each target peaks at a different denoising phase**, revealing the temporal structure of the generation process:
- t=0: Conditioning injection
- t=4–9: Scene geometry formation
- t=14+: Commitment to details + uncertainty crystallization

---

## 7. Practical Implications

### For LoRA Placement
The depth signal peaks in **S1–S7** and builds through **J11–J18**. These are the blocks where LoRA adapters would have the most leverage to improve SPAD→depth reasoning. Late single blocks (S20+) are mostly about texture — the pretrained FLUX already handles this well.

### For DPS/PaDIS Guidance Timing
Physics-guided corrections (DPS) are most effective when the model is actively forming geometric structure. Our data shows this happens at **t=4–14** (steps 4–14 of 28). Applying DPS at t=0 is wasteful (model hasn't started reasoning yet); applying at t=24+ is too late (model has committed).

### For ControlNet Architecture
Bit density R² ≈ 0.998 throughout the joint blocks confirms the ControlNet successfully injects SPAD measurements into the early layers. The information persists deep into the network (even S37 retains R² ≈ 0.97).

### For Understanding Model Uncertainty
The fact that cross-seed variance is linearly decodable (R² ≈ 0.42 in joint blocks) means the model **internally represents its own uncertainty** — it "knows" where it's guessing. This could be leveraged for:
- Adaptive sampling (more steps in uncertain regions)
- Confidence-weighted loss functions
- Active sensing (requesting additional SPAD frames for uncertain areas)

---

## 8. Experiments In Progress

### Experiment 3: Control Baseline (No LoRA)
- **Status**: 170/776 samples extracted (22%), ~7 hours remaining
- **What it tests**: Does LoRA fine-tuning change what information is linearly decodable?
- **Why it matters**: If the base FLUX + ControlNet already encodes depth equally well, then LoRA is only helping with *generation quality*, not *understanding*. If the control has lower depth R², LoRA is genuinely teaching the model to better reason about SPAD→3D geometry.
- **Expected outcome**: Bit density should be similar (it's in the conditioning). Depth and variance may differ — this is the key comparison.
- **After extraction completes**: Probes will be trained with identical hyperparameters and a delta analysis (main minus control R² per block×timestep) will quantify LoRA's contribution.

---

## 9. File Structure

```
probing_results/                    ← Experiment 1: 100 samples, 10 sparse blocks
├── activations/                    ←   Extracted features (joint_{0,4,9,14,18}, single_{0,9,19,28,37})
├── targets.json                    ←   Ground truth targets (bit_density, depth, variance)
└── probes/
    ├── probing_results.json        ←   All R² and Pearson r values (global + spatial)
    └── *.png                       ←   Heatmaps, line plots, AC3D curves

probing_results_allblocks/          ← Experiment 2: 776 samples, all 57 blocks
├── activations/                    ←   Features for all 57 blocks × 7 timesteps
├── targets.json                    ←   Same targets, larger dataset
└── probes/
    ├── probing_results.json        ←   All R² and Pearson r (global only, 399 entries)
    └── *.png                       ←   Full 57-block heatmaps, temporal plots

probing_results_control/            ← Experiment 3: 776 samples, all 57 blocks, NO LoRA
├── activations/                    ←   IN PROGRESS (22% complete)
└── targets.json                    ←   Same targets (copied from Experiment 2)
```

---

## 10. Generated Figures

All figures are in `probing_results_allblocks/probes/`:

| Figure | Description |
|--------|-------------|
| `allblocks_heatmap_bit_density.png` | 57×7 heatmap: R² for every block × timestep |
| `allblocks_heatmap_depth.png` | Same for depth — shows S1–S7 hotspot |
| `allblocks_heatmap_variance.png` | Same for variance — shows joint block concentration |
| `temporal_bit_density.png` | R² vs denoising step for selected blocks |
| `temporal_depth.png` | Shows depth peaking at t=4–9 |
| `temporal_variance.png` | Shows variance peaking at t=14–27 |
| `ac3d_curve_bit_density.png` | AC3D-style: best R² per block (block ordering on x-axis) |
| `ac3d_curve_depth.png` | Shows depth peak in early single blocks |
| `ac3d_curve_variance.png` | Shows variance peak in joint blocks |
| `multi_target_all_blocks,_776_samples.png` | All 3 targets overlaid at their best timesteps |
| `comparison_best_timestep.png` | Side-by-side best-timestep comparison |

Spatial probing figures are in `probing_results/probes/`:

| Figure | Description |
|--------|-------------|
| `heatmap_spatial_bit_density.png` | 10×7 spatial R² heatmap |
| `heatmap_spatial_depth.png` | Shows spatial depth >> global depth |
| `heatmap_spatial_variance.png` | Shows spatial variance dramatically better |

---

## Appendix A: Full Heatmap Tables — All-Blocks Global (Experiment 2)

### A.1 Bit Density R² (Joint Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.120 | +0.350 | +0.550 | +0.668 | +0.736 | +0.757 | +0.753 | **+0.757** |
| J1 | +0.993 | +0.996 | +0.996 | +0.995 | +0.992 | +0.978 | +0.945 | **+0.996** |
| J2 | +0.993 | +0.996 | +0.996 | +0.996 | +0.991 | +0.975 | +0.944 | **+0.996** |
| J3 | +0.993 | +0.995 | +0.995 | +0.994 | +0.988 | +0.970 | +0.942 | **+0.995** |
| J4 | +0.992 | +0.993 | +0.993 | +0.992 | +0.983 | +0.957 | +0.924 | **+0.993** |
| J5 | +0.996 | +0.997 | +0.997 | +0.994 | +0.987 | +0.960 | +0.922 | **+0.997** |
| J6 | +0.998 | +0.997 | +0.997 | +0.994 | +0.987 | +0.964 | +0.925 | **+0.998** |
| J7 | +0.998 | +0.997 | +0.996 | +0.993 | +0.984 | +0.963 | +0.929 | **+0.998** |
| J8 | +0.998 | +0.995 | +0.992 | +0.988 | +0.980 | +0.956 | +0.924 | **+0.998** |
| J9 | +0.997 | +0.994 | +0.988 | +0.983 | +0.975 | +0.950 | +0.915 | **+0.997** |
| J10 | +0.997 | +0.993 | +0.987 | +0.981 | +0.972 | +0.947 | +0.908 | **+0.997** |
| J11 | +0.996 | +0.993 | +0.986 | +0.980 | +0.966 | +0.946 | +0.904 | **+0.996** |
| J12 | +0.995 | +0.991 | +0.986 | +0.980 | +0.966 | +0.952 | +0.912 | **+0.995** |
| J13 | +0.995 | +0.989 | +0.983 | +0.977 | +0.959 | +0.946 | +0.911 | **+0.995** |
| J14 | +0.993 | +0.984 | +0.978 | +0.971 | +0.952 | +0.938 | +0.913 | **+0.993** |
| J15 | +0.991 | +0.981 | +0.971 | +0.966 | +0.947 | +0.932 | +0.914 | **+0.991** |
| J16 | +0.988 | +0.976 | +0.966 | +0.960 | +0.942 | +0.919 | +0.911 | **+0.988** |
| J17 | +0.984 | +0.969 | +0.959 | +0.952 | +0.936 | +0.904 | +0.900 | **+0.984** |
| J18 | +0.981 | +0.963 | +0.949 | +0.947 | +0.929 | +0.895 | +0.885 | **+0.981** |

### A.2 Bit Density R² (Single Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| S0 | +0.976 | +0.955 | +0.946 | +0.947 | +0.928 | +0.888 | +0.885 | **+0.976** |
| S1 | +0.969 | +0.945 | +0.931 | +0.934 | +0.924 | +0.885 | +0.879 | **+0.969** |
| S2 | +0.963 | +0.936 | +0.919 | +0.924 | +0.909 | +0.865 | +0.861 | **+0.963** |
| S3 | +0.960 | +0.927 | +0.914 | +0.922 | +0.897 | +0.861 | +0.862 | **+0.960** |
| S4 | +0.957 | +0.920 | +0.910 | +0.911 | +0.894 | +0.857 | +0.854 | **+0.957** |
| S5 | +0.952 | +0.916 | +0.911 | +0.912 | +0.883 | +0.855 | +0.856 | **+0.952** |
| S6 | +0.944 | +0.912 | +0.906 | +0.909 | +0.881 | +0.843 | +0.855 | **+0.944** |
| S7 | +0.946 | +0.913 | +0.909 | +0.911 | +0.882 | +0.847 | +0.845 | **+0.946** |
| S8 | +0.943 | +0.915 | +0.914 | +0.910 | +0.886 | +0.853 | +0.848 | **+0.943** |
| S9 | +0.937 | +0.913 | +0.915 | +0.910 | +0.883 | +0.849 | +0.856 | **+0.937** |
| S10 | +0.934 | +0.915 | +0.914 | +0.906 | +0.879 | +0.850 | +0.845 | **+0.934** |
| S11 | +0.935 | +0.913 | +0.915 | +0.906 | +0.892 | +0.855 | +0.854 | **+0.935** |
| S12 | +0.940 | +0.919 | +0.908 | +0.901 | +0.891 | +0.854 | +0.854 | **+0.940** |
| S13 | +0.939 | +0.918 | +0.914 | +0.907 | +0.889 | +0.856 | +0.849 | **+0.939** |
| S14 | +0.944 | +0.923 | +0.920 | +0.914 | +0.892 | +0.866 | +0.851 | **+0.944** |
| S15 | +0.946 | +0.930 | +0.924 | +0.922 | +0.901 | +0.873 | +0.854 | **+0.946** |
| S16 | +0.947 | +0.936 | +0.929 | +0.923 | +0.912 | +0.878 | +0.849 | **+0.947** |
| S17 | +0.945 | +0.931 | +0.926 | +0.920 | +0.900 | +0.866 | +0.863 | **+0.945** |
| S18 | +0.946 | +0.933 | +0.927 | +0.922 | +0.900 | +0.860 | +0.846 | **+0.946** |
| S19 | +0.949 | +0.939 | +0.932 | +0.919 | +0.896 | +0.857 | +0.835 | **+0.949** |
| S20 | +0.951 | +0.947 | +0.937 | +0.930 | +0.907 | +0.856 | +0.829 | **+0.951** |
| S21 | +0.946 | +0.942 | +0.934 | +0.927 | +0.904 | +0.852 | +0.825 | **+0.946** |
| S22 | +0.949 | +0.943 | +0.936 | +0.927 | +0.906 | +0.854 | +0.826 | **+0.949** |
| S23 | +0.949 | +0.945 | +0.932 | +0.927 | +0.901 | +0.861 | +0.828 | **+0.949** |
| S24 | +0.951 | +0.943 | +0.934 | +0.930 | +0.913 | +0.874 | +0.831 | **+0.951** |
| S25 | +0.955 | +0.947 | +0.942 | +0.934 | +0.918 | +0.874 | +0.801 | **+0.955** |
| S26 | +0.952 | +0.949 | +0.949 | +0.936 | +0.921 | +0.876 | +0.801 | **+0.952** |
| S27 | +0.959 | +0.955 | +0.953 | +0.943 | +0.925 | +0.877 | +0.814 | **+0.959** |
| S28 | +0.961 | +0.956 | +0.949 | +0.941 | +0.931 | +0.883 | +0.817 | **+0.961** |
| S29 | +0.959 | +0.953 | +0.952 | +0.940 | +0.928 | +0.887 | +0.816 | **+0.959** |
| S30 | +0.961 | +0.953 | +0.954 | +0.940 | +0.931 | +0.889 | +0.818 | **+0.961** |
| S31 | +0.961 | +0.951 | +0.949 | +0.941 | +0.933 | +0.890 | +0.808 | **+0.961** |
| S32 | +0.963 | +0.958 | +0.952 | +0.945 | +0.932 | +0.886 | +0.812 | **+0.963** |
| S33 | +0.963 | +0.956 | +0.956 | +0.948 | +0.937 | +0.892 | +0.807 | **+0.963** |
| S34 | +0.964 | +0.954 | +0.955 | +0.946 | +0.935 | +0.897 | +0.825 | **+0.964** |
| S35 | +0.966 | +0.952 | +0.953 | +0.948 | +0.941 | +0.910 | +0.827 | **+0.966** |
| S36 | +0.966 | +0.957 | +0.957 | +0.952 | +0.946 | +0.917 | +0.835 | **+0.966** |
| S37 | +0.968 | +0.959 | +0.958 | +0.954 | +0.947 | +0.919 | +0.832 | **+0.968** |

### A.3 Depth R² (Joint Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.147 | -0.049 | -0.022 | +0.017 | -0.002 | +0.013 | -0.035 | **+0.017** |
| J1 | -0.310 | -0.378 | -0.225 | -0.142 | -0.061 | +0.106 | -0.011 | **+0.106** |
| J2 | -0.333 | -0.452 | -0.193 | -0.054 | -0.009 | +0.086 | +0.037 | **+0.086** |
| J3 | -0.427 | -0.460 | -0.189 | -0.069 | -0.030 | +0.123 | +0.142 | **+0.142** |
| J4 | -0.370 | -0.555 | -0.259 | -0.088 | -0.063 | -0.038 | +0.146 | **+0.146** |
| J5 | -0.331 | -0.333 | -0.202 | -0.201 | -0.132 | -0.260 | +0.080 | **+0.080** |
| J6 | -0.167 | -0.267 | -0.193 | -0.146 | -0.023 | -0.063 | +0.011 | **+0.011** |
| J7 | -0.044 | -0.218 | -0.198 | +0.070 | -0.073 | -0.083 | +0.077 | **+0.077** |
| J8 | -0.012 | -0.118 | -0.065 | -0.067 | +0.001 | -0.048 | +0.202 | **+0.202** |
| J9 | +0.023 | -0.123 | -0.042 | -0.173 | +0.062 | -0.005 | +0.105 | **+0.105** |
| J10 | -0.006 | -0.200 | -0.236 | -0.031 | +0.061 | -0.018 | +0.177 | **+0.177** |
| J11 | +0.007 | -0.220 | -0.184 | +0.251 | +0.295 | +0.096 | +0.141 | **+0.295** |
| J12 | -0.001 | -0.080 | -0.034 | +0.256 | +0.223 | +0.149 | +0.112 | **+0.256** |
| J13 | -0.175 | -0.002 | -0.148 | +0.098 | +0.175 | +0.072 | +0.120 | **+0.175** |
| J14 | -0.235 | +0.013 | -0.194 | +0.146 | +0.232 | -0.028 | +0.153 | **+0.232** |
| J15 | -0.353 | +0.102 | -0.300 | -0.038 | +0.255 | +0.012 | +0.067 | **+0.255** |
| J16 | -0.300 | -0.137 | -0.108 | +0.012 | +0.183 | -0.017 | -0.050 | **+0.183** |
| J17 | -0.494 | -0.142 | -0.043 | -0.007 | +0.211 | -0.065 | -0.029 | **+0.211** |
| J18 | -0.270 | -0.025 | -0.051 | -0.033 | +0.189 | +0.037 | -0.033 | **+0.189** |

### A.4 Depth R² (Single Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| S0 | -0.131 | +0.059 | +0.102 | -0.002 | +0.044 | +0.106 | +0.072 | **+0.106** |
| S1 | -0.126 | +0.420 | +0.272 | +0.084 | +0.105 | +0.148 | +0.050 | **+0.420** |
| S2 | -0.049 | +0.249 | +0.255 | +0.071 | +0.087 | +0.143 | +0.114 | **+0.255** |
| S3 | +0.018 | +0.382 | +0.166 | -0.000 | +0.143 | +0.106 | +0.205 | **+0.382** |
| S4 | -0.029 | +0.437 | +0.372 | -0.037 | +0.164 | +0.151 | +0.144 | **+0.437** |
| S5 | -0.155 | +0.258 | +0.371 | +0.275 | +0.077 | +0.120 | +0.178 | **+0.371** |
| S6 | -0.003 | +0.148 | +0.315 | +0.245 | +0.098 | +0.215 | +0.114 | **+0.315** |
| S7 | +0.035 | +0.214 | +0.409 | +0.233 | +0.064 | +0.177 | +0.018 | **+0.409** |
| S8 | -0.019 | +0.266 | +0.275 | +0.199 | +0.087 | +0.040 | +0.021 | **+0.275** |
| S9 | +0.022 | +0.269 | +0.204 | -0.067 | -0.007 | +0.105 | +0.110 | **+0.269** |
| S10 | -0.155 | +0.004 | -0.122 | -0.113 | -0.056 | -0.026 | +0.140 | **+0.140** |
| S11 | -0.048 | +0.133 | -0.127 | -0.157 | -0.134 | +0.007 | +0.106 | **+0.133** |
| S12 | -0.073 | +0.153 | -0.031 | -0.211 | -0.142 | -0.031 | +0.121 | **+0.153** |
| S13 | -0.095 | +0.157 | -0.048 | -0.122 | -0.146 | +0.017 | +0.050 | **+0.157** |
| S14 | -0.272 | -0.011 | -0.116 | -0.156 | -0.001 | -0.258 | +0.108 | **+0.108** |
| S15 | -0.193 | +0.104 | -0.121 | -0.081 | -0.090 | +0.032 | +0.076 | **+0.104** |
| S16 | -0.048 | +0.106 | -0.258 | -0.115 | -0.030 | +0.073 | +0.114 | **+0.114** |
| S17 | -0.043 | +0.163 | -0.085 | -0.048 | +0.078 | +0.118 | +0.083 | **+0.163** |
| S18 | +0.001 | +0.300 | +0.170 | +0.103 | +0.136 | +0.084 | +0.138 | **+0.300** |
| S19 | +0.059 | +0.289 | +0.069 | +0.011 | +0.221 | +0.077 | +0.158 | **+0.289** |
| S20 | +0.049 | +0.242 | +0.112 | +0.146 | +0.203 | +0.058 | +0.027 | **+0.242** |
| S21 | -0.078 | +0.205 | +0.157 | +0.256 | +0.172 | +0.103 | +0.109 | **+0.256** |
| S22 | -0.035 | +0.282 | +0.106 | +0.297 | +0.042 | +0.147 | +0.080 | **+0.297** |
| S23 | +0.091 | +0.213 | +0.065 | +0.222 | +0.017 | +0.099 | +0.099 | **+0.222** |
| S24 | -0.027 | +0.203 | +0.038 | +0.211 | +0.168 | +0.123 | +0.104 | **+0.211** |
| S25 | -0.053 | +0.206 | -0.069 | +0.146 | +0.193 | +0.091 | +0.133 | **+0.206** |
| S26 | -0.178 | +0.240 | +0.077 | +0.105 | +0.241 | +0.055 | +0.127 | **+0.241** |
| S27 | -0.157 | +0.179 | -0.134 | +0.029 | +0.208 | +0.105 | +0.188 | **+0.208** |
| S28 | -0.111 | +0.193 | +0.013 | +0.138 | +0.178 | +0.035 | +0.151 | **+0.193** |
| S29 | -0.210 | +0.282 | -0.011 | -0.027 | -0.024 | -0.082 | +0.196 | **+0.282** |
| S30 | -0.074 | +0.230 | +0.065 | +0.026 | +0.155 | -0.062 | +0.186 | **+0.230** |
| S31 | -0.043 | +0.183 | +0.063 | -0.087 | +0.086 | +0.043 | +0.063 | **+0.183** |
| S32 | -0.044 | +0.088 | +0.052 | -0.006 | -0.048 | +0.158 | +0.195 | **+0.195** |
| S33 | -0.096 | +0.053 | +0.127 | -0.013 | -0.090 | +0.118 | +0.101 | **+0.127** |
| S34 | +0.048 | +0.024 | -0.118 | -0.016 | +0.044 | +0.160 | +0.089 | **+0.160** |
| S35 | -0.005 | +0.076 | +0.007 | +0.138 | +0.009 | +0.145 | +0.156 | **+0.156** |
| S36 | -0.030 | +0.150 | +0.082 | +0.147 | +0.005 | +0.127 | +0.165 | **+0.165** |
| S37 | -0.036 | +0.160 | +0.060 | +0.142 | +0.009 | +0.146 | +0.178 | **+0.178** |

### A.5 Variance R² (Joint Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.436 | +0.121 | +0.244 | +0.365 | +0.380 | +0.364 | +0.360 | **+0.380** |
| J1 | +0.054 | +0.282 | +0.359 | +0.424 | +0.418 | +0.422 | +0.412 | **+0.424** |
| J2 | -0.054 | +0.181 | +0.305 | +0.345 | +0.361 | +0.384 | +0.374 | **+0.384** |
| J3 | -0.338 | -0.040 | +0.091 | +0.213 | +0.173 | +0.232 | +0.274 | **+0.274** |
| J4 | -0.301 | +0.002 | +0.074 | +0.193 | +0.117 | +0.180 | +0.310 | **+0.310** |
| J5 | +0.060 | +0.054 | +0.093 | +0.183 | +0.114 | +0.108 | +0.317 | **+0.317** |
| J6 | +0.182 | +0.109 | +0.149 | +0.176 | +0.094 | +0.189 | +0.392 | **+0.392** |
| J7 | +0.280 | +0.109 | +0.078 | +0.205 | +0.172 | +0.306 | +0.403 | **+0.403** |
| J8 | +0.197 | -0.043 | +0.055 | +0.128 | +0.217 | +0.229 | +0.304 | **+0.304** |
| J9 | +0.269 | -0.012 | +0.064 | +0.075 | +0.124 | +0.283 | +0.234 | **+0.283** |
| J10 | +0.195 | +0.027 | +0.127 | +0.163 | +0.104 | +0.269 | +0.274 | **+0.274** |
| J11 | +0.205 | +0.094 | +0.040 | -0.006 | +0.198 | +0.370 | +0.276 | **+0.370** |
| J12 | +0.140 | +0.175 | +0.015 | +0.032 | +0.223 | +0.418 | +0.280 | **+0.418** |
| J13 | +0.215 | +0.159 | +0.127 | -0.016 | +0.238 | +0.288 | +0.309 | **+0.309** |
| J14 | +0.195 | +0.170 | +0.089 | -0.029 | +0.270 | +0.334 | +0.189 | **+0.334** |
| J15 | +0.155 | +0.061 | +0.155 | +0.100 | +0.353 | +0.349 | +0.202 | **+0.353** |
| J16 | +0.122 | -0.126 | +0.135 | +0.079 | +0.360 | +0.340 | +0.232 | **+0.360** |
| J17 | +0.183 | +0.112 | +0.127 | +0.109 | +0.238 | +0.331 | +0.212 | **+0.331** |
| J18 | +0.152 | +0.046 | +0.067 | +0.015 | +0.205 | +0.269 | +0.193 | **+0.269** |

### A.6 Variance R² (Single Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| S0 | +0.116 | +0.044 | +0.052 | -0.009 | +0.154 | +0.229 | +0.217 | **+0.229** |
| S1 | +0.018 | -0.002 | +0.035 | +0.084 | +0.131 | +0.152 | +0.071 | **+0.152** |
| S2 | +0.190 | -0.001 | -0.104 | -0.032 | +0.162 | +0.167 | +0.080 | **+0.190** |
| S3 | +0.161 | -0.076 | +0.113 | +0.136 | +0.161 | +0.187 | +0.077 | **+0.187** |
| S4 | +0.117 | +0.079 | +0.206 | +0.244 | +0.239 | +0.156 | +0.004 | **+0.244** |
| S5 | +0.085 | +0.246 | +0.186 | +0.028 | +0.179 | +0.166 | -0.006 | **+0.246** |
| S6 | +0.222 | +0.306 | +0.026 | -0.379 | +0.097 | -0.039 | +0.051 | **+0.306** |
| S7 | +0.290 | +0.326 | -0.045 | -0.259 | +0.015 | +0.119 | +0.094 | **+0.326** |
| S8 | +0.143 | +0.098 | -0.052 | -0.074 | +0.000 | +0.057 | +0.209 | **+0.209** |
| S9 | +0.188 | +0.088 | +0.027 | -0.241 | -0.085 | +0.104 | +0.115 | **+0.188** |
| S10 | +0.324 | +0.178 | +0.151 | -0.192 | -0.025 | -0.041 | +0.006 | **+0.324** |
| S11 | +0.044 | +0.110 | +0.047 | -0.138 | -0.126 | -0.014 | +0.102 | **+0.110** |
| S12 | +0.096 | +0.128 | +0.098 | -0.016 | -0.016 | -0.080 | +0.112 | **+0.128** |
| S13 | +0.177 | +0.041 | +0.016 | -0.020 | +0.106 | +0.120 | +0.153 | **+0.177** |
| S14 | +0.215 | +0.057 | +0.003 | +0.010 | -0.052 | +0.077 | +0.138 | **+0.215** |
| S15 | +0.177 | +0.028 | -0.110 | +0.047 | -0.034 | +0.204 | +0.277 | **+0.277** |
| S16 | +0.268 | -0.012 | +0.025 | +0.043 | -0.054 | -0.067 | +0.247 | **+0.268** |
| S17 | +0.314 | +0.044 | +0.033 | +0.031 | +0.069 | +0.028 | +0.170 | **+0.314** |
| S18 | +0.303 | +0.167 | +0.033 | +0.114 | -0.013 | -0.017 | +0.062 | **+0.303** |
| S19 | +0.043 | +0.116 | +0.028 | +0.071 | -0.074 | +0.040 | +0.132 | **+0.132** |
| S20 | +0.058 | +0.170 | +0.058 | -0.015 | +0.031 | -0.039 | -0.057 | **+0.170** |
| S21 | +0.047 | +0.084 | +0.166 | +0.095 | -0.091 | -0.072 | -0.017 | **+0.166** |
| S22 | +0.163 | +0.096 | +0.021 | -0.050 | +0.021 | +0.001 | +0.026 | **+0.163** |
| S23 | +0.205 | +0.132 | +0.178 | +0.166 | +0.193 | +0.184 | +0.067 | **+0.205** |
| S24 | +0.272 | +0.004 | +0.135 | +0.030 | +0.002 | +0.149 | +0.163 | **+0.272** |
| S25 | +0.201 | -0.119 | -0.079 | -0.016 | +0.164 | +0.167 | +0.229 | **+0.229** |
| S26 | +0.205 | +0.062 | -0.010 | +0.031 | +0.062 | +0.143 | +0.127 | **+0.205** |
| S27 | +0.204 | +0.000 | +0.038 | +0.085 | +0.159 | +0.208 | +0.112 | **+0.208** |
| S28 | +0.208 | +0.105 | +0.050 | +0.133 | +0.173 | +0.237 | +0.106 | **+0.237** |
| S29 | +0.235 | +0.101 | -0.008 | +0.079 | +0.133 | +0.171 | +0.024 | **+0.235** |
| S30 | +0.217 | +0.171 | +0.054 | +0.002 | +0.093 | +0.094 | -0.019 | **+0.217** |
| S31 | +0.197 | +0.153 | +0.124 | -0.036 | +0.136 | +0.073 | +0.042 | **+0.197** |
| S32 | +0.123 | +0.124 | +0.061 | -0.009 | +0.093 | +0.065 | +0.103 | **+0.124** |
| S33 | +0.134 | +0.086 | +0.014 | -0.033 | +0.057 | +0.077 | +0.076 | **+0.134** |
| S34 | +0.096 | +0.132 | +0.069 | -0.057 | -0.009 | +0.168 | +0.130 | **+0.168** |
| S35 | +0.097 | +0.104 | +0.017 | -0.026 | +0.015 | +0.146 | +0.132 | **+0.146** |
| S36 | +0.119 | +0.130 | +0.071 | -0.008 | +0.057 | +0.159 | +0.212 | **+0.212** |
| S37 | +0.110 | +0.124 | +0.080 | -0.046 | +0.015 | +0.164 | +0.222 | **+0.222** |

---

## Appendix B: Spatial Probing Full Tables (Experiment 1)

### B.1 Spatial Bit Density R²

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | 0.022 | 0.022 | 0.159 | 0.448 | 0.649 | 0.694 | 0.687 | **0.694** |
| J4 | 0.970 | 0.970 | 0.969 | 0.965 | 0.961 | 0.946 | 0.914 | **0.970** |
| J9 | 0.988 | 0.990 | 0.987 | 0.982 | 0.972 | 0.943 | 0.904 | **0.990** |
| J14 | 0.990 | 0.987 | 0.983 | 0.977 | 0.960 | 0.926 | 0.893 | **0.990** |
| J18 | 0.986 | 0.978 | 0.972 | 0.964 | 0.942 | 0.898 | 0.868 | **0.986** |
| S0 | 0.986 | 0.976 | 0.969 | 0.959 | 0.937 | 0.893 | 0.863 | **0.986** |
| S9 | 0.973 | 0.966 | 0.954 | 0.942 | 0.918 | 0.871 | 0.834 | **0.973** |
| S19 | 0.972 | 0.962 | 0.954 | 0.941 | 0.910 | 0.856 | 0.820 | **0.972** |
| S28 | 0.978 | 0.963 | 0.953 | 0.940 | 0.912 | 0.849 | 0.800 | **0.978** |
| S37 | 0.986 | 0.971 | 0.967 | 0.955 | 0.933 | 0.880 | 0.806 | **0.986** |

### B.2 Spatial Depth R²

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.023 | -0.026 | 0.002 | 0.049 | 0.100 | 0.114 | 0.119 | **0.119** |
| J4 | 0.151 | 0.213 | 0.211 | 0.254 | 0.292 | 0.296 | 0.286 | **0.296** |
| J9 | 0.195 | 0.258 | 0.259 | 0.380 | 0.448 | 0.424 | 0.329 | **0.448** |
| J14 | 0.202 | 0.361 | 0.475 | 0.544 | 0.567 | 0.527 | 0.422 | **0.567** |
| J18 | 0.217 | 0.424 | 0.531 | 0.606 | 0.603 | 0.589 | 0.476 | **0.606** |
| S0 | 0.229 | 0.482 | 0.565 | 0.610 | 0.601 | 0.588 | 0.476 | **0.610** |
| S9 | 0.359 | 0.567 | 0.631 | 0.648 | 0.619 | 0.605 | 0.526 | **0.648** |
| S19 | 0.367 | 0.568 | 0.601 | 0.598 | 0.580 | 0.552 | 0.425 | **0.601** |
| S28 | 0.366 | 0.578 | 0.586 | 0.576 | 0.535 | 0.505 | 0.374 | **0.586** |
| S37 | 0.385 | 0.564 | 0.564 | 0.548 | 0.533 | 0.498 | 0.368 | **0.564** |

### B.3 Spatial Variance R²

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.134 | -0.132 | -0.052 | 0.066 | 0.148 | 0.211 | 0.222 | **0.222** |
| J4 | 0.040 | 0.050 | 0.146 | 0.230 | 0.304 | 0.334 | 0.364 | **0.364** |
| J9 | 0.087 | 0.161 | 0.234 | 0.327 | 0.377 | 0.326 | 0.318 | **0.377** |
| J14 | 0.202 | 0.275 | 0.309 | 0.370 | 0.368 | 0.329 | 0.331 | **0.370** |
| J18 | 0.179 | 0.362 | 0.366 | 0.409 | 0.372 | 0.384 | 0.376 | **0.409** |
| S0 | 0.145 | 0.355 | 0.335 | 0.387 | 0.360 | 0.391 | 0.379 | **0.391** |
| S9 | 0.132 | 0.364 | 0.424 | 0.433 | 0.385 | 0.380 | 0.403 | **0.433** |
| S19 | 0.235 | 0.373 | 0.408 | 0.397 | 0.398 | 0.370 | 0.365 | **0.408** |
| S28 | 0.181 | 0.363 | 0.385 | 0.381 | 0.403 | 0.368 | 0.374 | **0.403** |
| S37 | 0.198 | 0.336 | 0.385 | 0.383 | 0.379 | 0.367 | 0.388 | **0.388** |
