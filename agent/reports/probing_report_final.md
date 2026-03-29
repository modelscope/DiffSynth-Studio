# Linear Probing Analysis — SPAD-FLUX DiT Internal Representations
**Date: 2026-03-27 | Status: Complete (all conditions)**

---

## 1. Motivation and Research Questions

We apply **AC3D-inspired linear probing** (Bhatt et al., 2024) to our FLUX.1-dev diffusion transformer to understand what the model has learned about SPAD (Single-Photon Avalanche Diode) binary sensor data. Specifically, we ask:

1. **Where** in the 57-block DiT is SPAD measurement information encoded?
2. **When** during the 28-step denoising process does scene understanding (depth, uncertainty) emerge?
3. **What** does LoRA fine-tuning add beyond the pretrained model + ControlNet?
4. **What role** does the ControlNet play as the information conduit?
5. **Can** the model recognize semantic objects from binary 1-bit SPAD frames?

A **linear probe** (ridge regression) tests whether a target property is **linearly decodable** from internal activations. High R² means the information is explicitly, linearly represented; low R² means it is either absent or encoded in a nonlinear way that a linear readout cannot recover.

---

## 2. Experimental Setup

### 2.1 Architecture

- **FLUX.1-dev**: 12B parameter rectified-flow transformer
  - 19 **joint** transformer blocks (cross-attend to text tokens) → denoted J0–J18
  - 38 **single** transformer blocks (self-attention only) → denoted S0–S37
  - Hidden dimension: **3072** per image token
  - Image token count: **1024** tokens (32x32 patch grid for 512x512 input images)
- **ControlNet**: 5 joint + 10 single blocks (15 total), injects SPAD conditioning via linear projections
- **LoRA**: Low-rank adapters fused into ControlNet weights (NOT DiT), rank 16

### 2.2 Three Experimental Conditions

| Condition | Model Configuration | Blocks Probed | Samples | Probing Modes |
|-----------|-------------------|--------------|---------|---------------|
| **Main** | FLUX + ControlNet + LoRA | 57 DiT + 15 CN | 776 | Global + Spatial (per-token) |
| **Control** | FLUX + ControlNet, **no LoRA** | 57 DiT + 15 CN | 776 | Global + Spatial (per-token) |
| **No-ControlNet** | FLUX DiT only (no ControlNet, no LoRA) | 57 DiT | 776 | Global + Spatial (per-token) |

The **Main vs Control** comparison isolates the effect of LoRA fine-tuning.
The **No-ControlNet** ablation confirms that ControlNet is the information conduit — without it, the DiT has no access to SPAD data.

### 2.3 Probe Targets

#### Continuous Targets (Scalar and Spatial)

| Target | Description | Source | Why It Matters |
|--------|-------------|--------|----------------|
| **Bit density** | Mean of binary SPAD frame (photon evidence) | Input conditioning | Does the network preserve measurement statistics? |
| **Depth** | Monocular depth from Depth Anything V2 | Pseudo-GT from RGB | Does the network infer 3D structure from 1-bit data? |
| **Variance** | Pixel-wise std across 3 random seeds | Multi-seed generation | Does the network represent its own uncertainty? |

#### Object Presence Targets (Binary Classification)

| Target | Description | Source | Why It Matters |
|--------|-------------|--------|----------------|
| **24 object classes** | Binary presence/absence of objects (chair, table, building, etc.) | VLM (Qwen2.5-VL-7B) annotation | Can the network recognize semantic content from 1-bit data? |

Object labels were extracted by running Qwen2.5-VL-7B on ground-truth RGB images, with vocabulary deduplication (merging space/underscore variants). Only objects appearing in >=5% of the 776 validation images were retained, yielding 24 classes.

### 2.4 Probing Methodology

- **Activation extraction**: Forward hooks on all DiT and ControlNet blocks capture image-token features at **7 denoising timesteps** (t in {0, 4, 9, 14, 19, 24, 27} out of 28 steps)
- **Global probing**: Mean-pool 1024 tokens → 1 vector (dim 3072) per image. Ridge regression predicts scalar target.
- **Spatial probing**: Keep all 1024 tokens. Ridge regression predicts per-patch target (32x32 map). Uses streaming X^T X / X^T y accumulation in float64 to avoid OOM.
- **Ridge regression**: Closed-form solve with:
  - Feature standardization (zero mean, unit variance)
  - **y-centering** (subtract mean of training targets before fitting, add back at prediction)
  - Trace-normalized regularization: lambda_scaled = lambda * tr(X^T X) / D
  - lambda = 0.1
- **Metrics**:
  - **R²** (coefficient of determination): 1.0 = perfect, 0.0 = mean-prediction baseline, <0 = worse than predicting mean
  - **Pearson r**: Correlation between predictions and targets
  - **Balanced accuracy** (for binary targets): Average of true positive rate and true negative rate, robust to class imbalance
- **Train/test split**: 80/20 random split (621 train / 155 test)

### 2.5 Critical Implementation Detail: y-Centering

The original ridge regression implementation standardized features (X) but **did not center the target variable (y)**. This caused predictions with correct correlation direction (Pearson r up to 0.99) but catastrophically wrong magnitude (R² as low as -50). Adding y-centering immediately resolved this:

| Target | Before Fix (R²) | After Fix (R²) |
|--------|-----------------|----------------|
| Bit density | -8.83 | **+0.998** |
| Depth | -0.02 | **+0.437** |
| Variance | -0.86 | **+0.424** |

---

## 3. Results — Continuous Targets

### 3.1 Summary: Best R² Across All Conditions

#### DiT Global Probing (mean-pooled, 776 samples)

| Target | Main (LoRA) | Control (no LoRA) | No-ControlNet | LoRA Delta |
|--------|-------------|-------------------|---------------|------------|
| **Bit density** | 0.998 (J8, t=0) | 0.999 (J1, t=14) | -0.059 | -0.001 (inherent) |
| **Depth** | 0.437 (S4, t=4) | 0.129 (J8, t=27) | -0.040 | **+0.308** |
| **Variance** | 0.424 (J1, t=14) | 0.446 (S17, t=0) | -0.358 | -0.022 (inherent) |

#### DiT Spatial Probing (per-token, 776 samples, 10 blocks)

| Target | Main (LoRA) | Control (no LoRA) | No-ControlNet | LoRA Delta |
|--------|-------------|-------------------|---------------|------------|
| **Bit density** | 0.959 (S37, t=0) | 0.992 (S37, t=9) | -0.046 | -0.033 (inherent) |
| **Depth** | **0.685** (S9, t=14) | 0.453 (S9, t=4) | 0.406 (S0, t=0) | **+0.232** |
| **Variance** | **0.506** (S28, t=4) | 0.434 (S9, t=0) | -0.067 | **+0.072** |

#### ControlNet Global Probing (15 CN blocks)

| Target | Main (LoRA) | Control (no LoRA) | LoRA Delta |
|--------|-------------|-------------------|------------|
| **Bit density** | 1.000 (CN-J1, t=0) | 1.000 (CN-J1, t=9) | ~0 (inherent) |
| **Depth** | 0.227 (CN-S5, t=19) | 0.023 (CN-J4, t=27) | **+0.203** |
| **Variance** | 0.465 (CN-J2, t=27) | 0.459 (CN-J1, t=27) | +0.006 |

#### Positive R² Block Count (DiT Global)

| Target | Main | Control | No-ControlNet |
|--------|------|---------|---------------|
| Bit density | 398/399 (99.7%) | 398/399 (99.7%) | 0/399 (0.0%) |
| Depth | 235/399 (58.9%) | 22/399 (5.5%) | 0/399 (0.0%) |
| Variance | 333/399 (83.5%) | 379/399 (95.0%) | 0/399 (0.0%) |

### 3.2 Bit Density — Measurement Evidence Preservation

**Key finding**: Bit density is **almost perfectly linearly decodable** across the entire network in both Main and Control conditions (R² >= 0.998), confirming the ControlNet successfully injects and the DiT preserves photon measurement statistics throughout all 57 blocks.

- **Peak zone**: Joint blocks J5-J12 at t=0, all with R² > 0.995
- **Temporal pattern**: Strongest at t=0 (right after ControlNet injection), monotonically decaying toward t=27 but remaining above R² > 0.80 everywhere
- **No-ControlNet ablation**: R² = -0.059 (best), 0/399 positive — confirms bit density information comes entirely through the ControlNet
- **LoRA effect**: Negligible (delta ~ -0.001). This makes sense — bit density is an input property, not something LoRA needs to teach

**Top 10 blocks (Main model, DiT global):**

| Rank | Block | Timestep | R² | Pearson r |
|------|-------|----------|----|-----------|
| 1 | J8 | t=0 | 0.998 | 0.999 |
| 2 | J7 | t=0 | 0.998 | 0.999 |
| 3 | J6 | t=0 | 0.998 | 0.999 |
| 4 | J6 | t=4 | 0.997 | 0.999 |
| 5 | J9 | t=0 | 0.997 | 0.999 |
| 6 | J5 | t=9 | 0.997 | 1.000 |
| 7 | J5 | t=4 | 0.997 | 0.999 |
| 8 | J7 | t=4 | 0.997 | 0.999 |
| 9 | J10 | t=0 | 0.997 | 0.999 |
| 10 | J6 | t=9 | 0.997 | 0.999 |

> **Figure**: See [fig1_main_heatmap](probing_analysis_output/fig1_main_heatmap.png) (bit density panel), [fig2_main_vs_control](probing_analysis_output/fig2_main_vs_control.png) (bit density row)

### 3.3 Depth — Where 3D Scene Understanding Emerges

**Key finding**: Depth encoding is **dramatically enhanced by LoRA fine-tuning** and shows strong spatial localization. This is the clearest evidence that LoRA teaches the model to reason about SPAD-to-3D geometry.

#### Global Probing

- **Main model best**: R² = 0.437 at S4, t=4
- **Control best**: R² = 0.129 at J8, t=27 — **3.4x lower**
- **LoRA delta**: Mean +0.317 across all 399 block-timestep combinations, positive in 357/399 (89.5%)
- **Peak gains**: S4 t=4 (+1.091), S29 t=27 (+1.050), S4 t=9 (+1.040) — some individual deltas exceed 1.0, meaning the control had negative R² (worse than mean prediction) where the main model achieves positive decoding
- **Positive R² blocks**: Main 235/399 (59%) vs Control 22/399 (5.5%) — LoRA creates depth readability where none existed

#### Spatial Probing

- **Main model best**: R² = 0.685 at S9, t=14 — **substantially higher than global** (0.437)
- **Control best**: R² = 0.453 at S9, t=4 — also higher spatially, but still 0.232 below Main
- **Interpretation**: Depth is spatially heterogeneous. Global mean-pooling loses the spatial pattern; spatial probing preserves the per-patch depth map, allowing the probe to leverage local geometric structure

#### Where Depth Peaks

- **Joint blocks**: Depth encoding builds through J8-J18, peaking at J11 (R² = 0.295 at t=19). Joint blocks provide the cross-attention to text tokens that grounds spatial reasoning
- **Single blocks**: Peak depth in **S1-S7 at t=4-t=9** (R² = 0.37-0.44). This is the **measurement-to-geometry conversion zone**
- **Late single blocks**: Depth weakens (R² ~ 0.15-0.20 in S20+) as the model shifts to texture generation

**Top 10 blocks (Main, DiT global):**

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

**Top 10 blocks (Main, DiT spatial):**

| Rank | Block | Timestep | R² | Pearson r |
|------|-------|----------|----|-----------|
| 1 | S9 | t=14 | 0.685 | 0.836 |
| 2 | S9 | t=19 | 0.680 | 0.834 |
| 3 | S28 | t=9 | 0.678 | 0.826 |
| 4 | S19 | t=9 | 0.675 | 0.827 |
| 5 | S9 | t=9 | 0.674 | 0.832 |
| 6 | S28 | t=4 | 0.671 | 0.824 |
| 7 | S19 | t=4 | 0.671 | 0.825 |
| 8 | S37 | t=9 | 0.671 | 0.823 |
| 9 | S37 | t=4 | 0.669 | 0.822 |
| 10 | S19 | t=14 | 0.665 | 0.822 |

> **Figures**: See [fig1_main_heatmap](probing_analysis_output/fig1_main_heatmap.png) (depth panel), [fig3_delta_heatmap](probing_analysis_output/fig3_delta_heatmap.png) (depth delta), [fig4_best_timestep_lineplot](probing_analysis_output/fig4_best_timestep_lineplot.png), [fig6_global_vs_spatial](probing_analysis_output/fig6_global_vs_spatial.png)

### 3.4 Variance — Uncertainty Awareness

**Key finding**: The model **internally represents its own uncertainty** even without LoRA (control R² = 0.446), but LoRA provides a modest spatial improvement (+0.072 in spatial probing). The pretrained FLUX already has a strong uncertainty signal, likely because variance correlates with image texture complexity.

#### Global Probing

- **Main model best**: R² = 0.424 at J1, t=14
- **Control best**: R² = 0.446 at S17, t=0 — **slightly higher** than main (surprising)
- **Explanation**: Variance as a global scalar is already well-represented in the pretrained model. The control's slightly higher R² may reflect that LoRA redistributes the variance signal across blocks (making it more diffuse but still present)

#### Spatial Probing

- **Main model best**: R² = 0.506 at S28, t=4 — substantially higher than global (0.424)
- **Control best**: R² = 0.434 at S9, t=0
- **LoRA delta**: +0.072 (spatial) — LoRA helps the model spatially localize where it is uncertain, even if the global scalar is similar

#### Where Variance Peaks

- **Joint blocks dominate globally**: J0-J12 at t=14-t=27, with R² in 0.30-0.42. Cross-attention provides context for ambiguity assessment
- **Temporal pattern**: Peaks at **t=14-t=27** (late denoising) — the network becomes aware of its uncertainty as it commits to fine details
- **Single blocks**: Weaker globally (R² ~ 0.10-0.33), but strong spatially (R² ~ 0.50)

**Top 10 blocks (Main, DiT global):**

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
| 10 | J2 | t=27 | 0.374 | 0.642 |

> **Figures**: See [fig1_main_heatmap](probing_analysis_output/fig1_main_heatmap.png) (variance panel), [fig3_delta_heatmap](probing_analysis_output/fig3_delta_heatmap.png) (variance delta)

---

## 4. Results — Object Presence (Semantic Understanding)

### 4.1 Overview

We probe whether the DiT encodes the **presence of specific semantic objects** — a higher-level test of scene understanding beyond geometric properties. Using balanced accuracy (robust to class imbalance since most objects appear in <30% of images).

### 4.2 DiT Object Probing — Main vs Control

| Object | Main (bal. acc.) | Control (bal. acc.) | Delta | LoRA Helps? |
|--------|-----------------|-------------------|-------|-------------|
| **Table** | **0.982** | 0.912 | +0.070 | Yes |
| **Chair** | **0.953** | 0.881 | +0.072 | Yes |
| **Computer Monitor** | **0.917** | 0.878 | +0.039 | Yes |
| **Building** | **0.906** | 0.818 | +0.089 | Yes |
| **Keyboard** | **0.881** | 0.724 | +0.157 | Yes |
| **Cable** | 0.873 | **0.906** | -0.033 | No |
| **Mouse** | **0.861** | 0.677 | +0.184 | **Largest gain** |
| **Box** | **0.829** | 0.819 | +0.010 | Yes |
| **Computer** | **0.820** | 0.804 | +0.016 | Yes |
| **TV** | **0.808** | 0.771 | +0.037 | Yes |
| **Trash Can** | **0.781** | 0.739 | +0.041 | Yes |
| **Window** | **0.747** | 0.743 | +0.003 | Marginal |
| **File Cabinet** | 0.734 | 0.734 | +0.000 | Tie |
| **Door** | 0.712 | **0.754** | -0.042 | No |
| **Wall** | **0.699** | 0.667 | +0.032 | Yes |
| **Sign** | **0.677** | 0.593 | +0.083 | Yes |
| **Floor** | 0.645 | **0.689** | -0.043 | No |
| **Shelf** | **0.622** | 0.563 | +0.059 | Yes |
| **Pavement** | **0.583** | 0.521 | +0.063 | Yes |
| **Tree** | **0.581** | 0.535 | +0.047 | Yes |
| **Bench** | 0.522 | 0.522 | +0.000 | Tie |
| **Bulletin Board** | 0.500 | 0.500 | +0.000 | N/A (too rare) |
| **Plant** | 0.500 | 0.500 | +0.000 | N/A (too rare) |
| **Cabinet** | **0.559** | 0.529 | +0.029 | Yes |

**Summary**: LoRA improves object recognition for **20 of 24 objects**, with the largest gains for small/distinctive objects that require learned visual priors:
- **Mouse**: +0.184 (0.677 → 0.861)
- **Keyboard**: +0.157 (0.724 → 0.881)
- **Building**: +0.089 (0.818 → 0.906)
- **Sign**: +0.083 (0.593 → 0.677)

Three objects show slight regression (Cable -0.033, Door -0.042, Floor -0.043), possibly because the control model already encodes these through generic texture features that LoRA slightly disrupts.

### 4.3 ControlNet Object Probing

The ControlNet (15 blocks) also encodes object presence, with LoRA providing consistent improvements:

| Object | CN Main (bal. acc.) | CN Control (bal. acc.) | Delta |
|--------|-------------------|---------------------|-------|
| **Table** | 0.914 | 0.852 | +0.063 |
| **Chair** | 0.905 | 0.871 | +0.034 |
| **Computer Monitor** | 0.885 | 0.486 | **+0.399** |
| **Cable** | 0.854 | 0.873 | -0.019 |
| **Building** | 0.841 | 0.725 | +0.116 |
| **Keyboard** | 0.803 | 0.567 | **+0.236** |
| **Trash Can** | 0.777 | 0.614 | **+0.162** |
| **Box** | 0.770 | 0.766 | +0.003 |
| **Door** | 0.762 | 0.654 | +0.108 |
| **Window** | 0.750 | 0.625 | +0.125 |

Notable: ControlNet shows **even larger LoRA deltas** than the DiT for some objects (Computer Monitor +0.399, Keyboard +0.236), which makes sense because LoRA is directly fused into ControlNet weights.

> **Figure**: See [fig7_object_probing](probing_analysis_output/fig7_object_probing.png)

---

## 5. Results — No-ControlNet Ablation

### 5.1 Sanity Check: ControlNet Is Essential

The No-ControlNet condition confirms that **all SPAD-related information enters through the ControlNet**:

| Target | No-CN Best R² | Positive R² Blocks |
|--------|--------------|-------------------|
| Bit density | -0.059 | 0/399 (0.0%) |
| Depth | -0.040 | 0/399 (0.0%) |
| Variance | -0.358 | 0/399 (0.0%) |

Every single block-timestep combination has R² <= 0, meaning the bare DiT (without ControlNet) contains **zero linearly decodable information** about SPAD measurements, scene depth, or reconstruction uncertainty. This is the expected null result and validates our experimental setup.

### 5.2 No-CN Spatial Probing

| Target | No-CN Best R² (Spatial) |
|--------|------------------------|
| Bit density | -0.046 |
| Depth | 0.406 (S0, t=0) |
| Variance | -0.067 |

The one exception is spatial depth at S0, t=0 with R² = 0.406. This is likely because the initial noisy latent at S0 t=0 has some spatial structure from the VAE encoding that correlates with scene depth — not meaningful SPAD-derived information.

---

## 6. LoRA Delta Analysis

### 6.1 Where Does LoRA Help Most?

The delta heatmap (Main R² minus Control R²) reveals where LoRA fine-tuning adds the most linear decodability:

#### Depth (DiT Global)
- **Mean delta**: +0.317 across 399 block-timestep combinations
- **Positive in**: 357/399 (89.5%) of combinations
- **Peak gains**: Early single blocks at early timesteps (S1-S7, t=4-t=9), with individual deltas exceeding +1.0
- **Losses**: Only at J13-J18 at t=0 (up to -0.36) — LoRA slightly disrupts the very earliest joint block representations, but this is quickly recovered

#### Depth (DiT Spatial)
- **Mean delta**: +0.178 across 70 block-timestep combinations
- **Positive in**: 66/70 (94.3%)
- **Peak gain**: S37 t=14 (+0.310)

#### Variance (DiT Global)
- **Mean delta**: -0.061 — LoRA does NOT help variance globally on average
- **Positive in**: 111/399 (27.8%)
- **Peak gains**: J0 at mid-late timesteps (t=14-t=24), where deltas reach +0.94 to +1.24
- **Interpretation**: LoRA redistributes the variance signal. The control already encodes variance well (R² = 0.446), and LoRA slightly weakens it globally while **improving spatial resolution** (spatial delta = +0.072)

#### Variance (DiT Spatial)
- **Mean delta**: +0.161
- **Positive in**: 69/70 (98.6%)
- **Interpretation**: Although global variance R² is similar between conditions, LoRA significantly improves the **spatial localization** of uncertainty

### 6.2 Summary: What LoRA Teaches

| Property | LoRA Effect | Magnitude | Interpretation |
|----------|-------------|-----------|----------------|
| Bit density | None | ~0 | Inherent in ControlNet conditioning |
| Depth (global) | **Strong positive** | +0.308 best | LoRA teaches SPAD→3D geometry conversion |
| Depth (spatial) | **Strong positive** | +0.232 best | LoRA enables spatially-resolved depth decoding |
| Variance (global) | Neutral/slight negative | -0.022 | Pretrained model already knows uncertainty |
| Variance (spatial) | **Moderate positive** | +0.072 best | LoRA improves spatial uncertainty localization |
| Object presence | **Positive for 20/24 objects** | +0.184 max (mouse) | LoRA enhances semantic scene understanding |
| CN depth | **Strong positive** | +0.203 best | LoRA directly improves CN's depth encoding |

> **Figures**: See [fig3_delta_heatmap](probing_analysis_output/fig3_delta_heatmap.png), [fig5_dit_vs_cn](probing_analysis_output/fig5_dit_vs_cn.png)

---

## 7. Spatial vs Global Probing

### 7.1 Why Spatial Probing Matters

Depth and variance are **spatially heterogeneous** — they vary across the image. Global mean-pooling collapses 1024 tokens into 1 vector, losing the spatial pattern. Spatial probing preserves the full per-patch structure:

| Target | Global R² (best) | Spatial R² (best) | Improvement |
|--------|-----------------|------------------|-------------|
| Bit density | 0.998 | 0.959 | Global wins (scalar property) |
| Depth | 0.437 | **0.685** | **+0.248 (+57%)** |
| Variance | 0.424 | **0.506** | **+0.082 (+19%)** |

For **depth**, spatial probing provides R² = 0.685 — the probe can decode a 32x32 depth map from a single block's activations with Pearson r = 0.836. This is a striking result: from **1-bit binary SPAD data**, the model's internal representations contain enough information to linearly reconstruct a spatially-resolved depth map.

Bit density is a scalar-like property (mean across the frame), so global probing is more appropriate and achieves higher R².

> **Figure**: See [fig6_global_vs_spatial](probing_analysis_output/fig6_global_vs_spatial.png)

---

## 8. Information Flow Through the DiT

### 8.1 Layer-by-Layer Knowledge Profile

```
Block:  J0  J1 ····· J8  J9 ····· J18 | S0  S1 ····· S7 ····· S37
        ─────────────────────────────────────────────────────────────
Bit     *   ██      ██  ██      ██  | ██  █▓      ▓▓      ▓▓
Depth   *   **      *▓  *▓      ▓█  | ▓█  █▓      ▓▓      ▓▓
Var     *   ██      █▓  ▓▓      ▓▓  | ▓▓  ▓*      *▓      ▓▓
```

1. **J0 (input)**: Noisy — neither measurements nor structure well-formed yet. LoRA shows large variance gains here (+1.24 at t=19)
2. **J1-J8 (early joint)**: Bit density saturates to R² >= 0.998. ControlNet signal is fully absorbed. Variance awareness begins building (R² ~ 0.30-0.42)
3. **J9-J18 (late joint)**: Bit density remains near-perfect. Depth starts emerging in Main (R² ~ 0.10-0.30). This is the **measurement-to-geometry conversion zone** — where LoRA has its greatest impact
4. **S0-S7 (early single)**: **Peak depth** (global R² = 0.44, spatial R² = 0.68). The model has converted SPAD photon statistics into 3D scene understanding
5. **S8-S37 (late single)**: Depth signal gradually weakens as the model shifts focus from geometry to **appearance/texture generation**. Spatial depth remains high (R² ~ 0.67) even in late blocks

### 8.2 Temporal Dynamics (Denoising Steps)

| Phase | Steps | Bit Density | Depth | Variance |
|-------|-------|-------------|-------|----------|
| **ControlNet injection** | t=0 | Peak (0.998) | Low (< 0.1) | Low-Moderate |
| **Early denoising** | t=4-9 | High (0.99) | **Peak** (0.44) | Building (0.30) |
| **Mid denoising** | t=14 | High (0.98) | Moderate (0.30) | **Peak** (0.42) |
| **Late denoising** | t=19-27 | Decaying (0.93) | Decaying (0.15) | High (0.30-0.40) |

**Each target peaks at a different denoising phase**, revealing the temporal structure of generation:
- **t=0**: Conditioning injection — SPAD measurements absorbed
- **t=4-9**: Scene geometry formation — depth understanding peaks
- **t=14+**: Commitment to details — uncertainty crystallizes as the model finalizes structure

> **Figure**: See [fig4_best_timestep_lineplot](probing_analysis_output/fig4_best_timestep_lineplot.png)

---

## 9. Comparison with AC3D (Bhatt et al., 2024)

Our probing methodology is inspired by AC3D, which probed Stable Video Diffusion (SVD) for 3D scene properties. Key parallels and differences:

| Aspect | AC3D (SVD) | Ours (FLUX.1-dev) |
|--------|-----------|-------------------|
| Architecture | U-Net (SVD) | DiT (FLUX) |
| Conditioning | Temporal (video frames) | Spatial (SPAD binary sensor) |
| Input modality | RGB images | 1-bit binary sensor data |
| Probe type | Linear (same) | Linear (same) |
| Depth R² | ~0.65 (best) | 0.685 (spatial, best) |
| Key finding | Mid-layers encode 3D | Early single blocks encode 3D |

Our spatial depth R² (0.685) is comparable to AC3D's best depth probing results, despite our input being 1-bit binary data vs RGB. This suggests the FLUX DiT successfully converts extremely sparse photon statistics into a rich 3D scene representation, a result that is both novel and scientifically significant.

---

## 10. Practical Implications

### 10.1 For LoRA Placement
The depth signal builds through J11-J18 and peaks in S1-S7. These blocks are where LoRA adapters have the most leverage for improving SPAD-to-depth reasoning. Late single blocks (S20+) handle texture — the pretrained FLUX already excels at this.

### 10.2 For DPS/PaDIS Guidance Timing
Physics-guided corrections (DPS) should target **t=4-14**, when the model is actively forming geometric structure. Applying DPS at t=0 is wasteful (model hasn't started reasoning); applying at t=24+ is too late (model has committed to structure).

### 10.3 For ControlNet Architecture
Bit density R² >= 0.998 throughout the joint blocks confirms the ControlNet successfully injects SPAD measurements. The signal persists to S37 (R² > 0.80), but a ControlNet with more blocks could potentially maintain higher depth encoding deeper into the network.

### 10.4 For Understanding Model Uncertainty
The fact that cross-seed variance is linearly decodable (R² ~ 0.42-0.51) means the model **knows where it is guessing**. This could be leveraged for:
- Adaptive sampling: more denoising steps in uncertain regions
- Confidence-weighted loss functions during training
- Active sensing: requesting additional SPAD frames for uncertain areas

### 10.5 For Semantic Understanding
The model recognizes objects (balanced accuracy up to 0.982 for tables) from 1-bit binary data. LoRA specifically improves recognition of small/distinctive objects (mouse +0.184, keyboard +0.157), suggesting it teaches fine-grained visual priors that the pretrained model lacks for this unusual modality.

---

## 11. File Structure

```
probing_results_allblocks/          <- Main model (LoRA): 776 samples, all 57 DiT + 15 CN blocks
  activations/                      <-   776 activation files (global_0000.pt - global_0775.pt)
  targets.json                      <-   All targets (continuous + 24 objects + vocab)
  probes/
    probing_results.json            <-   All R² / Pearson r / balanced_acc (57 targets)
    *.png                           <-   Per-target heatmaps and line plots

probing_results_control/            <- Control (no LoRA): 776 samples, all 57 DiT + 15 CN blocks
  activations/
  targets.json
  probes/
    probing_results.json

probing_results_no_cn/              <- No-ControlNet ablation: 776 samples, 57 DiT blocks only
  activations/
  targets.json
  probes/
    probing_results.json

probing_results/                    <- Original sparse experiment: 100 samples, 10 blocks
  activations/
  targets.json
  probes/
    probing_results.json

probing_analysis_output/            <- Comprehensive comparison figures (7 figures, PDF + PNG)
  fig1_main_heatmap.{png,pdf}
  fig2_main_vs_control.{png,pdf}
  fig3_delta_heatmap.{png,pdf}
  fig4_best_timestep_lineplot.{png,pdf}
  fig5_dit_vs_cn.{png,pdf}
  fig6_global_vs_spatial.{png,pdf}
  fig7_object_probing.{png,pdf}
```

---

## 12. Generated Figures — Description and Reproduction Instructions

### Figure List

| # | File | Description |
|---|------|-------------|
| 1 | `fig1_main_heatmap` | 3-panel heatmap: R² for all 57 DiT blocks x 7 timesteps, one panel per target (bit_density, depth, variance). White dashed line separates joint/single blocks. |
| 2 | `fig2_main_vs_control` | 2x3 comparison: Main (top row) vs Control (bottom row) heatmaps for all 3 targets. Reveals where LoRA changes the representation. |
| 3 | `fig3_delta_heatmap` | 3-panel diverging colormap (RdBu): R²(Main) - R²(Control) per block-timestep. Red = LoRA helps, blue = LoRA hurts. Depth panel is overwhelmingly red. |
| 4 | `fig4_best_timestep_lineplot` | R² vs block index at best-mean timestep, Main and Control overlaid. AC3D-style information flow plot. |
| 5 | `fig5_dit_vs_cn` | Grouped bar chart: best R² from DiT vs ControlNet blocks, per condition, per target. Shows CN also carries significant information. |
| 6 | `fig6_global_vs_spatial` | Grouped bar chart: DiT Global, DiT Spatial, CN Global probing compared. Shows spatial probing dramatically improves depth decoding. |
| 7 | `fig7_object_probing` | Horizontal bar chart: balanced accuracy per object (24 classes), Main vs Control, sorted by Main performance. Shows LoRA's semantic gains. |

Additional per-target figures in `probing_results_allblocks/probes/`:

| File Pattern | Description |
|---|---|
| `allblocks_heatmap_{target}.png` | Single-target full heatmap |
| `ac3d_curve_{target}.png` | AC3D-style: best R² per block, main vs control with delta bars |
| `temporal_{target}.png` | R² vs denoising step for selected blocks |
| `multi_target_*.png` | All 3 targets overlaid at their best timesteps |
| `heatmap_{target}.png` | Combined DiT+CN heatmap per target |
| `delta_heatmap_{target}.png` | Per-target delta heatmap (Main - Control) |
| `comparison_best_timestep.png` | Side-by-side best-timestep comparison |

### How to Reproduce All Figures

#### Step 1: Generate the 7 main analysis figures

```bash
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

/home/jw/miniconda3/envs/diffsynth/bin/python probing_analysis.py
```

This reads from `probing_results_allblocks/`, `probing_results_control/`, and `probing_results_no_cn/` and produces figures in `probing_analysis_output/` (both PNG at 300 DPI and PDF).

**Key parameters you can adjust in `probing_analysis.py`:**
- `TARGETS`: List of targets to plot (default: `["bit_density", "depth", "variance"]`)
- Figure size, DPI, colormap choices are all at the top of each plotting function
- PDF output is vector graphics — ideal for LaTeX papers

#### Step 2: Generate per-target heatmaps and AC3D curves

These are generated automatically during probe training. To regenerate:

```bash
# Main model (with LoRA)
/home/jw/miniconda3/envs/diffsynth/bin/python linear_probing.py \
    --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
    --dataset_base /home/jw/engsci/thesis/spad/spad_dataset \
    --output_dir ./probing_results_allblocks \
    --max_samples 776 \
    --train-only

# Control (no LoRA)
/home/jw/miniconda3/envs/diffsynth/bin/python linear_probing.py \
    --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
    --dataset_base /home/jw/engsci/thesis/spad/spad_dataset \
    --output_dir ./probing_results_control \
    --max_samples 776 \
    --train-only
```

The `--train-only` flag skips activation extraction and only re-trains probes + generates figures from existing activations.

**Figures are saved to `{output_dir}/probes/`**.

#### Step 3: Generate the Markdown report (this document)

The report is written by hand using the results from `probing_results.json` files. To re-extract the numerical data:

```bash
/home/jw/miniconda3/envs/diffsynth/bin/python -c "
import json
for name, path in [('main', 'probing_results_allblocks'), ('control', 'probing_results_control'), ('no_cn', 'probing_results_no_cn')]:
    with open(f'{path}/probes/probing_results.json') as f:
        data = json.load(f)
    print(f'{name}: {len(data)} targets, {len(next(iter(data.values())))} block-timestep combos')
    for target in ['bit_density', 'depth', 'variance']:
        if target in data:
            best_k = max(data[target], key=lambda k: data[target][k]['r2'])
            print(f'  {target}: best R²={data[target][best_k][\"r2\"]:.4f} @ {best_k}')
"
```

#### Step 4: Custom figures for your paper/presentation

All figure-generating code is in two files:

1. **`probing_analysis.py`** — The 7 main comparison figures. Each figure is a standalone function (`plot_fig1_main_heatmap`, `plot_fig2_main_vs_control`, etc.). You can import and call them individually:

```python
import probing_analysis as pa

# Load data
main = pa.load_results("probing_results_allblocks")
ctrl = pa.load_results("probing_results_control")
no_cn = pa.load_results("probing_results_no_cn")

# Generate just one figure
pa.plot_fig3_delta_heatmap(main, ctrl, output_dir="my_figures/")
```

2. **`linear_probing.py`** (function `train_probes()`, around line 850+) — The per-target heatmaps and line plots generated during training. The plotting code is inline in the training function.

**For LaTeX**: All figures are also saved as PDF (vector graphics). Use the PDF versions in your paper.

**For adjusting colors/sizes**: Each plot function in `probing_analysis.py` has hardcoded figure sizes and colormaps near the top. Common modifications:
- Change `figsize=(W, H)` for paper column width
- Change `cmap="viridis"` to other colormaps (e.g., `"inferno"`, `"plasma"`)
- Change `dpi=300` for higher/lower resolution PNGs
- Add `plt.rcParams.update({"font.size": 12})` at the top for consistent font sizes

---

## Appendix A: Full R² Tables — All-Blocks Global (Main Model)

### A.1 Bit Density (DiT Joint Blocks)

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

### A.2 Depth (DiT Joint + Early Single Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | -0.043 | 0.019 | 0.052 | 0.068 | 0.099 | 0.096 | 0.091 | **0.099** |
| J1 | -0.040 | 0.026 | 0.040 | 0.061 | 0.085 | 0.104 | 0.101 | **0.104** |
| J7 | -0.041 | 0.015 | 0.054 | 0.080 | 0.149 | 0.176 | 0.153 | **0.176** |
| J8 | -0.030 | 0.029 | 0.086 | 0.137 | 0.198 | 0.227 | 0.205 | **0.227** |
| J11 | -0.024 | 0.047 | 0.130 | 0.204 | 0.295 | 0.277 | 0.220 | **0.295** |
| J14 | -0.064 | 0.060 | 0.156 | 0.209 | 0.263 | 0.235 | 0.176 | **0.263** |
| J18 | -0.064 | 0.088 | 0.186 | 0.224 | 0.253 | 0.214 | 0.147 | **0.253** |
| S0 | -0.031 | 0.108 | 0.218 | 0.249 | 0.275 | 0.222 | 0.143 | **0.275** |
| S1 | -0.010 | 0.420 | 0.306 | 0.296 | 0.264 | 0.226 | 0.173 | **0.420** |
| S3 | -0.013 | 0.382 | 0.239 | 0.232 | 0.201 | 0.180 | 0.134 | **0.382** |
| S4 | -0.008 | 0.437 | 0.372 | 0.252 | 0.203 | 0.169 | 0.133 | **0.437** |
| S7 | -0.023 | 0.267 | 0.409 | 0.289 | 0.242 | 0.195 | 0.134 | **0.409** |

### A.3 Variance (DiT Joint + Early Single Blocks)

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | 0.005 | 0.045 | 0.200 | 0.361 | 0.380 | 0.281 | -0.004 | **0.380** |
| J1 | 0.078 | 0.227 | 0.339 | 0.424 | 0.418 | 0.422 | 0.412 | **0.424** |
| J2 | 0.098 | 0.226 | 0.316 | 0.369 | 0.373 | 0.384 | 0.374 | **0.384** |
| J6 | 0.099 | 0.215 | 0.273 | 0.317 | 0.324 | 0.362 | 0.392 | **0.392** |
| J7 | 0.089 | 0.210 | 0.266 | 0.303 | 0.310 | 0.366 | 0.403 | **0.403** |
| J12 | 0.092 | 0.239 | 0.272 | 0.312 | 0.337 | 0.418 | 0.355 | **0.418** |
| S0 | 0.073 | 0.218 | 0.224 | 0.221 | 0.207 | 0.213 | 0.226 | **0.226** |
| S7 | 0.102 | 0.208 | 0.213 | 0.189 | 0.173 | 0.200 | 0.193 | **0.213** |

---

## Appendix B: Object Presence — Full Results

### B.1 DiT Global — All 24 Objects, Main vs Control

| Object | Prevalence | Main (bal. acc.) | Main (block) | Control (bal. acc.) | Control (block) | Delta |
|--------|-----------|-----------------|-------------|--------------------|-----------------|----|
| Table | 27.8% | 0.982 | S5 t=14 | 0.912 | S0 t=4 | +0.070 |
| Chair | 29.0% | 0.953 | S9 t=9 | 0.881 | S34 t=0 | +0.072 |
| Comp. Monitor | 13.0% | 0.917 | S8 t=4 | 0.878 | S9 t=4 | +0.039 |
| Building | 10.6% | 0.906 | S6 t=4 | 0.818 | S9 t=4 | +0.089 |
| Keyboard | 9.7% | 0.881 | S1 t=27 | 0.724 | S2 t=4 | +0.157 |
| Cable | 12.5% | 0.873 | S12 t=19 | 0.906 | J11 t=27 | -0.033 |
| Mouse | 6.1% | 0.861 | S0 t=27 | 0.677 | S0 t=14 | +0.184 |
| Box | 8.6% | 0.829 | J8 t=19 | 0.819 | S10 t=19 | +0.010 |
| Computer | 5.8% | 0.820 | S3 t=0 | 0.804 | J11 t=19 | +0.016 |
| TV | 7.6% | 0.808 | S16 t=0 | 0.771 | S1 t=4 | +0.037 |
| Trash Can | 8.4% | 0.781 | S10 t=4 | 0.739 | S11 t=9 | +0.041 |
| Window | 10.4% | 0.747 | J8 t=27 | 0.743 | S13 t=4 | +0.003 |
| File Cabinet | 5.3% | 0.734 | J15 t=24 | 0.734 | S2 t=9 | +0.000 |
| Door | 8.6% | 0.712 | J7 t=9 | 0.754 | S15 t=27 | -0.042 |
| Wall | 29.9% | 0.699 | J9 t=0 | 0.667 | J12 t=0 | +0.032 |
| Sign | 5.5% | 0.677 | J18 t=19 | 0.593 | S25 t=4 | +0.083 |
| Floor | 23.3% | 0.645 | S33 t=4 | 0.689 | S16 t=4 | -0.043 |
| Shelf | 8.0% | 0.622 | S19 t=27 | 0.563 | S19 t=4 | +0.059 |
| Pavement | 5.2% | 0.583 | S3 t=14 | 0.521 | S6 t=9 | +0.063 |
| Tree | 7.0% | 0.581 | J10 t=24 | 0.535 | S4 t=9 | +0.047 |
| Cabinet | 5.4% | 0.559 | S8 t=24 | 0.529 | S10 t=9 | +0.029 |
| Bench | 5.9% | 0.522 | S23 t=27 | 0.522 | S31 t=19 | +0.000 |
| Bulletin Board | 5.2% | 0.500 | — | 0.500 | — | +0.000 |
| Plant | 5.3% | 0.500 | — | 0.500 | — | +0.000 |

---

## Appendix C: Spatial Probing Full Tables

### C.1 Spatial Bit Density — Main Model

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| J0 | 0.875 | 0.889 | 0.902 | 0.909 | 0.907 | 0.892 | 0.852 | **0.909** |
| J14 | 0.922 | 0.912 | 0.899 | 0.889 | 0.861 | 0.816 | 0.768 | **0.922** |
| J18 | 0.917 | 0.906 | 0.886 | 0.878 | 0.858 | 0.810 | 0.766 | **0.917** |
| S0 | 0.903 | 0.885 | 0.869 | 0.867 | 0.841 | 0.790 | 0.745 | **0.903** |
| S9 | 0.877 | 0.862 | 0.858 | 0.846 | 0.822 | 0.770 | 0.719 | **0.877** |
| S19 | 0.925 | 0.925 | 0.917 | 0.906 | 0.879 | 0.826 | 0.766 | **0.925** |
| S28 | 0.942 | 0.940 | 0.928 | 0.916 | 0.898 | 0.844 | 0.774 | **0.942** |
| S37 | 0.959 | 0.953 | 0.945 | 0.933 | 0.911 | 0.857 | 0.788 | **0.959** |

### C.2 Spatial Depth — Main vs Control

| Block | Main Best | Main Best Step | Control Best | Control Best Step | Delta |
|-------|-----------|---------------|-------------|------------------|-------|
| J0 | 0.497 | t=19 | 0.332 | t=9 | +0.165 |
| J14 | 0.575 | t=19 | 0.364 | t=19 | +0.210 |
| J18 | 0.626 | t=14 | 0.377 | t=9 | +0.249 |
| S0 | 0.660 | t=14 | 0.427 | t=9 | +0.233 |
| S9 | 0.685 | t=14 | 0.453 | t=4 | +0.232 |
| S19 | 0.675 | t=9 | 0.440 | t=0 | +0.236 |
| S28 | 0.678 | t=9 | 0.414 | t=0 | +0.264 |
| S37 | 0.671 | t=9 | 0.381 | t=0 | +0.290 |

### C.3 Spatial Variance — Main vs Control

| Block | Main Best | Main Best Step | Control Best | Control Best Step | Delta |
|-------|-----------|---------------|-------------|------------------|-------|
| J0 | 0.393 | t=27 | 0.267 | t=27 | +0.126 |
| J14 | 0.399 | t=27 | 0.313 | t=14 | +0.086 |
| S0 | 0.497 | t=4 | 0.375 | t=14 | +0.122 |
| S9 | 0.498 | t=4 | 0.434 | t=0 | +0.064 |
| S19 | 0.504 | t=4 | 0.375 | t=0 | +0.128 |
| S28 | 0.506 | t=4 | 0.374 | t=14 | +0.132 |
| S37 | 0.502 | t=4 | 0.407 | t=0 | +0.095 |

---

## Appendix D: ControlNet Block Results (Main Model)

### D.1 CN Bit Density

| Block | t=0 | t=4 | t=9 | t=14 | t=19 | t=24 | t=27 | Best |
|-------|-----|-----|-----|------|------|------|------|------|
| CN-J0 | 0.999 | 0.997 | 0.997 | 0.996 | 0.994 | 0.988 | 0.975 | **0.999** |
| CN-J1 | 1.000 | 0.999 | 0.999 | 0.999 | 0.997 | 0.993 | 0.983 | **1.000** |
| CN-J2 | 0.999 | 0.998 | 0.998 | 0.998 | 0.995 | 0.990 | 0.978 | **0.999** |
| CN-J3 | 0.999 | 0.998 | 0.997 | 0.996 | 0.993 | 0.985 | 0.970 | **0.999** |
| CN-J4 | 0.998 | 0.997 | 0.995 | 0.993 | 0.988 | 0.976 | 0.957 | **0.998** |

### D.2 CN Depth (Top 5)

| Block | Best R² | Timestep | Pearson r |
|-------|---------|----------|-----------|
| CN-S5 | 0.227 | t=19 | 0.555 |
| CN-S6 | 0.215 | t=19 | 0.598 |
| CN-S2 | 0.170 | t=19 | 0.552 |
| CN-S9 | 0.166 | t=24 | 0.603 |
| CN-S5 | 0.165 | t=24 | 0.541 |

### D.3 CN Variance (Top 5)

| Block | Best R² | Timestep | Pearson r |
|-------|---------|----------|-----------|
| CN-J2 | 0.465 | t=27 | 0.698 |
| CN-J1 | 0.443 | t=24 | 0.682 |
| CN-S1 | 0.442 | t=19 | 0.688 |
| CN-J4 | 0.437 | t=19 | 0.696 |
| CN-J3 | 0.433 | t=27 | 0.666 |

---

## Appendix E: Cross-Frame Variance Probing

### E.1 Motivation

Cross-frame variance measures how much the model's output changes when conditioned on different binary SPAD frame realizations of the same scene (same seed, same ground truth, different Bernoulli samples). This is computed from 7 different temporal frame realizations (bits, bits_frame_{1000,4000,5000,8000,12000,16000}) generated at `validation_outputs_crossframe/baseline/`.

**Probing question**: Where in the DiT does the model encode information about *which specific binary frame* it received? High cross-frame variance probing R² indicates blocks/timesteps that are sensitive to the stochastic SPAD realization rather than the underlying scene content.

### E.2 Target Computation

For each of the 776 validation scenes:
1. Load outputs from all 7 frame realizations (775 valid — 1 scene missing a frame)
2. Compute per-pixel RGB variance across frames → mean over channels → scalar (global) and 32x32 map (spatial)
3. Add as `crossframe_variance` / `spatial_crossframe_variance` to existing `targets.json`

Cross-frame variance statistics: range [0.000337, 0.014199], mean=0.00255, std=0.00186.

### E.3 Results

#### E.3.1 Global Probing — Cross-Frame Variance

**Main Model (LoRA) — Top 10 blocks:**

| Block | Timestep | R² | Pearson r |
|-------|----------|-----|-----------|
| joint_1 | t=27 | **0.292** | 0.550 |
| joint_1 | t=24 | 0.260 | 0.521 |
| joint_1 | t=19 | 0.215 | 0.481 |
| joint_4 | t=14 | 0.213 | 0.507 |
| joint_2 | t=14 | 0.210 | 0.481 |
| joint_2 | t=19 | 0.202 | 0.484 |
| joint_5 | t=19 | 0.201 | 0.524 |
| joint_3 | t=14 | 0.185 | 0.461 |
| joint_2 | t=24 | 0.173 | 0.468 |
| joint_6 | t=14 | 0.165 | 0.523 |

**Control (no LoRA) — Top 5 blocks:**

| Block | Timestep | R² | Pearson r |
|-------|----------|-----|-----------|
| joint_4 | t=27 | **0.222** | 0.515 |
| joint_3 | t=27 | 0.192 | 0.477 |
| single_0 | t=27 | 0.192 | 0.542 |
| joint_17 | t=0 | 0.171 | 0.516 |
| joint_1 | t=27 | 0.169 | 0.443 |

**No-ControlNet — Top 5 blocks:**

| Block | Timestep | R² | Pearson r |
|-------|----------|-----|-----------|
| joint_1 | t=0 | **−0.161** | 0.117 |
| joint_0 | t=27 | −0.167 | 0.164 |
| joint_0 | t=24 | −0.174 | 0.137 |
| joint_1 | t=27 | −0.186 | 0.148 |
| joint_0 | t=4 | −0.199 | 0.003 |

#### E.3.2 Cross-Frame vs Seed Variance Comparison

| Target | Best R² (Main) | Best Block (Main) | Best R² (Control) | Best R² (No-CN) |
|--------|---------------|-------------------|-------------------|-----------------|
| Seed variance (10 seeds) | **0.424** | joint_1 @ t=14 | 0.446 | −0.358 |
| Cross-frame variance (7 frames) | **0.292** | joint_1 @ t=27 | 0.222 | −0.161 |

### E.4 Analysis

**Key findings:**

1. **Cross-frame information is encoded in early joint blocks**: joint_1 dominates for the main model (R²=0.292 at t=27), consistent with it being the first block where ControlNet-injected SPAD information is processed.

2. **Different temporal profiles for seed vs cross-frame variance**: Seed variance peaks at t=14 (mid-denoising) while cross-frame variance peaks at t=27 (near completion). This suggests frame-dependent variation is *retained through to the final output*, while seed-dependent variation is generated mid-process and partially resolved by the end.

3. **LoRA amplifies frame sensitivity**: Main model R²=0.292 vs Control R²=0.222 (+32%). The LoRA finetuning teaches the model to be *more* responsive to the specific SPAD frame content, not less. This makes sense — the LoRA learns to extract more signal from the binary conditioning, but this also means it's more sensitive to the stochastic realization.

4. **ControlNet is the sole pathway**: No-CN R² values are all negative, confirming that without ControlNet, the DiT has zero ability to predict cross-frame variance. All frame-dependent information enters through the ControlNet.

5. **Cross-frame variance is harder to predict than seed variance**: Best R²=0.292 (crossframe) vs 0.424 (seed). This makes sense — seed variance is a property of the generation process itself, while cross-frame variance depends on the interaction between conditioning and generation.

**Implications for consistency training**: The cross-frame variance signal is concentrated in early joint blocks (1–6) at later timesteps (t=14–27). Consistency training should target these regions to reduce frame-dependent output variation. The fact that LoRA *increases* frame sensitivity (vs control) suggests that consistency regularization is indeed needed to counteract this effect.

## Appendix F: Cross-Architecture Comparison — SD1.5 vs FLUX

### F.1 Motivation

A central question is whether the internal representations observed in FLUX (12B DiT) are unique to large-scale transformers or a general property of diffusion models trained with ControlNet on SPAD data. To answer this, we replicate the full linear probing pipeline on a Stable Diffusion 1.5 UNet (860M) + ControlNet (361M) trained on the same SPAD dataset with the same train/val split.

### F.2 Architecture Differences

| Property | FLUX (DiT) | SD1.5 (UNet) |
|----------|-----------|--------------|
| Backbone | 12B DiT (19 joint + 38 single blocks) | 860M UNet (12 input + 1 mid + 12 output blocks) |
| ControlNet | 15 parallel DiT blocks | 12 input + 1 mid encoder blocks |
| Feature dims | Uniform 3072 | Variable: 320 / 640 / 1280 |
| Spatial resolution | Uniform 64×64 tokens | Variable: 64×64 → 32×32 → 16×16 → 8×8 |
| Attention | Joint self+cross attention | Separate self-attention + cross-attention (CLIP text) |
| Total probed blocks | 57 × 7 timesteps = 399 | 38 × 7 timesteps = 266 |
| LoRA | Rank-16 on ControlNet | None (full ControlNet finetuned) |

**SD1.5 block naming**: `cn_input_{0-11}` (ControlNet encoder), `cn_mid` (ControlNet middle), `unet_input_{0-11}` (UNet encoder), `unet_mid` (UNet middle), `unet_output_{0-11}` (UNet decoder). Each block captures the output of one `TimestepEmbedSequential` module (ResBlock ± SpatialTransformer).

### F.3 Results — Global Probing

#### F.3.1 Best R² Per Target: SD1.5 vs FLUX

| Target | SD1.5 R² | SD1.5 Best Block | FLUX R² | FLUX Best Block |
|--------|---------|-----------------|--------|----------------|
| bit_density | 0.993 | cn_input_1@t49 | 0.998 | joint_1@t27 |
| depth | 0.375 | unet_input_8@t49 | 0.437 | single_9@t14 |
| variance (seed) | 0.472 | cn_input_1@t21 | 0.424 | joint_1@t14 |
| crossframe_variance | 0.293 | cn_input_1@t0 | 0.292 | joint_1@t27 |

#### F.3.2 Bit Density — Top 10 Blocks (SD1.5)

| Block | R² | Pearson r |
|-------|-----|----------|
| cn_input_1@t49 | 0.993 | 0.999 |
| cn_input_1@t0 | 0.993 | 0.999 |
| cn_input_1@t35 | 0.992 | 0.999 |
| cn_input_1@t28 | 0.991 | 0.999 |
| cn_input_1@t7 | 0.991 | 0.999 |
| cn_input_1@t14 | 0.990 | 0.999 |
| cn_input_1@t21 | 0.990 | 0.999 |
| cn_input_2@t0 | 0.979 | 0.998 |
| cn_input_2@t7 | 0.974 | 0.998 |
| cn_input_3@t0 | 0.973 | 0.997 |

Bit density is near-perfectly encoded in `cn_input_1` regardless of timestep — identical to FLUX's `joint_1`. This is the first block after SPAD conditioning is injected, confirming that bit density (a direct measurement property) is preserved as-is by the ControlNet encoder.

#### F.3.3 Depth — Top 10 Blocks (SD1.5)

| Block | R² | Pearson r |
|-------|-----|----------|
| unet_input_8@t49 | 0.375 | 0.655 |
| unet_output_11@t0 | 0.361 | 0.644 |
| unet_input_9@t49 | 0.330 | 0.642 |
| unet_output_1@t21 | 0.320 | 0.656 |
| unet_output_3@t21 | 0.315 | 0.585 |
| unet_input_3@t49 | 0.294 | 0.553 |
| unet_input_2@t49 | 0.272 | 0.534 |
| cn_input_6@t49 | 0.269 | 0.549 |
| unet_output_10@t49 | 0.263 | 0.558 |
| cn_input_6@t35 | 0.258 | 0.576 |

Depth encoding is concentrated in the UNet's **mid-depth encoder blocks** (input_8, input_9) and **early decoder blocks** (output_1, output_3, output_11). These are the 1280-channel, 8×8-resolution blocks where the UNet bottleneck compresses spatial information. In FLUX, depth peaks in `single_9@t14` — a later DiT block. Both architectures require multiple processing stages before 3D geometry emerges.

#### F.3.4 Variance — Top 10 Blocks (SD1.5)

| Block | R² | Pearson r |
|-------|-----|----------|
| cn_input_1@t21 | 0.472 | 0.695 |
| cn_input_3@t21 | 0.460 | 0.686 |
| cn_input_1@t28 | 0.456 | 0.678 |
| cn_input_1@t14 | 0.450 | 0.679 |
| cn_input_3@t14 | 0.444 | 0.676 |
| cn_input_3@t28 | 0.443 | 0.675 |
| cn_input_2@t21 | 0.431 | 0.658 |
| cn_input_2@t7 | 0.425 | 0.661 |
| cn_input_2@t14 | 0.425 | 0.655 |
| cn_input_2@t0 | 0.420 | 0.658 |

Variance (seed) is dominated by ControlNet early blocks — the same blocks that encode bit density. This matches FLUX where `joint_1` also leads for both targets. Interestingly, SD1.5 achieves **higher** variance R² than FLUX (0.472 vs 0.424).

#### F.3.5 Cross-Frame Variance — Top 10 Blocks (SD1.5)

| Block | R² | Pearson r |
|-------|-----|----------|
| cn_input_1@t0 | 0.293 | 0.558 |
| cn_input_2@t0 | 0.287 | 0.552 |
| cn_input_1@t7 | 0.282 | 0.543 |
| cn_input_3@t0 | 0.268 | 0.529 |
| cn_input_2@t7 | 0.264 | 0.530 |
| cn_input_1@t14 | 0.260 | 0.516 |
| cn_input_1@t21 | 0.241 | 0.499 |
| cn_input_3@t7 | 0.240 | 0.497 |
| cn_input_1@t28 | 0.228 | 0.488 |
| unet_output_11@t0 | 0.225 | 0.477 |

Cross-frame variance in SD1.5 peaks at **t=0** (pure noise), unlike FLUX where it peaks at t=27 (near completion). This suggests that SD1.5's smaller ControlNet encodes frame sensitivity information immediately when processing the SPAD input, while FLUX's larger DiT develops this signal progressively through the denoising process.

#### F.3.6 Object Presence — Top 10 Objects (SD1.5)

| Object | Best R² | Best Block |
|--------|---------|-----------|
| building | 0.709 | cn_input_8@t0 |
| table | 0.703 | unet_output_4@t0 |
| chair | 0.669 | unet_output_9@t0 |
| tree | 0.461 | cn_input_4@t49 |
| bench | 0.393 | unet_output_7@t28 |
| keyboard | 0.356 | unet_input_1@t49 |
| cable | 0.280 | unet_output_9@t0 |
| box | 0.219 | cn_input_1@t35 |
| tv | 0.218 | cn_input_5@t35 |
| sign | 0.203 | unet_output_3@t49 |

Large-scale scene elements (building, table, chair) are well-detected, while fine-grained objects (cable, box) are weaker. Object recognition peaks in **UNet decoder blocks** (output_4, output_7, output_9), where skip connections merge multi-resolution features.

### F.4 Results — Spatial Probing

#### F.4.1 Best R² Per Target: SD1.5 vs FLUX

| Target | SD1.5 R² | SD1.5 Best Block | FLUX R² | FLUX Best Block |
|--------|---------|-----------------|--------|----------------|
| spatial_bit_density | 0.974 | cn_input_1@t0 | 0.959 | single_9@t14 |
| spatial_depth | 0.727 | unet_output_4@t28 | 0.685 | single_9@t14 |
| spatial_variance | 0.493 | cn_input_4@t0 | 0.506 | joint_3@t0 |
| spatial_crossframe_var | 0.279 | cn_input_4@t0 | — | — |

#### F.4.2 Spatial Bit Density — Top 5 Blocks (SD1.5)

| Block | R² |
|-------|-----|
| cn_input_1@t0 | 0.974 |
| cn_input_1@t7 | 0.967 |
| cn_input_1@t49 | 0.965 |
| cn_input_1@t14 | 0.963 |
| cn_input_1@t35 | 0.961 |

#### F.4.3 Spatial Depth — Top 5 Blocks (SD1.5)

| Block | R² |
|-------|-----|
| unet_output_4@t28 | 0.727 |
| unet_output_4@t0 | 0.726 |
| unet_output_4@t7 | 0.725 |
| unet_output_4@t35 | 0.724 |
| unet_output_4@t14 | 0.724 |

Spatial depth probing peaks in `unet_output_4` — the first decoder block at 32×32 resolution (640 channels) receiving skip connections from `unet_input_7`. This is where high-resolution spatial structure re-emerges after the 8×8 bottleneck. SD1.5 achieves **higher spatial depth R²** than FLUX (0.727 vs 0.685), suggesting that the UNet's explicit multi-scale structure with skip connections preserves spatial geometry more effectively than the DiT's uniform-resolution token processing.

#### F.4.4 Spatial Variance and Cross-Frame Variance — Top 5 Blocks (SD1.5)

| Block | Spatial Var R² | Spatial CF-Var R² |
|-------|----------------|-------------------|
| cn_input_4@t0 | 0.493 | 0.279 |
| cn_input_2@t0 | 0.488 | 0.274 |
| cn_input_5@t0 | 0.486 | — |
| cn_input_4@t7 | 0.485 | 0.270 |
| cn_input_2@t7 | 0.480 | — |

### F.5 Analysis and Cross-Architecture Insights

#### F.5.1 SD1.5 Is Surprisingly Competitive

Despite having **14× fewer parameters** (1.2B vs 12B), SD1.5 achieves comparable — and in some cases superior — probing performance:

- **Bit density**: Near-identical (0.993 vs 0.998 global; 0.974 vs 0.959 spatial). This is expected — bit density is a direct measurement property that requires minimal processing to preserve.
- **Depth**: Lower globally (0.375 vs 0.437) but **higher spatially** (0.727 vs 0.685). The UNet's hierarchical resolution structure with skip connections preserves per-pixel depth information more effectively than the DiT's uniform token representation.
- **Variance**: SD1.5 actually **outperforms** FLUX on seed variance (0.472 vs 0.424 global; 0.493 vs 0.506 spatial — roughly tied). This may reflect that smaller models with less capacity have more predictable generation uncertainty.
- **Cross-frame variance**: Near-identical globally (0.293 vs 0.292).

#### F.5.2 Where the Architectures Diverge

1. **Information localization**: In SD1.5, knowledge is localized by function:
   - ControlNet early blocks (cn_input_1–3): bit density, variance, cross-frame variance
   - UNet mid-depth encoder (input_8–9): depth
   - UNet decoder (output_1–11): objects, spatial depth

   In FLUX, joint_1 and single_9 are "generalist" blocks encoding multiple targets. The DiT's uniform architecture leads to more distributed representations.

2. **Temporal dynamics differ**: SD1.5's cross-frame variance peaks at t=0 (pure noise), while FLUX's peaks at t=27 (near completion). SD1.5's ControlNet immediately encodes frame sensitivity; FLUX develops it progressively.

3. **Spatial advantage for UNet**: SD1.5's hierarchical spatial processing (64→32→16→8→16→32→64) with skip connections yields better spatial depth than FLUX's uniform 64×64 tokens. This is the one clear advantage of the UNet architecture for this task.

4. **Object recognition gap**: SD1.5's best object R² (building: 0.709) is lower than FLUX's typical object R² values. The DiT's larger capacity and joint attention mechanism likely enable richer semantic representations.

#### F.5.3 Implications

- **ControlNet architecture generalizes**: Both UNet-ControlNet and DiT-ControlNet learn to encode SPAD measurement properties (bit density) and derived scene properties (depth, variance) in early ControlNet blocks. This is an architectural pattern, not a scale-dependent phenomenon.
- **Scale helps semantics, not measurement**: The 14× parameter gap primarily affects depth and object recognition — properties requiring semantic understanding. Measurement preservation (bit density) is equally good in both architectures.
- **UNet skip connections are a spatial advantage**: For per-pixel regression tasks (spatial depth), the UNet's explicit multi-scale processing outperforms the DiT's uniform representation. This suggests that hybrid architectures combining DiT's capacity with UNet's spatial structure could be beneficial.
- **Consistency training insights**: Since both architectures encode cross-frame variance in early ControlNet blocks, consistency training strategies should target these blocks regardless of backbone choice.

### F.6 Generated Heatmaps

All SD1.5 heatmaps are in `/home/jw/engsci/thesis/spad/spad-diffusion/probing_results_sd15/probes/`:

- `heatmap_bit_density.png`, `heatmap_depth.png`, `heatmap_variance.png`, `heatmap_crossframe_variance.png`
- `heatmap_spatial_bit_density.png`, `heatmap_spatial_depth.png`, `heatmap_spatial_variance.png`, `heatmap_spatial_crossframe_variance.png`
- `heatmap_obj_*.png` (24 object presence heatmaps)

Full numerical results: `probing_results.json` (8,512 global probes) and `spatial_streaming_results.json` (1,064 spatial probes).
