# Diffusion Posterior Sampling & Per-Frame Consistency — Experimental Report
**Date: 2026-03-28 | Status: In Progress (experiments ongoing)**

---

## 1. Motivation and Research Questions

We investigate two complementary inference- and training-time techniques for improving SPAD-to-RGB reconstruction beyond a standard ControlNet + LoRA baseline:

1. **Diffusion Posterior Sampling (DPS)**: Inference-time physics-consistent guidance that corrects the denoising trajectory using the known SPAD forward model.
2. **Per-Frame Consistency Loss**: Training-time regularization that encourages the model to produce identical outputs regardless of which binary SPAD frame realization is used as conditioning.

Specifically, we ask:

1. **Can DPS improve reconstruction quality** by incorporating physics at inference time?
2. **Does operating in pixel space vs latent space** matter for measurement guidance?
3. **When during denoising** is guidance most effective? (informed by probing results)
4. **How should the NLL loss handle class imbalance** in single-frame binary SPAD data?
5. **Does consistency training reduce frame-dependent output variation** without degrading quality?
6. **Do DPS and consistency combine** constructively?

---

## 2. Background

### 2.1 SPAD Forward Model

The SPAD sensor follows a Bernoulli detection model:

```
Intensity I ∈ [0, ∞) (linear, after sRGB→linear conversion)
   ↓
Exposure H = softplus(α·I + β),    clamp H ≥ H_min = 1e-6
   ↓
Detection probability p = 1 − exp(−H)
   ↓
Binary observation b ~ Bernoulli(p)
```

- **Default parameters**: α = 1.0, β = 0.0, N = 1 (single frame)
- **NLL**: −[b·log(p) + (1−b)·log(1−p)]
- **Numerically stable**: log(p) = log(−expm1(−H)), log(1−p) = −H

Implementation: `diffsynth/diffusion/spad_forward.py`

### 2.2 Diffusion Posterior Sampling (DPS)

DPS (Chung et al., ICLR 2023) modifies the denoising trajectory to enforce measurement consistency:

At each step t:
1. Predict clean sample: x̂₀ = x_t − σ·v_θ(x_t, t)
2. Compute measurement loss: L = −log p(y | A(x̂₀))
3. Correct velocity: v' = v + η · normalize(∇_x L)

The correction is added to the velocity prediction before the scheduler step. Because the rectified-flow scheduler computes x_{t+1} = x_t + v·(σ_{t+1} − σ_t) with σ_{t+1} < σ_t, adding +∇L to velocity moves latents in −∇L direction, decreasing the NLL.

**PaDIS-style preconditioning** (Song et al., ICLR 2023): gradients are normalized by their mean absolute value before scaling by η, making the correction unit-scale regardless of the loss magnitude.

### 2.3 Per-Frame Consistency

Different binary SPAD frame realizations of the same scene are random samples from the same Bernoulli process. The reconstructed RGB image should be identical regardless of which realization is used.

**Loss** (inspired by IC-Light, ICLR 2024):
```
L = L_flow_match(F1) + λ · ||v_θ(z_t, t, F1) − v_θ(z_t, t, F2)||²
```

- F1, F2: two different binary frames of the same scene
- Same noisy latent z_t and timestep t for both
- **Stop-gradient on F2** to prevent degenerate collapse (F1/F2 randomly assigned per sample, so both directions are covered over training)
- λ-weighted consistency term uses the same scheduler training weight as the flow-matching loss

---

## 3. Experimental Setup

### 3.1 Common Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | FLUX.1-dev (12B) + ControlNet-Union + LoRA (rank 16) |
| **Checkpoint** | `FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors` |
| **Resolution** | 512 × 512 |
| **Denoising steps** | 28 |
| **Evaluation seed** | 42 |
| **Evaluation set** | 776 validation images |
| **Metrics** | PSNR (grayscale), SSIM, LPIPS, FID, CFID, Measurement NLL |
| **Primary metric** | CFID (Conditional Fréchet Inception Distance; Soloveitchik et al., 2021) |
| **GPU** | NVIDIA RTX 5090 (32 GB) |

### 3.2 Metrics Notes

- All per-image metrics (PSNR, SSIM, LPIPS) computed in **grayscale** by default to avoid penalizing color shifts (SPAD is monochrome, so color is inherently ambiguous)
- **CFID** measures whether generated outputs are faithful to their *specific* SPAD inputs, not just realistic in general — the most relevant metric for conditional reconstruction
- **Measurement NLL**: per-pixel Bernoulli NLL measuring physics consistency (lower = output better explains the SPAD observation). Supplementary metric, not standard in the literature.
- **FID**: distributional quality (standard benchmark metric)
- PSNR reported for completeness but de-emphasized due to color sensitivity

### 3.3 Baseline Reference (10-Seed)

The baseline model was evaluated across 10 random seeds to establish variance:

| Metric | Mean | Std |
|--------|------|-----|
| PSNR (dB) | 17.857 | ±0.087 |
| SSIM | 0.598 | ±0.001 |
| LPIPS | 0.413 | ±0.001 |
| FID | 66.29 | ±0.74 |
| **CFID** | **152.04** | **±1.08** |
| Meas. NLL | 0.7472 | ±0.0004 |

Seed 42 is used as the canonical comparison point (CFID = 151.94).

---

## 4. Experiment 1: Latent-Space DPS

### 4.1 Method

The SPAD conditioning image is encoded through the VAE encoder once, producing a latent reference z_SPAD. At each denoising step, a simple L2 gradient in latent space corrects the velocity:

```
x̂₀ = x_t − σ · v_θ
correction = −η · normalize(x̂₀ − z_SPAD)
v' = v + correction
```

This is a **heuristic approximation** — it does not decode through the VAE or compute the true measurement likelihood.

Implementation: `diffsynth/diffusion/latent_dps.py`, `validate_dps.py`

### 4.2 Configuration

| Parameter | Value |
|-----------|-------|
| Schedule | `linear_decay` (1.0 → 0.0 over all steps) |
| Step range | 0–28 (all steps) |
| η values tested | 0.01, 0.05, 0.1, 0.5, 1.0 |
| Gradient normalization | PaDIS (divide by mean \|grad\|) |

### 4.3 Results (N=776, seed 42)

| Config | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | CFID ↓ | Meas. NLL ↓ |
|--------|--------|--------|---------|-------|--------|-------------|
| **Baseline** | 17.89 | 0.5962 | 0.4152 | 66.84 | 151.94 | **0.7470** |
| η = 0.01 | 18.02 | 0.5957 | 0.4132 | 65.81 | 151.66 | 0.7488 |
| η = 0.05 | 18.02 | 0.5958 | 0.4132 | 65.76 | 151.39 | 0.7488 |
| η = 0.1 | 18.02 | 0.5958 | 0.4132 | 65.86 | 151.45 | 0.7489 |
| η = 0.5 | 18.03 | 0.5961 | 0.4131 | 65.83 | 151.50 | 0.7490 |
| η = 1.0 | 18.05 | 0.5969 | 0.4131 | 65.97 | 151.35 | 0.7493 |

### 4.4 Analysis

**Quality metrics**: Latent DPS produces small, consistent improvements: +0.13–0.16 dB PSNR, −0.002 LPIPS, −0.5 to −0.6 CFID. However, these improvements are **within the seed-to-seed variance** of the baseline (CFID std = ±1.08), making them statistically insignificant.

**η insensitivity**: Results are virtually identical across two orders of magnitude of η (0.01 to 1.0). The PaDIS normalization makes the correction unit-scale, and the latent-space L2 gradient is so smooth that the direction barely changes with scale.

**Measurement NLL worsens**: NLL increases from 0.7470 to 0.7488–0.7493, indicating that the latent-space heuristic does NOT improve physics consistency. The L2 objective `||x̂₀ − z_SPAD||²` in latent space does not correspond to the actual measurement likelihood in pixel space.

**Conclusion**: Latent-space DPS is effectively a no-op. The guidance is too weak and too approximate to meaningfully modify outputs.

---

## 5. Experiment 2: Pixel-Space DPS

### 5.1 Method

At each denoising step, the predicted clean latent x̂₀ is decoded through the VAE to pixel space. The Bernoulli measurement NLL is computed against the actual SPAD observation, and the gradient is backpropagated through the VAE decoder to latent space.

```
x̂₀ = x_t − σ · v_θ
I = mean(sRGB_to_linear(VAE_decode(x̂₀)))     [grayscale intensity]
H = softplus(α·I + β)
NLL = −[y·log(1−exp(−H)) + (1−y)·(−H)]       [per-pixel Bernoulli]
correction = +η · normalize(∇_x NLL)
v' = v + correction
```

Implementation: `diffsynth/diffusion/flow_dps.py`, `validate_flow_dps.py`

**VRAM management**: Each DPS step requires offloading the DiT (~10 GB) and loading the VAE decoder (~0.5 GB) for the gradient computation, then swapping back. The pipeline's `@torch.no_grad()` decorator is overridden with `torch.enable_grad()` for the DPS gradient.

### 5.2 The Dark-Collapse Problem

The initial pixel-space DPS implementation used the standard ("full") NLL over all pixels. This caused **dark collapse**: outputs became nearly black.

**Root cause**: With single-frame SPAD (N=1), typically 80–90% of pixels are 0 (no detection). The NLL gradient for SPAD=0 pixels pushes intensity **down** (darker to explain non-detection), while SPAD=1 pixels push intensity **up**. With the overwhelming majority being SPAD=0, the net gradient pushes the entire image dark.

**Solution**: Two new NLL modes were implemented:
- **`balanced`**: Reweights so SPAD=0 and SPAD=1 pixel groups contribute equally (50/50) to the loss
- **`detections`**: Only computes NLL on SPAD=1 pixels (ignores non-detections entirely)

### 5.3 Hyperparameter Sweep (N=10, quick screening)

A focused sweep tested 4 configurations to find viable hyperparameters:

| Config | PSNR ↑ | ΔPSNR | Meas. NLL | % Better | sec/sample |
|--------|--------|-------|-----------|----------|------------|
| balanced η=0.1, all steps | 18.63 | +0.68 | 0.777 | 40% | 60 |
| balanced η=0.1, steps 4–14 | 18.71 | +0.76 | 0.781 | 40% | 28 |
| detections η=0.1, all steps | 18.69 | +0.74 | 0.813 | 60% | 60 |
| **detections η=0.1, steps 4–14** | **18.90** | **+0.95** | 0.793 | **60%** | **28** |
| balanced η=0.5, all steps | *(collapsed to noise/dark)* | — | — | — | — |

**Key findings from sweep**:
- `detections` mode outperforms `balanced` (60% vs 40% image win rate)
- Mid-step timing (steps 4–14) is better than all steps AND faster (28s vs 60s per sample)
- η=0.5 with balanced mode still collapses, confirming the guidance scale must be small
- The probing-informed timing (depth encoding peaks at t=4–9) is validated

### 5.4 Full Validation (N=776, seed 42)

Best configuration: `detections` mode, η=0.1, constant schedule, steps 4–14.

| Metric | Baseline | Pixel DPS | Δ | Improved? |
|--------|----------|-----------|---|-----------|
| PSNR (dB) | 17.75 | **18.18** | **+0.43** | YES |
| SSIM | 0.5983 | 0.5960 | −0.002 | ~same |
| LPIPS | 0.4137 | **0.4097** | **−0.004** | YES |
| FID | 66.84 | **65.62** | **−1.22** | YES |
| **CFID** | **151.94** | **150.53** | **−1.41** | **YES** |
| Meas. NLL | 0.7470 | 0.7548 | +0.008 | no |

### 5.5 Analysis

**Quality improvement across the board**: Pixel-space DPS with detection-only guidance improves 4 of 5 quality metrics, including CFID (our primary metric). The CFID improvement of −1.41 exceeds the baseline's seed-to-seed standard deviation (±1.08), suggesting a meaningful effect.

**Comparison to latent DPS**: Pixel DPS achieves roughly **2× the CFID improvement** of the best latent DPS configuration (−1.41 vs −0.59), confirming that operating in pixel space with the correct physics objective matters.

**Measurement NLL increases**: This is expected and not a concern. The `detections` mode only guides SPAD=1 pixels. The full NLL (which also scores SPAD=0 pixels) was not optimized. Moreover, we are optimizing for *reconstruction quality* (CFID, LPIPS), not raw measurement consistency.

**Probing-informed timing works**: Steps 4–14 (of 28) was chosen based on the probing finding that depth information peaks at t=4–9 in the DiT. This timing outperforms full-step guidance both in quality and speed (2× faster due to fewer DPS steps).

**Qualitative observations**: DPS tends to help most on difficult/dark images where the baseline struggles, and can hurt on easy images where the baseline is already good. Across 776 images, the net effect is positive.

### 5.6 Pixel DPS vs Latent DPS: Why the Difference?

| Aspect | Latent DPS | Pixel DPS |
|--------|-----------|-----------|
| **Objective** | ||x̂₀ − z_SPAD||² (heuristic L2) | −log p(y \| D(x̂₀)) (true Bernoulli NLL) |
| **Physics model** | None (operates on latent codes) | Full SPAD Bernoulli (sRGB→linear→softplus→Bernoulli) |
| **VAE decode** | Not needed | Required (gradient through VAE decoder) |
| **Speed** | ~2s/sample (no extra model loading) | ~28s/sample (VAE swap each step) |
| **CFID improvement** | −0.59 | **−1.41** (2.4× better) |
| **Measurement NLL** | Worsens (+0.002) | Worsens (+0.008) |

The latent-space L2 objective has no physics grounding — it measures similarity in VAE latent space, which has no known relationship to the SPAD measurement likelihood. Pixel-space DPS directly optimizes the correct physics objective.

---

## 6. Experiment 3: Per-Frame Consistency Training

### 6.1 Method

A LoRA fine-tuning run starting from the baseline epoch-15 checkpoint adds a per-frame consistency regularization:

```
L = L_flow_match(F1) + λ · ||v_θ(z_t, t, F1) − v_θ(z_t, t, F2)||²
```

Implementation: `diffsynth/diffusion/consistency_loss.py`, `train_consistency.py`, `train_consistency.sh`

### 6.2 Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | SceneAware-RAW epoch-15 |
| λ (consistency weight) | 0.1 |
| Learning rate | 5e-5 |
| LoRA rank | 32 |
| Epochs | 30 |
| Max pixels | 262,144 (512×512) |
| Frame folders | 7 temporal realizations (bits, bits_frame_{1000,4000,5000,8000,12000,16000}) |
| F2 branch | Stop-gradient (prevents collapse) |
| F1/F2 selection | Random per sample per epoch |
| Scheduler weighting | Applied to both flow-match and consistency terms |

### 6.3 Dataset

`PairedSPADDataset` loads:
- Ground-truth RGB image
- SPAD frame F1 (random folder selection)
- SPAD frame F2 (different random folder from same scene)

7 frame folders correspond to different temporal windows from the raw SPAD data (frames 0–0, 1000–1000, 4000–4000, 5000–5000, 8000–8000, 12000–12000, 16000–16000).

### 6.4 Results (N=776, seed 42)

Only epoch-0 has been evaluated so far:

| Metric | Baseline | Consistency (epoch-0) | Δ | Improved? |
|--------|----------|-----------------------|---|-----------|
| PSNR (dB) | 17.89 | 17.72 | −0.17 | no |
| SSIM | 0.5962 | 0.5888 | −0.007 | no |
| LPIPS | 0.4152 | 0.4215 | +0.006 | no |
| FID | 66.84 | 66.51 | −0.33 | marginal |
| **CFID** | 151.94 | 154.99 | **+3.05** | **no (worse)** |
| Meas. NLL | 0.7470 | **0.7460** | **−0.001** | marginal |

### 6.5 Consistency + Latent DPS (N=776, seed 42)

Combining consistency epoch-0 with latent DPS (η=1.0, linear_decay):

| Metric | Baseline | Consist. only | Consist. + DPS | Δ vs baseline |
|--------|----------|---------------|----------------|---------------|
| PSNR (dB) | 17.89 | 17.72 | 17.86 | −0.03 |
| SSIM | 0.5962 | 0.5888 | 0.5898 | −0.006 |
| LPIPS | 0.4152 | 0.4215 | 0.4199 | +0.005 |
| FID | 66.84 | 66.51 | 65.75 | −1.09 |
| **CFID** | 151.94 | 154.99 | 154.15 | **+2.20 (worse)** |

### 6.6 Analysis

**Consistency training degrades quality**: Epoch-0 shows worse metrics across nearly all dimensions, with a substantial CFID regression (+3.05). The marginal NLL improvement (−0.001) is within noise.

**Important caveat — epoch-0**: This is the *very first* checkpoint (one epoch of fine-tuning). The consistency loss adds a new objective that the model hasn't yet learned to balance. Later epochs (up to epoch-29 are available) may show improvement as the model adapts.

**DPS partially recovers degradation**: Adding latent DPS to the consistency checkpoint improves FID (−0.76 vs consistency-only) and partially recovers PSNR, but CFID remains substantially worse than baseline.

**The consistency objective is correct in principle**: The model *should* produce identical outputs for different SPAD realizations of the same scene. The training loss formulation (stop-gradient on F2, scheduler-weighted, random F1/F2 assignment) follows best practices. The poor results likely reflect:
1. Insufficient training (only epoch-0 evaluated)
2. Possible learning rate or λ mismatch
3. The consistency constraint may conflict with the flow-matching objective during early adaptation

### 6.7 Best-of-K NLL Reranking (N=776, 10 seeds)

Best-of-K selects, for each image, the seed whose output has the lowest measurement NLL from the existing 10-seed generations. This is a pure post-processing step with no additional GPU inference.

| Metric | Baseline (seed 42) | Best-of-10 | Δ |
|--------|-------------------|------------|---|
| PSNR (dB) | 17.75 | 15.30 | **−2.45** |
| SSIM | 0.598 | 0.585 | −0.013 |
| LPIPS | 0.414 | 0.433 | +0.019 |
| FID | 66.84 | 66.46 | −0.38 |
| **CFID** | 151.94 | 156.96 | **+5.02 (worse)** |
| Meas. NLL | 0.747 | **0.721** | **−0.026** |

**Seed selection distribution**: Nearly uniform (68–88 images per seed, ~9–11% each). No single seed dominates, indicating NLL-optimal images are distributed across the stochastic space.

### 6.8 Best-of-K Analysis

**NLL-optimal selection is adversarial to image quality.** While measurement NLL improves by 3.5%, all perceptual quality metrics degrade — most dramatically PSNR (−2.45 dB) and CFID (+5.02). This is a key negative result.

**Why this happens**: Minimizing NLL encourages images that best explain the observed SPAD detections under the Bernoulli model. For single-frame SPAD (N=1), this pushes images toward high-contrast outputs that maximize detection probability at SPAD=1 pixels and minimize it at SPAD=0 pixels. This is physically correct but perceptually distorted — the model is being rewarded for extreme brightness at detection sites and darkness elsewhere.

**Implication for DPS**: This validates the dark-collapse insight from pixel-space DPS experiments. NLL alone (even balanced or detection-weighted) is an imperfect proxy for image quality. The success of low-η DPS (η=0.1) comes from using the NLL gradient as a *gentle nudge* rather than a hard optimization target.

---

## 7. Consolidated Results

### 7.1 Full Comparison Table (all experiments, N=776, seed 42)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | CFID ↓ | Meas. NLL ↓ |
|--------|--------|--------|---------|-------|--------|-------------|
| **Baseline** | 17.75 | 0.598 | 0.414 | 66.84 | 151.94 | 0.747 |
| Latent DPS η=0.01 | 18.02 | 0.596 | 0.413 | 65.81 | 151.66 | 0.749 |
| Latent DPS η=0.05 | 18.02 | 0.596 | 0.413 | 65.76 | 151.39 | 0.749 |
| Latent DPS η=0.1 | 18.02 | 0.596 | 0.413 | 65.86 | 151.45 | 0.749 |
| Latent DPS η=0.5 | 18.03 | 0.596 | 0.413 | 65.83 | 151.50 | 0.749 |
| Latent DPS η=1.0 | 18.05 | 0.597 | 0.413 | 65.97 | 151.35 | 0.749 |
| **Pixel DPS (detections, η=0.1, steps 4–14)** | **18.18** | 0.596 | **0.410** | **65.62** | **150.53** | 0.755 |
| Consistency (epoch-0) | 17.72 | 0.589 | 0.422 | 66.51 | 154.99 | 0.746 |
| Consistency + Latent DPS | 17.86 | 0.590 | 0.420 | 65.75 | 154.15 | — |
| Best-of-10 NLL Reranking | 15.30 | 0.585 | 0.433 | 66.46 | 156.96 | **0.721** |

### 7.2 Relative to Baseline Variance

Baseline 10-seed statistics: CFID = 152.04 ± 1.08, PSNR = 17.86 ± 0.09

- **Pixel DPS CFID** (150.53): **1.4σ below** the baseline mean — likely meaningful
- **Best Latent DPS CFID** (151.35): 0.6σ below — within noise
- **Consistency CFID** (154.99): 2.7σ above — significantly worse

---

## 8. Limitations

### 8.1 Single-Seed Evaluation for DPS

All DPS experiments use seed 42 only. Given the baseline CFID variance of ±1.08 across seeds, the pixel DPS improvement of −1.41 is suggestive but not conclusive. Multi-seed evaluation of the best DPS configuration would strengthen the result.

### 8.2 Narrow Hyperparameter Exploration

- **Latent DPS**: Only tested `linear_decay` schedule. The probing results suggest `ramp_up` or mid-step-only guidance should be better, but this was never evaluated for latent DPS.
- **Pixel DPS**: Only the best sweep configuration was run at scale. Additional schedules (ramp_up, cosine) and step ranges were not fully evaluated.
- **Consistency**: Only λ=0.1 was tested. The optimal balance between flow-matching and consistency objectives is unknown.

### 8.3 Consistency Training Under-Explored

Only **epoch-0** of 30 has been evaluated. The consistency model has 30 checkpoints available (`epoch-0` through `epoch-29`). The epoch-0 result may not be representative of the converged model. A training curve (evaluate every 5 epochs) would clarify whether the consistency objective eventually helps.

### 8.4 Single-Frame SPAD Limitation

All experiments use single-frame (N=1) binary SPAD data. The Bernoulli observation from a single frame is extremely noisy (~50% photon detection probability for a moderately lit pixel), which limits the signal-to-noise ratio of the DPS gradient. Multi-frame accumulation would provide a much stronger measurement signal.

### 8.5 Cross-Frame Variance Measurement (In Progress)

The consistency training is designed to reduce output variation across different SPAD frame realizations. Cross-frame generation (7 different binary frame realizations × 776 scenes) is currently running for the baseline model. Once complete, this will provide: (1) cross-frame output variance statistics, (2) a new probing target to understand where in the DiT cross-frame information is encoded, (3) a baseline for comparison against consistency-trained checkpoints.

### 8.6 Pixel DPS Speed

Pixel-space DPS requires VAE decode + gradient computation at each active denoising step, plus model swapping (offload DiT, load VAE decoder, swap back). This adds ~28s per sample (vs ~2s for latent DPS, ~0.6s for baseline). For the mid-step configuration (10 active steps out of 28), this is manageable, but full-step pixel DPS is prohibitively slow (~60s/sample).

### 8.7 NLL Mode Selection

The `detections` NLL mode (only guide SPAD=1 pixels) was chosen because it prevents dark collapse, but it has a theoretical limitation: it only encourages the model to explain *detections*, not *non-detections*. An ideal solution would weight both terms but with proper class-balance correction. The `balanced` mode attempts this but showed a lower image win rate in the sweep (40% vs 60%).

---

## 9. Missing Experiments

### 9.1 High Priority

| Experiment | Description | Estimated Time | Status |
|------------|-------------|----------------|--------|
| **Consistency epoch sweep** | Evaluate epochs 5, 10, 15, 20, 25, 29 | ~20 min per epoch | **Queued** (overnight pipeline) |
| **Multi-seed pixel DPS** | Run pixel DPS (best config) with 3–5 additional seeds | ~6h per seed | Not started |
| **Cross-frame variance** | Generate from 7 different SPAD frames per scene, measure output variance | ~13h | **In progress** (overnight pipeline) |
| ~~**Best-of-K NLL Reranking**~~ | Select best-NLL output from 10 seeds per image | 0 (post-processing) | **Done** (Section 6.7–6.8) — negative result |

### 9.2 Medium Priority

| Experiment | Description | Estimated Time | Why It Matters |
|------------|-------------|----------------|----------------|
| **Pixel DPS + ramp_up schedule** | Test ramp_up (probing-informed) vs constant for pixel DPS | ~6h | The sweep used constant; ramp_up might be better |
| **Pixel DPS + consistency** | Apply pixel DPS to best consistency checkpoint | ~6h | Can training + inference-time techniques be combined? |
| **Latent DPS with ramp_up** | Re-run latent DPS with ramp_up and mid-step timing | ~1h | Fair comparison to pixel DPS (latent DPS used linear_decay) |
| **Consistency λ sweep** | Test λ = {0.01, 0.05, 0.5, 1.0} | ~12h training each | Find optimal consistency weight |

### 9.3 Lower Priority

| Experiment | Description | Why |
|------------|-------------|-----|
| **Multi-frame DPS** | Use N>1 accumulated SPAD frames for DPS guidance | Stronger measurement signal, but requires multi-frame data |
| **Balanced NLL with smaller η** | Test `balanced` mode with η=0.01–0.05 | May avoid dark collapse while using both SPAD=0 and SPAD=1 signal |
| **Pixel DPS with L2 + NLL** | Combine L2 and NLL losses in pixel space | May provide smoother gradients |
| **DPS step ablation** | Systematic sweep of (start, stop) step pairs | Map the full landscape of timing effects |

---

## 10. Code Status & Known Issues

### 10.1 Code Patched During This Session

The following files were modified to support the experiments:

| File | Change | Status |
|------|--------|--------|
| `diffsynth/diffusion/flow_dps.py` | Added `nll_mode` parameter with `balanced` and `detections` modes | Working, tested |
| `validate_flow_dps.py` | Added `--nll_mode` CLI argument, passes through to `compute_dps_correction` | Working, tested |
| `run_metrics.py` | Added measurement NLL computation, JSON output | Working, tested |
| `sweep_pixel_dps.py` | New file for hyperparameter sweep | Working (sweep script, not production) |
| `best_of_k.py` | New file for best-of-K NLL reranking | Working, tested |
| `validate_crossframe.py` | New file for cross-frame generation | Working, running |
| `compute_crossframe_targets.py` | New file for cross-frame variance probing targets | Written, not yet run |
| `overnight_pipeline.sh` | Chains cross-frame → probing → consistency epoch sweep | Running |

### 10.2 Consistency Training Code Status

The consistency training pipeline (`train_consistency.py`, `consistency_loss.py`, `paired_spad_dataset.py`) was written in a previous session. **Potential issues to verify**:

1. **F2 conditioning path**: The code encodes F2 through the VAE encoder as a *latent*, then passes it as `controlnet_conditionings_f2`. This is correct for the consistency loss (both F1 and F2 velocity predictions use the same latent representation), but the ControlNet conditioning path should be verified to ensure F2 is processed the same way as F1 by the ControlNet.

2. **Stop-gradient correctness**: The F2 branch uses `torch.no_grad()` which prevents gradient flow. This is intentional (prevents collapse) but means the model only learns to make F1's prediction match F2's, not vice versa. Over training, random F1/F2 assignment covers both directions.

3. **Training weight application**: Both the flow-matching and consistency loss terms are multiplied by `scheduler.training_weight(timestep)`. This is correct — it ensures the balance between the two losses is timestep-invariant.

### 10.3 Latent DPS Schedule Mismatch

The latent DPS experiments (`run_physics_ablation.sh`) all used `--dps_schedule linear_decay`, which applies maximum guidance at the start of denoising when representations are most noisy and least informative. The probing results suggest that information encoding peaks at mid-steps (t=4–9 for depth), so `ramp_up` or mid-step-only guidance should be more effective. This was tested for pixel DPS but **not** for latent DPS.

---

## 11. Reproduction Instructions

### 11.1 Latent DPS (All Configurations)

```bash
# Run full ablation matrix (baseline + 5 eta values)
bash run_physics_ablation.sh \
    models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors
```

### 11.2 Pixel DPS (Best Configuration)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth

python validate_flow_dps.py \
    --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
    --output_dir ./validation_outputs_flow_dps/detections_eta0.1_mid4_14 \
    --dps_eta 0.1 \
    --dps_schedule constant \
    --dps_start_step 4 \
    --dps_stop_step 14 \
    --nll_mode detections \
    --seed 42 --steps 28

python run_metrics.py ./validation_outputs_flow_dps/detections_eta0.1_mid4_14 --save
```

### 11.3 Pixel DPS Hyperparameter Sweep

```bash
python sweep_pixel_dps.py \
    --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
    --max_samples 10 \
    --output_root ./sweep_pixel_dps_results \
    --baseline_dir ./validation_outputs_multiseed/seed_42
```

### 11.4 Consistency Training

```bash
bash train_consistency.sh
```

### 11.5 Consistency Evaluation

```bash
python validate_lora.py \
    --lora_checkpoint models/train/FLUX-SPAD-LoRA-Consistency/epoch-0.safetensors \
    --output_dir ./validation_outputs_consistency/epoch-0 \
    --seed 42 --steps 28

python run_metrics.py ./validation_outputs_consistency/epoch-0 --save
```

### 11.6 Best-of-K NLL Reranking

```bash
python best_of_k.py \
    --multiseed_dir ./validation_outputs_multiseed \
    --output_dir ./validation_outputs_best_of_k

python run_metrics.py ./validation_outputs_best_of_k --save
```

### 11.7 Cross-Frame Variance Generation

```bash
python validate_crossframe.py \
    --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
    --output_dir ./validation_outputs_crossframe/baseline \
    --seed 42

# Compute variance targets for probing
python compute_crossframe_targets.py \
    --crossframe_dir ./validation_outputs_crossframe/baseline
```

### 11.8 Recompute Metrics (All Multiseed)

```bash
for seed_dir in validation_outputs_multiseed/seed_*; do
    python run_metrics.py "$seed_dir" --save --batch-size 16
done
```

### 11.9 Qualitative Comparison Grid

```bash
python sweep_pixel_dps_results/make_qualitative_grid.py
# Output: sweep_pixel_dps_results/qualitative_comparison.png
```

---

## 12. Key Takeaways

1. **Pixel-space DPS with detection-only NLL is the best inference-time improvement** we have found: CFID 151.94 → 150.53 (−1.41), with improvements across PSNR, LPIPS, and FID.

2. **Latent-space DPS is ineffective** — the heuristic L2 in latent space has no physics grounding and produces negligible, statistically insignificant improvements.

3. **The dark-collapse problem** in pixel-space DPS is caused by class imbalance in single-frame SPAD data (majority SPAD=0 pixels). The `detections` NLL mode (only guide where SPAD=1) resolves this effectively.

4. **Probing-informed timing matters**: Applying DPS only at steps 4–14 (where the probing shows depth information is strongest) outperforms full-step guidance and is 2× faster.

5. **Consistency training needs more evaluation**: Only epoch-0 has been tested and shows degradation. 30 checkpoints are available. Epoch sweep is queued in the overnight pipeline.

6. **Best-of-K NLL reranking destroys quality**: Selecting NLL-optimal outputs from 10 seeds dramatically improves NLL (0.747→0.721) but degrades CFID by +5.02 and PSNR by −2.45 dB. This confirms that NLL alone is adversarial to perceptual quality for single-frame SPAD, validating the need for low-η DPS rather than hard NLL optimization.

6. **Multi-seed evaluation is needed** to establish statistical significance of the pixel DPS improvement over baseline variance.

---

## Appendix A: Directory Map

```
validation_outputs_multiseed/
├── aggregated_metrics.json          # 10-seed aggregate
├── baseline -> seed_42              # symlink for convenience
├── seed_{0,13,23,42,55,67,77,88,99,123}/
│   ├── input/                       # 776 SPAD conditioning images
│   ├── output/                      # 776 generated images
│   ├── ground_truth/                # 776 GT RGB images
│   ├── metrics.json                 # Full metrics including NLL
│   └── metrics.txt

validation_outputs_physics_ablation/
├── baseline/                        # Same as seed_42
├── dps_eta{0.01,0.05,0.1,0.5,1.0}/ # Latent DPS ablation
│   ├── {input,output,ground_truth}/
│   └── metrics.json

validation_outputs_flow_dps/
├── detections_eta0.1_mid4_14/       # Best pixel DPS (776 images)
│   ├── {input,output,ground_truth}/
│   └── metrics.json
└── ramp_up_eta0.1/                  # Partial run (197 images, abandoned)

validation_outputs_consistency/
└── epoch-0/                         # Only epoch evaluated
    ├── {input,output,ground_truth}/
    └── metrics.json

validation_outputs_consistency_dps/
└── eta1.0/                          # Consistency + latent DPS
    └── metrics.txt

validation_outputs_best_of_k/
├── input/                          # Best-of-K selected inputs
├── output/                         # Best-of-K selected outputs
├── ground_truth/                   # Shared GT
├── best_of_k_meta.json            # Per-image selection + stats
└── metrics.json                    # Full metrics

validation_outputs_crossframe/
└── baseline/                       # Cross-frame generation (in progress)
    ├── bits -> ../../validation_outputs_multiseed/seed_42
    ├── bits_frame_{1000,4000,5000,8000,12000,16000}/
    │   ├── input/                  # 776 per frame
    │   └── output/                 # 776 per frame
    └── ground_truth/               # Shared GT

sweep_pixel_dps_results/
├── balanced_eta0.1_{full,mid_4_14}/ # Sweep configs (10 images each)
├── detections_eta0.1_{full,mid_4_14}/
├── balanced_eta0.5_full/            # Partial (collapsed)
├── qualitative_comparison.png       # Visual comparison grid
└── make_qualitative_grid.py

models/train/
├── FLUX-SPAD-LoRA-SceneAware-RAW/
│   └── epoch-15.safetensors         # Baseline checkpoint
└── FLUX-SPAD-LoRA-Consistency/
    ├── epoch-0.safetensors          # ... through ...
    └── epoch-29.safetensors         # 30 consistency checkpoints
```

## Appendix B: Sweep Quick-Metrics (N=10 each, not FID/CFID)

From the hyperparameter sweep console output:

| Config | PSNR | ΔPSNR vs baseline | Meas. NLL | % Images Better | sec/sample |
|--------|------|-------------------|-----------|-----------------|------------|
| balanced η=0.1 full steps | 18.63 | +0.68 | 0.777 | 40% | 60 |
| balanced η=0.1 steps 4–14 | 18.71 | +0.76 | 0.781 | 40% | 28 |
| detections η=0.1 full steps | 18.69 | +0.74 | 0.813 | 60% | 60 |
| **detections η=0.1 steps 4–14** | **18.90** | **+0.95** | 0.793 | **60%** | **28** |
| balanced η=0.5 full steps | *(collapsed)* | — | — | — | — |

Note: These metrics use grayscale PSNR and measurement NLL only (FID/CFID require more samples). The ΔPSNR values are relative to baseline seed_42 on the same 10 images.
