# Consolidated Experiment Results

**Updated**: 2026-03-31
**Validation set**: 776 images, scene-aware split (no data leakage)
**GPU**: RTX 5090 (32GB)

---

## 1. Main Reconstruction Results

### 1.1 Baseline and Core Methods

| Experiment | PSNR | SSIM | LPIPS | FID | CFID | Notes |
|------------|------|------|-------|-----|------|-------|
| **Baseline (LoRA, seed 42)** | **17.89** | **0.596** | **0.415** | **66.84** | **151.94** | ControlNet + LoRA rank-32, 28 Euler steps |
| Baseline (10-seed mean) | 17.99 +/- 0.09 | 0.596 | 0.415 | 66.29 | 152.04 | Seeds: 0,13,23,42,55,67,77,99,123 |
| FlowDPS (eta=0.1) | **18.18** | 0.596 | **0.410** | **65.62** | **150.53** | Best DPS config, mid-step guidance t=[4,14] |
| Consistency (epoch 0) | 17.72 | 0.589 | 0.422 | 66.51 | 154.99 | Consistency-trained LoRA |
| Best-of-10 (NLL) | 15.30 | 0.585 | 0.433 | 66.46 | 156.96 | NLL reranking from 10 seeds per image |

### 1.2 DPS Physics Guidance Sweep

| Eta | PSNR | SSIM | LPIPS | FID | CFID |
|-----|------|------|-------|-----|------|
| 0 (baseline) | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| 0.01 | 18.02 | 0.596 | 0.413 | 65.81 | 151.66 |
| 0.05 | 18.02 | 0.596 | 0.413 | 65.76 | 151.39 |
| 0.1 | 18.02 | 0.596 | 0.413 | 65.86 | 151.45 |
| 0.5 | 18.03 | 0.596 | 0.413 | 65.83 | 151.50 |
| **1.0** | **18.05** | **0.597** | **0.413** | 65.97 | **151.35** |
| FlowDPS 0.1 (mid-step) | **18.18** | 0.596 | **0.410** | **65.62** | 150.53 |

**Observation**: DPS provides consistent but modest improvement (~+0.15 dB PSNR). FlowDPS mid-step guidance is the best variant (+0.29 dB over baseline).

### 1.3 Frame-Count Ablation (Zero-Shot Multi-Frame)

| Frames (N) | PSNR | SSIM | LPIPS | FID | CFID |
|------------|------|------|-------|-----|------|
| 1 (single binary) | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| 4 | 17.82 | 0.636 | 0.376 | 71.44 | 138.05 |
| 16 | 16.91 | 0.651 | 0.359 | 74.85 | 131.04 |
| 64 | 15.47 | 0.640 | 0.346 | 74.50 | 120.90 |
| 256 | 14.12 | 0.605 | 0.339 | 70.66 | 110.11 |
| 1000 | 13.04 | 0.551 | 0.347 | 68.52 | 108.11 |

**Observation**: PSNR *decreases* with more frames (model trained on N=1 binary, not multi-frame). But LPIPS and CFID *improve* — perceptual quality gets better as the model hallucinates less. Classic perception-distortion tradeoff.

---

## 2. Ablation Studies

### 2.1 img2img Ablation (No ControlNet)

FLUX img2img with LoRA-on-DiT (rank-32). SPAD goes through VAE encoder as denoising start point. No ControlNet.

| Strength | PSNR | SSIM | LPIPS | FID | CFID |
|----------|------|------|-------|-----|------|
| 0.3 | 7.59 | 0.013 | 1.093 | 388.42 | 351.22 |
| 0.5 | 7.50 | 0.013 | 1.103 | 356.75 | 365.46 |
| 0.7 | 7.44 | 0.016 | 1.088 | 322.68 | 391.50 |
| 0.8 | 7.52 | 0.026 | 1.055 | 283.86 | 396.38 |
| 0.9 | 8.26 | 0.118 | 0.928 | 182.24 | 355.44 |
| 1.0 | 10.28 | 0.392 | 0.692 | 107.85 | 308.51 |

**Conclusion**: Catastrophic failure at all strengths. Even strength=1.0 (maximum denoising, minimal SPAD signal) only reaches PSNR 10.28 — still far below baseline 17.89. Confirms VAE domain gap: binary SPAD produces OOD latents unsuitable as denoising start points. See gQIR paper analysis in `literature_gqir_vae.md`.

### 2.2 No-LoRA Ablation (Frozen ControlNet, No Adaptation)

Pretrained ControlNet Union Alpha + frozen FLUX DiT, NO LoRA. Tests whether the off-the-shelf ControlNet can handle SPAD without any fine-tuning.

| Mode | PSNR | SSIM | LPIPS | FID | CFID |
|------|------|------|-------|-----|------|
| gray | 8.57 | 0.046 | 1.088 | 313.88 | 388.94 |
| lq (low quality) | 8.74 | 0.112 | 0.848 | 317.66 | 373.48 |
| canny | **10.81** | **0.189** | 0.797 | **274.57** | **363.70** |
| tile | 9.68 | 0.188 | **0.751** | 284.79 | 352.09 |
| depth | 10.13 | 0.171 | 0.784 | 288.28 | 373.83 |

**Key findings**:
1. **LoRA is essential**: Best no-LoRA (canny, 10.81 dB) is 7.08 dB below our LoRA baseline (17.89 dB). The pretrained ControlNet cannot interpret SPAD data without adaptation.
2. **"gray" mode is worst** (8.57 dB) — ironic since it's what we train with. The pretrained gray mode expects natural grayscale images, not binary SPAD.
3. **"canny" mode is best no-LoRA** (10.81 dB) — makes sense because SPAD binary frames have edge-like structure (sparse white pixels on black background), which is closest to what a canny edge map looks like.
4. **"tile" has best LPIPS** (0.751) — tile mode tries to preserve spatial layout, producing somewhat coherent but wrong images.
5. **All modes fail catastrophically** — even the best is PSNR 10.81, confirming that LoRA fine-tuning is what teaches the model to understand SPAD data.

### 2.3 Ablation Summary

| Configuration | PSNR | Delta vs Baseline | What It Tests |
|--------------|------|-------------------|---------------|
| **Baseline (ControlNet + LoRA)** | **17.89** | — | Full pipeline |
| FlowDPS (eta=0.1) | 18.18 | +0.29 | Physics guidance at inference |
| No-LoRA (best: canny) | 10.81 | **-7.08** | LoRA adaptation necessity |
| img2img (best: s=1.0) | 10.28 | **-7.61** | ControlNet vs VAE pathway |
| img2img (worst: s=0.7) | 7.44 | **-10.45** | VAE domain gap severity |
| No-LoRA gray | 8.57 | **-9.32** | Pretrained ControlNet on SPAD |

**Takeaway**: LoRA on ControlNet provides **+7-10 dB** over any non-adapted configuration. ControlNet pathway provides **+7+ dB** over img2img (VAE pathway). Both are essential.

---

## 3. Linear Probing Results

### 3.1 FLUX: Global Probing (Best R^2)

| Target | Main (LoRA) | Control (no LoRA) | No-ControlNet | LoRA Delta |
|--------|-------------|-------------------|---------------|------------|
| Bit density | 0.998 | 0.999 | -0.060 | ~0 |
| Depth | 0.437 | 0.129 | -0.040 | **+0.308** |
| Variance (seed) | 0.424 | 0.446 | -0.360 | -0.022 |
| Cross-frame variance | 0.292 | 0.222 | -0.161 | +0.070 |

### 3.2 FLUX: Spatial Probing (Best R^2)

| Target | Main (LoRA) | Best Block |
|--------|-------------|-----------|
| Bit density | 0.959 | single_37 @ t=0 |
| **Depth** | **0.685** | single_9 @ t=14 |
| Variance (seed) | 0.506 | single_28 @ t=4 |
| Cross-frame variance | 0.359 | single_28 @ t=9 |

### 3.3 FLUX: ControlNet Block Probing (Best R^2)

| Target | Best R^2 | Best Block |
|--------|---------|-----------|
| Bit density | **0.9995** | cn_joint_1 @ t=0 |
| Cross-frame variance | **0.360** | cn_single_1 @ t=24 |
| Variance (seed) | 0.465 | cn_joint_2 @ t=27 |
| Depth | 0.227 | cn_single_5 @ t=19 |

### 3.4 SD1.5 Cross-Architecture Comparison (Best R^2)

| Target | SD1.5 Global | FLUX Global | SD1.5 Spatial | FLUX Spatial |
|--------|-------------|-------------|---------------|-------------|
| Bit density | 0.993 | 0.998 | 0.974 | 0.959 |
| Depth | 0.375 | 0.437 | **0.727** | 0.685 |
| Variance (seed) | **0.472** | 0.424 | 0.493 | 0.506 |
| Cross-frame var | 0.293 | 0.292 | 0.279 | **0.359** |

### 3.5 Global vs Spatial Improvement

| Target | Global R^2 | Spatial R^2 | Gain |
|--------|-----------|------------|------|
| Bit density | 0.998 | 0.959 | -0.039 (ceiling) |
| Depth | 0.437 | **0.685** | **+0.248 (+57%)** |
| Variance | 0.424 | 0.506 | +0.082 (+19%) |
| Cross-frame var | 0.292 | 0.359 | +0.067 (+23%) |

### 3.6 Object Recognition (Best Balanced Accuracy, FLUX Main)

| Object | Balanced Acc | Object | Balanced Acc |
|--------|-------------|--------|-------------|
| Table | 0.982 | Tree | 0.813 |
| Chair | 0.953 | Cable | 0.786 |
| Building | 0.936 | Trash can | 0.750 |
| Keyboard | 0.881 | TV | 0.738 |
| Mouse | 0.861 | Door | 0.629 |
| Bench | 0.845 | Computer monitor | 0.565 |

---

## 4. Experiments Not Yet Run

| Experiment | Status | Est. Time |
|------------|--------|-----------|
| Consistency epoch sweep (ep 5,10,15,20,25,29) | Checkpoints exist, inference not run | ~2h |
| Consistency + DPS | Inference not run | ~30min |
| OD filter ablation (4 models x 4 datasets) | Models trained, inference not run | ~3h |
| SD1.5 baseline re-evaluation (scene-aware split) | Script ready | ~1h |

---

## 5. Key Figures

All publication figures in `thesis_figures/publication/` (PDF + PNG).
Probing figures in `probing_results_allblocks/probes/`.
SD1.5 probing figures in `spad-diffusion/probing_results_sd15/probes/`.

See `agent/FIGURES.md` for full inventory and regeneration commands.
