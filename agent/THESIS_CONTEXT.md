# THESIS_CONTEXT.md -- Master Project Document
**SPAD Binary Sensor to RGB Image Reconstruction via Diffusion Priors**

**Author**: JW | **Supervisor**: Prof. David Lindell
**Program**: Engineering Science, University of Toronto
**Date**: 2026-03-29 (living document -- update continuously)
**Deadlines**: Practice presentation Mon Mar 31 | Actual presentation Wed Apr 2 | Thesis due Mon Apr 7 | NeurIPS ~May 2026
**Deliverables**: (1) Thesis paper, (2) Presentation slides, (3) Potential NeurIPS 2026 submission

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Architecture and Method](#3-architecture-and-method)
4. [Dataset and Data Pipeline](#4-dataset-and-data-pipeline)
5. [Training Details](#5-training-details)
6. [Experimental Results](#6-experimental-results)
7. [Linear Probing Analysis (Key Contribution)](#7-linear-probing-analysis-key-contribution)
8. [Physics-Guided DPS and Consistency Training](#8-physics-guided-dps-and-consistency-training)
9. [Key Insights and Narrative](#9-key-insights-and-narrative)
10. [Figures Inventory](#10-figures-inventory)
11. [Thesis/Paper Structure](#11-thesispaper-structure)
12. [Presentation Plan](#12-presentation-plan)
13. [Codebase Map](#13-codebase-map)
14. [Current Status and Next Steps](#14-current-status-and-next-steps)
15. [Key Decisions Log](#15-key-decisions-log)
16. [Competitor Landscape](#16-competitor-landscape)
17. [References](#17-references)

---

## 1. Project Overview

We study **generative reconstruction from single-photon measurements** as a testbed for understanding how large diffusion priors interact with extreme inverse problems. Using a **12B-parameter FLUX.1-dev rectified-flow transformer** and an **860M SD1.5 UNet**, both conditioned on SPAD binary frames via **ControlNet + LoRA**, we:

1. **Probe the model's internal representations** to show it implicitly encodes scene geometry (depth R^2=0.685 spatial, 0.727 for SD1.5) despite never receiving depth supervision
2. **Compare cross-architecture representations**: SD1.5 UNet (1.2B) is surprisingly competitive with FLUX DiT (12B), revealing architecture-general ControlNet encoding patterns
3. **Characterize the uncertainty-hallucination tradeoff** in the single-photon regime, including cross-frame variance probing (R^2=0.292) and best-of-K NLL reranking
4. **Introduce physics-consistent posterior sampling** that enforces the Bernoulli photon-detection likelihood at inference time

**The framing is NOT "we applied ControlNet to SPAD"** -- it is **"we used SPAD as a lens to understand generative priors."**

### Three Pillars

| Pillar | Description | Status | Novelty |
|--------|-------------|--------|---------|
| **"What does the model know?"** | AC3D-inspired linear probing of DiT + UNet activations | **Complete** — FLUX (3 conditions) + SD1.5 cross-architecture + cross-frame variance + spatial streaming | First probing of diffusion model conditioned on SPAD; first cross-architecture (UNet vs DiT) probing comparison |
| **Uncertainty-Hallucination-Consistency** | Multi-seed + cross-frame distributional analysis, best-of-K, img2img ablation | **Complete** | Quantified uncertainty in extreme 1-bit regime; img2img ablation justifies ControlNet |
| **Physics-Consistent Posterior Sampling** | Bernoulli likelihood DPS guidance at inference time | **Complete** (modest results) | FlowDPS for SPAD with Bernoulli forward model |

---

## 2. Motivation and Problem Statement

### What are SPADs?

**Single-Photon Avalanche Diodes (SPADs)** are binary sensors that detect individual photons:
- Output: **1 bit per pixel per exposure** ({0, 1} -- photon detected or not)
- Operating model: Bernoulli process with detection probability p = 1 - exp(-H), where H is exposure
- Advantages: Extreme low-light sensitivity, ultra-high-speed capture (MHz frame rates), compact
- Challenge: Reconstructing a full RGB image from a single binary frame is a massively ill-posed inverse problem

### Why this problem is interesting for ML research

1. **Extreme information asymmetry**: 1 bit/pixel input vs 24 bits/pixel output (RGB 8-bit). The measurement provides minimal constraint on the output.
2. **Known physics**: The forward model (Bernoulli detection) is fully characterized, enabling physics-guided approaches.
3. **Quantifiable uncertainty**: Multi-seed sampling from a generative model gives distributional estimates of reconstruction uncertainty.
4. **Clean testbed for probing**: Unlike RGB-conditioned models, SPAD conditioning is so sparse that the model *must* rely on its learned prior, making internal representations more interpretable.

### Real-world dataset

We use a real SPAD dataset captured at the University of Toronto:
- **~2,500 views** of indoor/outdoor scenes
- **512x512 resolution**, up to 20,000 binary frames per view
- Multiple optical density (OD) filter conditions: RAW (no filter), OD0.1, OD0.3, OD0.7
- Ground truth RGB images via log-inversion flux estimation pipeline (hot-pixel suppression, white balance, gamma correction)
- Raw captures stored on remote server (9TB); extracted PNGs local at `spad_dataset/`

---

## 3. Architecture and Method

### 3.1 Model Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            DATA PREPROCESSING                │
                    │  (frozen, no gradients, runs once per batch)  │
                    └──────────────────────────────────────────────┘

  GT RGB (3, 512, 512)                SPAD binary (1-bit, 512x512)
        |                                       |
        v                                       v
  ┌─────────────┐                        ┌─────────────┐
  │ VAE Encoder │ (frozen)               │ VAE Encoder │ (frozen, SAME encoder)
  └─────────────┘                        └─────────────┘
        |                                       |
        v                                       v
  target latent z_gt                     SPAD latent z_spad
  (16, 64, 64)                           (16, 64, 64)
        |                                       |
        |                      ┌────────────────┘
        |                      v
        |        ┌─────────────────────────────┐
        |        │   ControlNet Union Alpha    │ <── LoRA rank-32 (80 layers, TRAINABLE)
        |        │   5 joint + 10 single blocks │
        |        └─────────────┬───────────────┘
        |                      │ conditioning injections
        |                      v
        |     ┌──────────────────────────────────┐
        |     │       FLUX.1-dev DiT Backbone     │  (12B params, frozen FP8)
        |     │  19 joint blocks + 38 single blks  │
        |     │  Hidden dim 3072, 1024 img tokens  │
        |     └───────────────┬────────────────────┘
        |                     │
  ┌─────┘                     v
  │              predicted latent z_pred (16, 64, 64)
  │                           │
  │    TRAINING:              │    INFERENCE:
  │    MSE(z_pred, z_gt)      │    ┌─────────────┐
  │    (flow matching loss)   └───>│ VAE Decoder │ (frozen)
  │                                └──────┬──────┘
  │                                       v
  │                              RGB output (3, 512, 512)
  └── (not used at inference)

  Text conditioning (T5 + CLIP):
  ┌──────────────┐   ┌──────────────┐
  │  CLIP (77 tok)│   │ T5-XXL (512) │   Both frozen, encode EMPTY prompt ""
  └──────┬───────┘   └──────┬───────┘   → unconditional text embeddings
         └───────┬──────────┘            → fed to DiT cross-attention
                 v                       → carry NO information (empty string)
         prompt_emb + pooled_emb
```

**Key components**:
- **FLUX.1-dev**: 12B parameter rectified-flow transformer (flow matching, NOT standard diffusion)
- **ControlNet Union Alpha**: Pre-trained multi-conditioning ControlNet, adapted via LoRA
- **LoRA**: Rank 32, on ControlNet only (NOT on DiT) — 80 LoRA layers, ~24M trainable params
- **VAE Encoder** (frozen): Encodes BOTH GT RGB and SPAD binary → 16-channel latents. SPAD does go through VAE — it produces OOD latents from binary input, but ControlNet learns to interpret them
- **VAE Decoder** (frozen): Converts denoised latents → RGB pixels (inference only)
- **T5-XXL + CLIP** (frozen): Encode empty prompt "" → unconditional text conditioning. Required by architecture but carry no task-relevant information
- **Sampling**: 28-step Euler ODE solver
- **All frozen models stay frozen** — `freeze_except()` in `base_pipeline.py:191` freezes everything except explicitly listed trainable models

### 3.2 Key Design Choice: LoRA-on-ControlNet

LoRA adapters are placed on the ControlNet module, not the DiT backbone.
- **Rationale**: ControlNet processes the SPAD conditioning signal; adapting it makes the model better at *interpreting* SPAD inputs. LoRA-on-DiT changes the generative prior itself, which is already strong.
- **Evidence**: LoRA-on-DiT was tested and performed worse
- **Confirmed by probing**: LoRA improves depth encoding by +0.308 R^2 in the DiT, confirming it teaches SPAD-to-geometry conversion through ControlNet

### 3.3 Alternative: FLUX img2img (No ControlNet) — Ablation Complete

FLUX natively supports image-to-image generation via `input_image` + `denoising_strength`:

```
SPAD binary frame (1-bit, 512x512)
    |
    v
[VAE Encoder] --> input_latents (16, 64, 64)
    |
    v (add noise at denoising_strength level)
[FLUX.1-dev DiT + LoRA] (12B params, DiT LoRA rank-32 trainable)
    |
    v
[VAE Decoder] --> RGB image (3, 512, 512)
```

- **No ControlNet** — simpler architecture, lower VRAM
- LoRA on DiT directly (rank-32), same rank as ControlNet approach
- Trained 20 epochs (script: `train_img2img_ablation.sh`)
- Swept denoising_strength {0.3, 0.5, 0.7, 0.8, 0.9, 1.0}
- **Status**: Training complete, sweep in progress. See Section 6.6.
- **Result**: Catastrophic failure — PSNR ~7.5 dB, LPIPS >1.0. VAE cannot encode 1-bit binary data meaningfully. **Strongly justifies ControlNet.**

### 3.4 Role of the VAE in Our Pipeline

**CORRECTION**: The SPAD binary frame **does** go through the frozen VAE encoder. See `FluxImageUnit_ControlNet.process()` at `flux_image.py:473-484` — ControlNet inputs are VAE-encoded before being fed to ControlNet blocks.

The FLUX VAE (AutoencoderKL, 16-channel latent) is **completely frozen** throughout training and inference:

| Component | Training | Inference | Processes SPAD? |
|-----------|----------|-----------|-----------------|
| **VAE Encoder** | Encodes GT RGB → latent targets | Encodes SPAD → ControlNet input | **YES** — encodes SPAD to latent |
| **VAE Encoder** | Encodes SPAD → ControlNet input | (same) | **YES** — same encoder, both paths |
| **VAE Decoder** | Not used | Denoised latents → RGB pixels | No — sees DiT output only |

**Data flow (corrected)**:
```
SPAD binary {0,255} → VAE encoder (frozen) → z_spad (16,64,64) → ControlNet → conditioning
GT RGB               → VAE encoder (frozen) → z_gt (16,64,64)   → flow matching loss target
```

**Why does this work despite the VAE domain gap?** The frozen VAE encoder produces OOD latents from binary SPAD input (gQIR confirms this). But the ControlNet+LoRA learns to *interpret* these OOD latents as conditioning signals. The ControlNet doesn't need the SPAD latents to be on-manifold — it just needs a consistent, deterministic mapping from SPAD → some latent representation. The LoRA adaptation on ControlNet learns this mapping.

**Contrast with img2img** (Section 6.6): In img2img, OOD SPAD latents are used as the *denoising starting point* — the DiT must denoise from them directly. This fails catastrophically (PSNR ~7.5) because the DiT expects on-manifold latents. In our ControlNet pipeline, OOD latents are used as *conditioning*, not as the denoising target — a much easier problem.

**Contrast with gQIR**: gQIR fine-tunes the VAE encoder (Stage 1, 600k steps) so SPAD latents ARE on-manifold, because their U-Net denoises from them directly. We avoid this by routing SPAD through ControlNet instead of the denoising path.

**The VAE decoder** only matters at inference — it converts the DiT's denoised latents to RGB pixels. It never sees SPAD data and needs no adaptation.

**T5-XXL + CLIP**: Both text encoders are frozen and encode empty prompts (""). They produce unconditional text embeddings fed to DiT cross-attention. Required by the FLUX architecture but carry zero task-relevant information. All conditioning comes from ControlNet.

#### Future Work: Improving the SPAD Conditioning Pathway

Three possible experiments to improve how SPAD data enters the pipeline. All are future work — the current pipeline already works without any of these. Listed from least to most feasible:

**Option A: gQIR-style Predegradation Removal (VAE encoder fine-tuning)**
- Fine-tune VAE encoder with LSA loss so SPAD latents land on the clean-image manifold
- **Hypothesis**: On-manifold SPAD latents could make ControlNet's job easier, potentially improving reconstruction quality
- **Why probably NOT worth it**: (1) Requires 600k steps on 8×A100 with fragile LSA loss — encoder collapses without it (gQIR Table 4: PSNR 24.78→10.30). (2) We have ~1,850 training images vs gQIR's 2.81M — too few to safely fine-tune the VAE. (3) ControlNet+LoRA already compensates for OOD latents; the 80 LoRA layers are effectively learning this mapping. (4) Risk of breaking the latent space for the decoder.
- **Expected gain**: Marginal. The bottleneck is likely not the OOD latents but the information content of 1-bit binary data itself.

**Option B: Custom SPAD Pre-Encoder (before VAE)**
```
SPAD {0,255} → [Learned CNN/ViT] → pseudo-natural-image (3,512,512) → VAE encoder → ControlNet
```
- A lightweight learned module that maps binary SPAD → something closer to natural image statistics before the VAE sees it
- **Hypothesis**: If the VAE receives input with natural-image-like statistics, it produces on-manifold latents, giving ControlNet cleaner conditioning
- **Pros**: Doesn't touch the VAE weights, trainable end-to-end, small model
- **Cons**: Adds inference latency, another module to train, unclear if the information bottleneck is at the VAE or downstream

**Option C: Custom SPAD Latent Encoder (replacing VAE for conditioning path)**
```
SPAD {0,255} → [Learned Latent Encoder] → z_spad (16,64,64) → ControlNet
```
- Bypass the VAE entirely for the SPAD path. A learned encoder directly produces 16-channel latents sized for ControlNet input.
- **Hypothesis**: A domain-specific encoder trained end-to-end could produce more informative conditioning latents than the frozen VAE's OOD outputs
- **Pros**: Cleanest design, no VAE dependency, can be optimized for SPAD specifically
- **Cons**: Must match the 16-channel latent format ControlNet expects, needs careful initialization to avoid training instability

**Overall assessment**: All three are reasonable future experiments but NONE are necessary for the thesis. The current ControlNet+LoRA pipeline already handles the OOD VAE latents effectively. The probing results confirm the model extracts depth (R²=0.685), bit density (R²=0.998), and cross-frame variance (R²=0.359) from the SPAD conditioning despite the VAE domain gap. The most promising option for future work is Option C (custom latent encoder) — cleanest design, most directly addresses the domain gap, and avoids the fragility of VAE fine-tuning.

### 3.5 ControlNet Choice: Why Union Alpha "grey"?

The available FLUX ControlNets in DiffSynth include:
- `InstantX/FLUX.1-dev-Controlnet-Union-alpha` ✅ (used) — multi-conditioning, "grey" mode closest to SPAD
- `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta` — for masked inpainting, wrong task
- `jasperai/Flux.1-dev-Controlnet-Upscaler` — for super-resolution, wrong task
- `InstantX/FLUX.1-dev-IP-Adapter` — for semantic/style transfer, wrong modality

Union Alpha "grey" is the best available match for binary/grayscale spatial conditioning. LoRA fine-tuning adapts it specifically for the SPAD domain.

### 3.6 Memory Management

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **FP8 quantization**: DiT backbone, text encoders (T5-XXL, CLIP), and ControlNet (when frozen) stored in FP8
- **PYTORCH_CUDA_ALLOC_CONF**: `expandable_segments:True`
- **Pixel-space DPS**: OOM on 32GB (requires backprop through VAE decoder at every step)
- **Latent-space DPS**: Feasible, operates entirely in VAE latent space

### 3.7 SD1.5 Baseline (spad-diffusion repo)

A second repo (`spad-diffusion/`) contains the SD1.5 ControlNet baseline:

```
SPAD binary frame (1-bit, 512x512)
    |
    v
[ControlNet Encoder] (12 input blocks + 1 mid, 361M params)
    |  channels: 320 → 640 → 1280 → 1280
    |  spatial: 64×64 → 32×32 → 16×16 → 8×8
    |  13 zero_conv outputs → UNet skip connections
    v
[SD1.5 UNet] (860M params, fully finetuned)
    |  12 input + 1 mid + 12 output blocks
    |  ResBlock + SpatialTransformer at attn_res=[4,2,1]
    |  CLIP text conditioning (768-dim), DDIM 50 steps
    v
[VAE Decoder] --> RGB image (3, 512, 512)
```

- Two-stage training: 10-frame → 1-frame → OD3 filter
- Best checkpoint: `best-epoch=14-val_loss=0.1057.ckpt` (8.6 GB)
- **Status**: Linear probing complete (see Section 7.6). Baseline re-evaluation on scene-aware split pending.
- **Key architectural differences from FLUX**: Variable spatial resolution (64→8→64 with skip connections) vs uniform 64×64 tokens; 1.2B total params vs 12B; no LoRA (full ControlNet finetuned)

---

## 4. Dataset and Data Pipeline

### 4.1 Dataset Structure

```
spad_dataset/
  RGB/                          # Ground truth RGB images (~2,637 views)
  bits/                         # RAW single-frame binary (frame 0)
  bits_frame_{1000,...,16000}/   # Different temporal frame indices (N=1)
  bits_multi_{4,16,64,256,1000}/ # Accumulated multi-frame (N frames)
  bits_RAW_OD_{01,03,07}/       # OD filter conditions (single frame)
  bits_RAW_OD_{01,03,07}_multi_{16,64}/  # OD + multi-frame
  metadata_train.csv            # 1,850 training views (77 locations)
  metadata_val.csv              # 776 validation views (20 locations)
  dataset_inventory.json        # Full dataset audit results
  split_manifest.json           # Location-level split documentation
```

### 4.2 Critical: Scene-Aware Stratified Split

**Problem discovered**: The original random train-test split had **94 out of 101 physical locations appearing in BOTH train and val** -- severe data leakage inflating all metrics.

**Fix**: Stratified split by physical location (session), balanced by indoor/outdoor:
- **77 train locations** (1,850 views)
- **20 val locations** (776 views) -- 14 indoor + 6 outdoor
- **Zero location leakage**
- Old split backed up at `spad_dataset/old_random_split/`

### 4.3 16-bit SPAD Image Loading Bug

Multi-frame SPAD images are 16-bit PNGs with intermediate values (e.g., 16384). PIL's `convert('RGB')` clamps values >255 to 255, destroying the data.

**Fix**: Custom `load_spad_image()` function in `diffsynth/core/data/operators.py`:
```python
arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
```

---

## 5. Training Details

### 5.1 Training Configurations

| Model | Script | Epochs | LR | LoRA | Base Checkpoint | Status |
|-------|--------|--------|-----|------|----------------|--------|
| RAW baseline | `train_scene_aware_raw.sh` | 40 | 1e-4 | Rank-16 on CN | From scratch | **Complete** (best=epoch-15) |
| Consistency | `train_consistency.sh` | 30 | 5e-5 | Rank-16 on CN | RAW epoch-15 | **Complete** (best=epoch-0) |
| img2img ablation | `train_img2img_ablation.sh` | 20 | 1e-4 | Rank-32 on DiT | From scratch | **Complete** |
| OD03 fine-tune | `train_od03_finetune.sh` | 20 | 5e-5 | Rank-16 on CN | RAW epoch-15 | Partial (4 epochs) |
| OD07 fine-tune | `train_od07_finetune.sh` | 20 | 5e-5 | Rank-16 on CN | RAW epoch-15 | Partial (4 epochs) |
| OD03 scratch | `train_od03_scratch.sh` | 40 | 1e-4 | Rank-16 on CN | From scratch | **Complete** (40 epochs) |
| ControlNet Full (lr=1e-5) | `train_controlnet_full.sh` | 40 | 1e-5 | None (full fine-tune) | From scratch | **Complete** (40 epochs, resume bug) |
| ControlNet Full (lr=1e-4) | — | 40 | 1e-4 | None (full fine-tune) | From scratch | Timed out (27 epochs, diverged) |
| LoRA Rank 8 | — | 40 | 1e-4 | Rank-8 on CN | From scratch | Timed out (31 epochs) |
| LoRA Rank 64 | — | 40 | 1e-4 | Rank-64 on CN | From scratch | Timed out (31 epochs) — **best val loss** |
| LoRA Rank 128 | — | 40 | 1e-4 | Rank-128 on CN | From scratch | Timed out (31 epochs) |
| Dual LoRA (CN+DiT) | — | 40 | 1e-4 | Rank-16 CN + Rank-8 DiT | From scratch | Timed out (32 epochs) |
| SPAD Encoder | — | 40 | 1e-4 | LoRA r32 on CN + SPADEncoder | From scratch | Timed out (32 epochs) |

### 5.2 Hyperparameters

- LoRA rank: 16 (on ControlNet)
- MAX_PIXELS: 262,144 (512x512)
- Gradient accumulation: 4
- Optimizer: AdamW
- FP8 models: DiT, T5-XXL, CLIP, ControlNet (frozen components)
- Conda env: `diffsynth`

---

## 6. Experimental Results

### 6.1 Baseline (Single-Frame Binary SPAD -> RGB)

| Metric | Seed 42 | 10-Seed Mean +/- Std |
|--------|---------|---------------------|
| PSNR (dB) | 17.89 | 17.99 +/- 0.09 |
| SSIM | 0.596 | 0.596 +/- 0.001 |
| LPIPS | 0.415 | 0.415 +/- 0.001 |
| FID | 66.84 | 66.29 +/- 0.74 |
| CFID | 151.94 | 152.04 +/- 1.16 |

- All metrics computed in **grayscale** (SPAD is monochrome; avoids penalizing color hallucination)
- CFID uses Ridge regularization + float64 for numerical stability (N=776 << D=2048)
- Low multi-seed variance indicates model is stable

### 6.2 Physics Ablation (Latent-Space DPS)

| Config | PSNR | SSIM | LPIPS | FID | CFID |
|--------|------|------|-------|-----|------|
| Baseline | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| eta=0.01 | 18.02 | 0.596 | 0.413 | 65.81 | 151.66 |
| eta=0.05 | 18.02 | 0.596 | 0.413 | 65.76 | 151.39 |
| **eta=1.0** | **18.05** | **0.597** | **0.413** | **65.97** | **151.35** |

- Modest improvement: +0.16 PSNR, -0.87 FID, -0.59 CFID
- Improvement monotonic with eta (no instability even at eta=1.0)
- **Interpretation**: ControlNet already captures measurement information so effectively that explicit physics guidance provides marginal additional benefit

### 6.3 Frame-Count Ablation (Zero-Shot Multi-Frame)

| Frames | PSNR | SSIM | LPIPS | FID | CFID |
|--------|------|------|-------|-----|------|
| 1 | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| 4 | 17.82 | 0.636 | 0.376 | 71.44 | 138.05 |
| 16 | 16.91 | 0.651 | 0.359 | 74.85 | 131.04 |
| 64 | 15.47 | 0.640 | 0.346 | 74.50 | 120.90 |
| 256 | 14.12 | 0.605 | 0.339 | 70.66 | 110.11 |
| 1000 | 13.04 | 0.551 | 0.347 | 68.52 | 108.11 |

- **CFID monotonically improves** with more frames (151.94 -> 108.11): model produces more measurement-consistent outputs
- **LPIPS improves** until ~256 frames (0.415 -> 0.339): perceptual quality increases
- **PSNR degrades** for >4 frames: model was trained on single-frame only (zero-shot transfer), domain gap increases with frame count
- Multi-frame inputs make the conditioning less binary and more "image-like", reducing hallucination but creating distribution shift

### 6.4 Consistency Training

| Config | PSNR | SSIM | LPIPS | FID | CFID |
|--------|------|------|-------|-----|------|
| Baseline | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| Consistency epoch-0 | 17.72 | 0.589 | 0.422 | 66.51 | 154.99 |
| Consistency + DPS | 17.86 | 0.590 | 0.420 | 65.75 | 154.15 |

- Consistency training **hurt performance** -- epoch-0 (barely fine-tuned) was best
- lambda=0.1 may have been too strong; stop-gradient on F2 makes loss asymmetric
- **This is a valid negative result** for the paper narrative
- **Epoch sweep** (5, 10, 15, 20, 25, 29): In progress via `run_all_experiments.sh` pipeline

### 6.5 Analysis Results

- **Variance**: Mean sigma^2 = 0.0061 across 776 images; bit-density correlation = -0.226 (darker regions have *more* uncertainty, as expected)
- **Calibration**: ECE = 0.269 (model is underconfident -- empirical coverage < nominal)
- **Intermediate latents**: Saved for 20 samples x 8 denoising steps; structure locks in early (~step 10/28)

### 6.6 img2img Ablation (No ControlNet)

Tests whether ControlNet is necessary by using FLUX's native img2img pathway with LoRA-on-DiT. Full formal audit in `agent/audits/AUDIT_IMG2IMG_ABLATION.md`.

**Training**: Rank-32 LoRA on FLUX DiT backbone, 20 epochs, standard flow matching on RGB targets only. **SPAD images are never seen during training** — the LoRA learns only the RGB target distribution. No ControlNet loaded.

**Inference**: SPAD binary frame is VAE-encoded → blended with noise at `sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength` → denoised by DiT+LoRA over 28 Euler steps. The SPAD signal contributes `(1 - sigma_start)` fraction of the initial latent (e.g., 30% at strength=0.7).

| Strength | PSNR | SSIM | LPIPS | FID | CFID | Output→SPAD r |
|----------|------|------|-------|-----|------|---------------|
| **ControlNet (ours)** | **17.89** | **0.596** | **0.415** | **66.84** | **151.94** | — |
| 0.3 | 7.59 | 0.013 | 1.093 | 388.42 | 351.22 | 0.994 |
| 0.5 | 7.50 | 0.013 | 1.103 | 356.75 | 365.46 | 0.983 |
| 0.7 | 7.44 | 0.016 | 1.088 | 322.68 | 391.50 | 0.931 |
| 0.8 | 7.52 | 0.026 | 1.055 | 283.86 | 396.38 | ~0.8 |
| 0.9 | 8.26 | 0.118 | 0.928 | 182.24 | 355.44 | ~0.56 |
| 1.0 | 10.28 | 0.392 | 0.692 | 107.85 | 308.51 | 0 (no input) |

**Root causes of failure** (two compounding problems):
1. **VAE domain mismatch**: FLUX VAE trained on natural images (continuous-tone, mean ~128). SPAD binary frames are {0, 255} with ~5-10% white pixels (mean ~10-28). VAE encodes them into OOD latents representing "very dark images", not scene structure. Output brightness tracks SPAD brightness (mean ~25) rather than GT brightness (mean ~106).
2. **No SPAD→RGB mapping learned**: Training never sees SPAD images (`--data_file_keys "image"` loads only RGB GT). LoRA learns the RGB distribution but has zero mechanism to map SPAD→RGB.

**Why ControlNet succeeds**: (1) ControlNet+LoRA learns to interpret OOD VAE latents as conditioning (not denoising start point). (2) Paired training with `--data_file_keys "image,controlnet_image"` and `--extra_inputs "controlnet_image"` provides explicit SPAD→RGB supervision.

**Note**: strength=1.0 (PSNR 10.28) is best because it's pure noise + LoRA generation — no SPAD signal at all, just the learned RGB prior. This is higher than the low-strength results where OOD SPAD latents actively corrupt the output.

**Known bugs**: Checkpoint sort bug selected epoch-9 instead of epoch-19 (does not affect conclusion).

### 6.7 Best-of-K NLL Reranking

Select the best reconstruction from K=10 seeds per image, ranked by Bernoulli measurement NLL:

| Config | PSNR | SSIM | LPIPS | FID | CFID | Mean NLL |
|--------|------|------|-------|-----|------|----------|
| Baseline (seed 42) | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 | 0.747 |
| Best-of-10 (NLL) | 15.30 | 0.585 | 0.433 | 66.46 | 156.96 | 0.721 |

- NLL-best samples have **lower measurement consistency** (NLL: 0.747 → 0.721) but **worse perceptual quality** (PSNR: 17.89 → 15.30)
- This reveals the **perception-consistency tradeoff**: samples that better match the measurement are not necessarily perceptually better
- The model achieves perceptual quality partly through *controlled hallucination* that violates strict measurement consistency

### 6.8 Cross-Frame Generation

Generated 7 independent reconstructions per image (different SPAD binary frames, same scene) to quantify frame-dependent variation:

| Config | PSNR | SSIM | LPIPS | FID | CFID | Meas. NLL |
|--------|------|------|-------|-----|------|-----------|
| Baseline (frame 0) | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 | 0.747 |
| FlowDPS (eta=0.1) | 18.18 | 0.596 | 0.410 | 65.62 | 150.53 | 0.755 |

- Cross-frame variance maps computed from 7 frames → used as probing target (Section 7.5)
- FlowDPS with detection-based guidance (eta=0.1, mid-4 steps 14+) provides modest improvement

### 6.9 No-LoRA Ablation (Frozen Pretrained ControlNet)

Tests the pretrained ControlNet Union Alpha without any LoRA adaptation. Frozen DiT + frozen ControlNet + no LoRA. Sweeps all relevant ControlNet conditioning modes to find the best off-the-shelf configuration.

| Mode | PSNR | SSIM | LPIPS | FID | CFID |
|------|------|------|-------|-----|------|
| **ControlNet + LoRA (ours)** | **17.89** | **0.596** | **0.415** | **66.84** | **151.94** |
| gray (our training mode) | 8.57 | 0.046 | 1.088 | 313.88 | 388.94 |
| lq (low quality) | 8.74 | 0.112 | 0.848 | 317.66 | 373.48 |
| canny | **10.81** | 0.189 | 0.797 | 274.57 | 363.70 |
| tile | 9.68 | **0.188** | **0.751** | 284.79 | **352.09** |
| depth | 10.13 | 0.171 | 0.784 | 288.28 | 373.83 |

**Key findings**:
1. **LoRA provides +7.08 dB** over best no-LoRA config (canny 10.81 → 17.89). LoRA adaptation is essential.
2. **"gray" mode is worst** (8.57 dB) — the pretrained gray mode expects natural grayscale, not binary SPAD. Ironic since this is our training mode; LoRA completely transforms how the ControlNet interprets this mode.
3. **"canny" is best no-LoRA** (10.81 dB) — SPAD binary frames resemble edge maps (sparse white pixels on black), which is closest to canny's expected input distribution.
4. **All modes fail** — even the best no-LoRA is far below baseline, confirming that LoRA fine-tuning teaches the SPAD→RGB mapping, not just the mode selection.

### 6.10 Ablation Summary

| Configuration | PSNR | Delta | What It Tests |
|--------------|------|-------|---------------|
| **Baseline (CN + LoRA)** | **17.89** | — | Full pipeline |
| FlowDPS (eta=0.1) | 18.18 | +0.29 | Physics guidance |
| No-LoRA (best: canny) | 10.81 | **-7.08** | LoRA necessity |
| img2img (best: s=1.0) | 10.28 | **-7.61** | ControlNet pathway |
| No-LoRA gray | 8.57 | **-9.32** | Pretrained CN on SPAD |
| img2img (worst: s=0.7) | 7.44 | **-10.45** | VAE domain gap |

### 6.11 Killarney H100 Training Ablations (March 2026)

Ran on Alliance Canada Killarney cluster (H100 80GB). All experiments: 512x512, FP8 frozen DiT+T5, gradient checkpointing, AdamW, flow matching loss.

#### Summary Table

| Experiment | Best Val Loss | Best Epoch | Final Epoch | Status | Notes |
|------------|-------------|------------|-------------|--------|-------|
| **LoRA Rank 64** | **0.3085** | 23 | 31/40 | Timed out | Best overall |
| Dual LoRA (CN r16 + DiT r8) | 0.3103 | 32 | 32/40 | Timed out | Still improving |
| LoRA Rank 128 | 0.3111 | 10 | 31/40 | Timed out | Converged fast |
| SPAD Encoder | 0.3192 | 26 | 32/40 | Timed out | Custom encoder |
| LoRA Rank 8 | 0.3278 | 21 | 31/40 | Timed out | Underfitting |
| ControlNet Full (lr=1e-5) | 0.3347 | 17 | 40/40 | Complete | **Checkpoint resume bug** |
| ControlNet Full (lr=1e-4) | 1.0503 | 27 | 27/40 | Timed out | Diverged after epoch 1 |
| **Baseline (LoRA r16, Section 6.1)** | **~0.37** | -- | -- | Reference | From original training |

#### ControlNet Full (epoch-39) Image Quality Metrics

| Metric | ControlNet Full (epoch-39) | Baseline LoRA r16 (Section 6.1) |
|--------|---------------------------|--------------------------------|
| PSNR (dB) | **18.20** | 17.89 |
| SSIM | 0.592 | 0.596 |
| LPIPS | **0.405** | 0.415 |

- ControlNet Full achieves +0.31 dB PSNR and -0.010 LPIPS improvement over LoRA r16 baseline
- SSIM slightly worse (0.592 vs 0.596)
- **Note**: These metrics may undercount the potential — checkpoint resume was broken (see below), so this is really ~22 epochs of training, not 40

#### Critical Bug: Checkpoint Resume Failure

The ControlNet Full training used `--remove_prefix_in_ckpt "pipe.controlnet.models.0."` which saves keys like `blocks.0.attn.a_to_qkv.weight`. However, the resume code loaded these into `pipe.controlnet` (the ControlNetUnit wrapper), which expects keys prefixed with `models.0.`. **Zero keys matched**, meaning each chained SLURM job restarted from the base ControlNet. The epoch-39 checkpoint is really only ~22 epochs of single-job training (epochs 18-39 from job 2774305).

**Fixed in**: `train_lora.py` (lines 371-387) and `validate_controlnet_full.py` (lines 94-113). Fix navigates to `trainable_model.models[0]` for wrapper models.

#### Detailed Epoch Losses

**LoRA Rank 64** (best experiment):
| Epoch | Train | Val | | Epoch | Train | Val |
|-------|-------|-----|-|-------|-------|-----|
| 1 | 0.3447 | 0.3464 | | 17 | 0.3031 | 0.3254 |
| 4 | 0.3391 | 0.3170 | | 20 | 0.3201 | 0.3262 |
| 8 | 0.3204 | 0.3148 | | 23 | 0.3101 | **0.3085** |
| 11 | 0.3289 | 0.3210 | | 27 | 0.3225 | 0.3182 |
| 15 | 0.3249 | 0.3279 | | 31 | 0.3169 | 0.3245 |

**LoRA Rank 128**:
Best val 0.3111 at epoch 10. Converged faster than rank 64 but plateaued at slightly worse loss.

**Dual LoRA (ControlNet r16 + DiT r8)**:
Best val 0.3103 at epoch 32. Was still improving when timed out — most promising for further training.

**SPAD Encoder**:
Best val 0.3192 at epoch 26. SPADEncoder (0.1M params) provides learned preprocessing. One variant (job 2774298) failed with shape mismatch (`1024x256 vs 64x3072` in `controlnet_x_embedder`).

**ControlNet Full lr=1e-4**:
Diverged catastrophically after epoch 1 (val loss jumped from 0.37 to 1.22). Loss was slowly recovering by epoch 27 (1.05) but still far from usable. Confirms lr=1e-5 is correct for full fine-tuning.

#### Key Takeaways

1. **LoRA rank 64 is optimal**: Best val loss (0.3085), significantly below LoRA r16 baseline (~0.37) and ControlNet Full (0.3347)
2. **More parameters ≠ better**: ControlNet Full fine-tuning (all 249 params) is worse than LoRA r64 (~26M params), likely due to overfitting risk and the checkpoint bug
3. **Dual LoRA (CN + DiT) is promising**: Still improving at epoch 32, may surpass LoRA r64 with more training
4. **lr sensitivity**: Full fine-tuning at lr=1e-4 diverges; LoRA at lr=1e-4 is fine
5. **All experiments incomplete**: Jobs timed out at 24h on Killarney H100s (reached epoch 31-32 of 40). ControlNet Full is the only complete run (40 epochs)

#### Cluster Details

- **Cluster**: Alliance Canada Killarney, H100 80GB (10 nodes x 8 GPUs)
- **Account**: aip-lindell (low fairshare priority, NormShares=0.004780)
- **Walltime**: 24h jobs (b3 partition) for experiments, 6h jobs for ControlNet Full (chained via `--dependency=afterok`)
- **L40S tested**: OOM at optimizer.step() — 48GB insufficient for full fine-tuning
- **Training speed**: ~1.4 it/s on H100, ~27 min/epoch (1850 steps), ~18h for 40 epochs

---

## 7. Linear Probing Analysis (Key Contribution)

**This is the strongest, most novel contribution. Make it the centerpiece of both presentation and paper.**

### 7.1 Setup

Four experimental conditions:
1. **Main (FLUX)**: FLUX + ControlNet + LoRA (full model, 12B DiT + 1.4B CN)
2. **Control (FLUX)**: FLUX + ControlNet, no LoRA (isolates LoRA effect)
3. **No-ControlNet (FLUX)**: FLUX DiT only (sanity check -- should have zero SPAD information)
4. **SD1.5**: SD1.5 UNet + ControlNet (cross-architecture comparison, 860M UNet + 361M CN)

Probing methodology:
- **FLUX**: Forward hooks on 57 DiT blocks (19 joint + 38 single) + 15 ControlNet blocks; 7 timesteps {0, 4, 9, 14, 19, 24, 27}; uniform 3072-dim features
- **SD1.5**: Forward hooks on 25 UNet blocks (12 input + 1 mid + 12 output) + 13 ControlNet blocks; 7 timesteps {0, 7, 14, 21, 28, 35, 49} (out of 50 DDIM steps); variable feature dims (320/640/1280)
- Ridge regression (closed-form) with trace-normalized regularization: λ_s = λ × tr(X^TX)/D
- 80/20 train/test split of 776 validation images
- **Global probing**: Mean-pool spatial features → 1 vector per block-timestep, predict scalar target
- **Spatial probing**: Keep per-token features (pooled to 32×32), predict 32×32 target map. Uses streaming X^TX/X^Ty sufficient-statistic accumulation in float64 to avoid saving huge per-token activation files

### 7.2 Headline Results

#### Continuous Targets — Global Probing (Best R^2)

| Target | Main (LoRA) | Control (no LoRA) | No-ControlNet | LoRA Delta |
|--------|-------------|-------------------|---------------|------------|
| Bit density | 0.998 | 0.999 | -0.060 | ~0 (inherent) |
| Depth | 0.437 | 0.129 | -0.040 | **+0.308** |
| Variance (seed) | 0.424 | 0.446 | -0.360 | -0.022 |
| **Cross-frame variance** | **0.292** | 0.222 | -0.161 | **+0.070** |

#### Continuous Targets — Spatial Probing (Best R^2)

| Target | Main (LoRA) | Control (no LoRA) | No-ControlNet | LoRA Delta |
|--------|-------------|-------------------|---------------|------------|
| Bit density (spatial) | 0.959 | 0.992 | -0.046 | ~0 (inherent) |
| **Depth (spatial)** | **0.685** | 0.453 | 0.406* | **+0.232** |
| **Variance (spatial)** | **0.506** | 0.434 | -0.067 | **+0.072** |
| **Cross-frame var (spatial)** | **0.359** | — | — | — (re-extraction complete) |

*No-CN depth at S0 t=0 is an artifact (VAE encoding spatial structure).

#### Object Presence (Best Balanced Accuracy, 24 classes)

| Object | Main | Control | Delta |
|--------|------|---------|-------|
| Table | 0.982 | 0.912 | +0.070 |
| Chair | 0.953 | 0.881 | +0.072 |
| Mouse | 0.861 | 0.677 | **+0.184** (largest) |
| Keyboard | 0.881 | 0.724 | **+0.157** |
| Building | 0.906 | 0.818 | +0.089 |

LoRA improves recognition for **20 of 24 objects**.

### 7.3 Key Findings

1. **Bit density is perfectly preserved**: R^2 >= 0.998 throughout all 57 DiT blocks. ControlNet successfully injects SPAD measurements and the DiT preserves them. Without ControlNet: R^2 = 0 everywhere.

2. **Depth emerges from 1-bit data**: Spatial depth R^2 = 0.685 from a single block's activations -- comparable to AC3D's depth probing of RGB-conditioned video DiT (R^2 ~ 0.65). The model converts extremely sparse photon statistics into a rich 3D scene representation.

3. **LoRA teaches geometry**: Depth R^2 jumps from 0.453 (control) to 0.685 (main) -- a +0.232 improvement. LoRA creates depth readability where none existed (235/399 positive R^2 blocks vs 22/399 for control).

4. **The model knows where it guesses**: Variance is linearly decodable (R^2 = 0.506 spatial), meaning the model internally represents its own uncertainty. This is present even without LoRA (0.434), suggesting it's an inherent property of large diffusion models.

5. **Semantic understanding from 1-bit data**: The model recognizes specific objects (tables, chairs, keyboards) from binary SPAD frames with balanced accuracy up to 0.982. LoRA specifically improves small/distinctive objects.

6. **Information flow through denoising**:
   - t=0: Conditioning injection (SPAD measurements absorbed, bit density peaks)
   - t=4-9: Scene geometry formation (depth peaks)
   - t=14+: Commitment to details (variance/uncertainty crystallizes)

7. **No-ControlNet ablation**: 0/399 block-timestep combinations have positive R^2 for ANY target. This confirms all SPAD information enters through the ControlNet.

### 7.4 Comparison with AC3D

| Aspect | AC3D (Bhatt et al., 2024) | Ours |
|--------|--------------------------|------|
| Architecture | U-Net (SVD) | DiT (FLUX.1-dev) |
| Conditioning | RGB video frames | 1-bit binary SPAD sensor |
| Depth R^2 | ~0.65 (best) | 0.685 (spatial, best) |
| Key finding | Mid-layers encode 3D | Early single blocks encode 3D |

We achieve comparable depth probing performance despite our input being 1-bit binary vs RGB.

### 7.5 Cross-Frame Variance Probing (NEW)

Cross-frame variance measures how much the reconstruction changes when a *different* SPAD binary frame (same scene) is used. This is distinct from seed variance (same frame, different seeds).

**How target was computed**: Generated 7 reconstructions per image from 7 different SPAD frames → per-image cross-frame variance = mean pixel-wise variance across 7 outputs.

#### Global Cross-Frame Variance (Best R^2)

| Condition | Best R^2 | Best Block | Best Timestep |
|-----------|---------|-----------|---------------|
| Main (LoRA) | 0.292 | joint_1 | t=27 |
| Control (no LoRA) | 0.222 | joint_4 | t=27 |
| No-ControlNet | -0.161 | — | — |

**Key findings**:
- Cross-frame variance is encoded in **early joint blocks** (joint_1-4) at **late timesteps** (t=24-27), unlike seed variance which peaks at t=14
- LoRA amplifies cross-frame sensitivity by +32% (0.292 vs 0.222), meaning LoRA makes the model MORE responsive to the specific SPAD frame
- Without ControlNet: all negative R^2 → all frame-dependent info enters through ControlNet
- **Different temporal profile**: Seed variance peaks mid-denoising (t=14), cross-frame variance peaks near completion (t=27). Frame-dependent variation is *retained through to the final output*.

#### Spatial Cross-Frame Variance

| Condition | Best R² | Best Block | Best Timestep |
|-----------|---------|-----------|---------------|
| Main (LoRA) — spatial | **0.359** | single_28 | t=9 |
| Main (LoRA) — CN blocks | **0.360** | cn_single_1 | t=24 |

**Spatial probing exceeds global by +23%** (0.359 vs 0.292), indicating that cross-frame variance has strong spatial structure — physically meaningful since low-photon-count image regions have higher Bernoulli variance and are more frame-sensitive.

**ControlNet blocks are most frame-sensitive**: CN best R²=0.360 matches spatial DiT R²=0.359. Frame information is strongest at the conditioning injection point.

**Spatial peak is deeper in the network**: single_28 @ t=9 vs joint_1 @ t=27 for global. Spatially-resolved frame information propagates deeper than the global signal.

### 7.6 SD1.5 Cross-Architecture Comparison (NEW — Complete)

Full linear probing pipeline replicated for SD1.5 UNet (860M) + ControlNet (361M) to answer: **Is the internal representation of SPAD data architecture-dependent?**

#### Best R^2 Comparison: SD1.5 vs FLUX

| Target | SD1.5 Global | FLUX Global | SD1.5 Spatial | FLUX Spatial |
|--------|-------------|-------------|---------------|-------------|
| Bit density | 0.993 | 0.998 | 0.974 | 0.959 |
| Depth | 0.375 | 0.437 | **0.727** | 0.685 |
| Variance (seed) | **0.472** | 0.424 | 0.493 | 0.506 |
| Cross-frame var | 0.293 | 0.292 | 0.279 | **0.359** |

#### SD1.5 Top Blocks Per Target

| Target | Best Block | Notes |
|--------|-----------|-------|
| Bit density | cn_input_1@t49 | ControlNet block 1, identical to FLUX's joint_1 |
| Depth | unet_input_8@t49 | UNet bottleneck (1280-ch, 8×8), mid-depth encoder |
| Variance | cn_input_1@t21 | Early ControlNet, mid-denoising |
| Cross-frame var | cn_input_1@t0 | Early ControlNet, pure noise step |
| Objects (building) | cn_input_8@t0 | R^2=0.709 |
| Objects (table) | unet_output_4@t0 | R^2=0.703 |

#### Key Cross-Architecture Insights

1. **SD1.5 is surprisingly competitive despite 14× fewer parameters** (1.2B vs 12B):
   - Bit density: near-identical (0.993 vs 0.998)
   - Cross-frame variance: near-identical (0.293 vs 0.292)
   - Variance: SD1.5 slightly **better** (0.472 vs 0.424)

2. **UNet skip connections give a spatial advantage**: SD1.5 spatial depth R^2=0.727 vs FLUX's 0.685. The hierarchical resolution structure (64→32→16→8→16→32→64) preserves per-pixel depth better than uniform 64×64 tokens.

3. **Information is localized by function in SD1.5**: ControlNet early blocks handle bit density/variance; UNet bottleneck handles depth; UNet decoder handles objects. In FLUX, joint_1 and single_9 are "generalist" blocks.

4. **Different temporal dynamics**: SD1.5's cross-frame variance peaks at t=0 (immediate encoding), while FLUX's peaks at t=27 (progressive development).

5. **Scale helps semantics, not measurement**: The 14× parameter gap primarily affects depth and object recognition. Measurement preservation (bit density) is equally good in both.

**Paper narrative**: "ControlNet-based SPAD encoding is architecture-general — both UNet and DiT learn to preserve measurement properties in early ControlNet blocks. However, model capacity determines the richness of emergent semantic representations (depth, objects)."

**Full results**: Appendix F of `agent/reports/probing_report_final.md`

### 7.7 Practical Implications

- **For LoRA placement**: Depth signal peaks in single_1-9 (FLUX) and unet_input_8-9 (SD1.5). These blocks have the most leverage for LoRA.
- **For DPS guidance timing**: Physics corrections should target t=4-14 (geometry formation phase in FLUX), t=28-49 (late steps in SD1.5).
- **For uncertainty quantification**: Variance is linearly decodable in both architectures -- could enable single-sample uncertainty estimation without expensive multi-seed sampling.
- **For active sensing**: The model knows where it's uncertain, enabling adaptive SPAD frame allocation.
- **For consistency training**: Cross-frame variance signal is concentrated in early ControlNet blocks in both architectures — target these blocks for consistency regularization.

---

## 8. Physics-Guided DPS and Consistency Training

### 8.1 SPAD Forward Model

```
Intensity I (linear, after sRGB->linear conversion)
    |
    v
Exposure H = softplus(alpha * I + beta),  H >= H_min = 1e-6
    |
    v
Detection probability p = 1 - exp(-H)
    |
    v
Binary observation b ~ Bernoulli(p)
```

**NLL** (numerically stable):
```
NLL = (1 - b) * H  -  b * log(1 - exp(-H))
                            ^^ computed via log(-expm1(-H))
```

Implementation: `diffsynth/diffusion/spad_forward.py`

### 8.2 Latent-Space DPS (Evaluated)

At each denoising step:
1. Estimate clean latent: x_hat_0 = x_t - sigma * v_theta
2. Compute L2 gradient: grad = 2 * (x_hat_0 - z_spad)
3. Normalize (PaDIS-style): grad = grad / ||grad||
4. Add correction to velocity: v' = v + eta * grad
5. Scheduler step: x_{t+1} = x_t + v' * (sigma_{t+1} - sigma_t)

**Key insight on sign**: Since (sigma_{t+1} - sigma_t) < 0 during denoising, adding +grad to velocity moves latents in -grad direction, decreasing the loss.

Implementation: `diffsynth/diffusion/latent_dps.py`, `validate_dps.py`

### 8.3 Pixel-Space DPS (Implemented but OOM)

Full Bernoulli likelihood computed after VAE decoding at every step. Theoretically correct but exceeds 32GB VRAM.

Implementation: `diffsynth/diffusion/flow_dps.py`

### 8.4 Per-Frame Consistency Training

Two different SPAD binary frames (F1, F2) of the same scene should produce identical reconstructions:
```
L = L_flow_match(F1) + lambda * ||v_theta(z_t, t, F1) - v_theta(z_t, t, F2)||^2
```
- Stop-gradient on F2 (VRAM constraint)
- F1/F2 randomly assigned each step (both directions covered over training)
- lambda=0.1 was too strong; epoch-0 was best

Implementation: `diffsynth/diffusion/consistency_loss.py`, `train_consistency.py`, `paired_spad_dataset.py`

### 8.5 Code Audit Findings (Fixed)

Three external audits (Cursor/Claude Opus, Gemini, Codex) found and fixed:
- Missing `softplus` and `beta` in exposure mapping
- Missing `log1mexp` numerical stability
- Missing sRGB-to-linear conversion
- DPS correction sign issues
- Missing PaDIS normalization

All fixes committed. See `agent/AUDIT_DPS_CONSISTENCY_2026-03-23.md`.

---

## 9. Key Insights and Narrative

### 9.1 Paper Framing (Strongest Story)

> "We study generative reconstruction from single-photon measurements as a testbed for understanding how large diffusion priors interact with extreme inverse problems. Using a 12B-parameter rectified-flow transformer conditioned on SPAD binary frames, we: (1) probe the model's internal representations to show it implicitly encodes scene geometry despite never receiving depth supervision, (2) characterize the uncertainty-hallucination tradeoff in the single-photon regime where the measurement provides minimal constraint, and (3) introduce physics-consistent posterior sampling that enforces the Bernoulli photon-detection likelihood at inference time."

### 9.2 Why DPS Results Are Modest (And That's OK)

The ControlNet's learned conditioning already captures measurement information so effectively that explicit physics guidance provides only marginal benefit. **The probing analysis explains why**: bit density R^2 = 0.998 means the model already near-perfectly preserves SPAD measurements internally.

Framing: "We find that the physics is already internalized through training. The linear probing analysis provides the first direct evidence for this -- the model encodes not just the raw measurement but also derived 3D structure."

### 9.3 Why Consistency Training Didn't Help (And That's OK)

This is a valid negative result:
- lambda=0.1 was too strong (over-regularized)
- Stop-gradient makes the loss asymmetric
- The model's multi-seed variance is already low (sigma^2 = 0.0061), suggesting it's already quite consistent
- **Could revisit** with lambda=0.01 or 0.001

### 9.4 The Frame Ablation Story

The CFID improvement (152 -> 108) with more frames, despite PSNR degradation, reveals a perception-distortion-consistency tradeoff:
- More frames = more measurement information = more consistent outputs (lower CFID)
- But model was trained on single-frame, so domain gap causes PSNR drop
- LPIPS improvement (0.415 -> 0.339) shows perceptual quality does increase
- **Key figure**: Plot PSNR vs CFID vs LPIPS as a function of frame count

### 9.5 Uncertainty Insights

- Variance correlates with bit density (r = -0.226): darker regions (fewer photons) produce more uncertain reconstructions
- ECE = 0.269: model is **underconfident** (empirical coverage < nominal)
- Variance is linearly decodable from activations (R^2 = 0.506): the model *knows where it's guessing*
- This enables potential single-sample uncertainty estimation

---

## 10. Figures Inventory

### 10.1 Existing Figures (Ready for Paper/Presentation)

#### Probing Analysis Figures (`probing_analysis_output/`)

| File | Description | Paper Use |
|------|-------------|-----------|
| `fig1_main_heatmap.{png,pdf}` | 3-panel R^2 heatmap (57 blocks x 7 timesteps) for bit density, depth, variance | Main probing figure |
| `fig2_main_vs_control.{png,pdf}` | 2x3 comparison: Main vs Control for all targets | LoRA effect |
| `fig3_delta_heatmap.{png,pdf}` | R^2(Main) - R^2(Control) diverging colormap. Red=LoRA helps | LoRA delta figure |
| `fig4_best_timestep_lineplot.{png,pdf}` | R^2 vs block index (AC3D-style information flow) | Core result |
| `fig5_dit_vs_cn.{png,pdf}` | Best R^2 from DiT vs ControlNet blocks | Architecture analysis |
| `fig6_global_vs_spatial.{png,pdf}` | Global vs spatial probing comparison | Methodology |
| `fig7_object_probing.{png,pdf}` | Balanced accuracy for 24 objects, Main vs Control | Semantic understanding |

#### Per-Target Figures — FLUX (`probing_results_allblocks/probes/`)

| File | Description |
|------|-------------|
| `ac3d_curve_{target}.png` | AC3D-style: best R^2 per block with delta bars |
| `allblocks_heatmap_{target}.png` | Full 57-block heatmap per target |
| `comparison_best_timestep.png` | Side-by-side best-timestep comparison |
| `delta_heatmap_{target}.png` | Per-target delta heatmap |
| `heatmap_{target}.png` | DiT+CN combined heatmap |
| `heatmap_cn_{target}.png` | ControlNet-only heatmaps |
| `heatmap_cn_obj_{object}.png` | Per-object ControlNet heatmaps |
| `temporal_{target}.png` | R^2 vs denoising step |
| `heatmap_crossframe_variance.png` | Cross-frame variance heatmap (global) |
| `heatmap_spatial_{target}.png` | Spatial streaming heatmaps (bit_density, depth, variance) |

#### Per-Target Figures — SD1.5 (`spad-diffusion/probing_results_sd15/probes/`)

| File | Description |
|------|-------------|
| `heatmap_{target}.png` | 38-block heatmaps: bit_density, depth, variance, crossframe_variance |
| `heatmap_spatial_{target}.png` | Spatial streaming heatmaps (4 targets) |
| `heatmap_obj_{object}.png` | 24 object presence heatmaps |

### 10.2 Existing Data for Figures (Not Yet Rendered)

| Data Source | Potential Figure |
|-------------|-----------------|
| `validation_outputs_scene_aware/seed_42/` | Input/Output/GT comparison grids |
| `variance_analysis/` | Variance heatmap overlays |
| `intermediate_latents/` | Denoising progression strips |
| `EXPERIMENTS.md` tables | Results bar charts / tables |
| `calibration_analysis/` | Calibration curves |

### 10.3 Figures Still Needed

| Figure | Priority | Source | Status |
|--------|----------|--------|--------|
| Architecture diagram (FLUX + CN + LoRA + DPS) | **High** | Manual / TikZ | Not started |
| Input -> Output -> GT comparison grid (cherry-picked) | **High** | `generate_montages.py` | Script exists, not curated |
| Frame ablation curve (PSNR/LPIPS/CFID vs # frames) | **High** | EXPERIMENTS.md data | Not plotted |
| Multi-seed variance overlay (heatmap on reconstruction) | Medium | `variance_analysis/` | Data exists |
| DPS comparison montage (baseline vs DPS) | Medium | Physics ablation outputs | Not composed |
| Perception-distortion-consistency triangle | Medium | Multi-experiment data | Not plotted |
| SD1.5 vs FLUX probing comparison heatmap | **High** | SD1.5 + FLUX probing results | Data ready, needs composite figure |
| img2img ablation failure montage | Medium | img2img outputs | Data ready |
| Cross-frame variance heatmaps (FLUX vs SD1.5) | Medium | Probing results | **Complete** — both done |
| OD filter ablation results | Medium | Needs OD training | **Blocked** |
| Spatial depth prediction visualization | **High** | Probing outputs | Needs extraction |

---

## 11. Thesis/Paper Structure

### 11.1 Thesis Structure (Suggested)

```
1. Introduction
   - SPAD sensors: motivation (low-power, high-speed, 1-bit binary)
   - Problem: binary sensor -> RGB reconstruction as extreme inverse problem
   - Our approach: diffusion transformer with ControlNet conditioning
   - Contributions summary

2. Related Work
   2.1 SPAD reconstruction methods (traditional + learning-based, gQIR, bit2bit, Quanta Diffusion)
   2.2 Diffusion models for inverse problems (DPS, PaDIS, PnP)
   2.3 VAE domain adaptation for degraded inputs (gQIR predegradation removal, DiffBIR, SUPIR)
   2.4 Model interpretability / probing (AC3D, Beyond Surface Statistics)
   2.5 Uncertainty quantification in generative models

3. Method
   3.1 Architecture: FLUX.1-dev + ControlNet Union Alpha + LoRA
   3.2 Training: FP8 quantization, scene-aware split, LoRA-on-ControlNet
   3.3 Latent-space DPS guidance (Bernoulli likelihood)
   3.4 Multi-frame temporal consistency loss

4. Experiments & Results
   4.1 Baseline reconstruction quality (PSNR/SSIM/LPIPS/FID/CFID)
   4.2 Frame-count ablation (zero-shot multi-frame transfer)
   4.3 Physics guidance (DPS) ablation
   4.4 Multi-seed uncertainty analysis
   4.5 Calibration analysis
   4.6 img2img ablation (no ControlNet) — justifies ControlNet
   4.7 Best-of-K NLL reranking — perception-consistency tradeoff
   4.8 [Consistency epoch sweep — in progress]
   4.9 [OD filter ablation -- partial]
   4.10 [SD1.5 vs FLUX reconstruction comparison -- needs re-eval]

5. "What Does the Model Know?" -- Linear Probing Analysis
   5.1 Bit density preservation (R^2=0.998)
   5.2 Emergent depth encoding (R^2=0.685 spatial)
   5.3 Uncertainty self-awareness (R^2=0.506 spatial)
   5.4 Cross-frame variance encoding (global R^2=0.292, spatial R^2=0.359, CN R^2=0.360)
   5.5 Object recognition from 1-bit data
   5.6 LoRA delta analysis + No-ControlNet ablation
   5.7 Information flow through denoising
   5.8 Cross-architecture comparison: SD1.5 vs FLUX

6. Discussion
   - Why ControlNet already captures the physics
   - Why consistency training didn't help (and what would)
   - Limitations of latent-space DPS
   - Generalization to other sparse sensors

7. Conclusion
```

### 11.2 NeurIPS Paper (9 pages)

Tighter version focusing on:
- Probing as main contribution (Section 5 above becomes the core)
- Architecture + training as brief method section
- DPS as secondary contribution
- Frame ablation + uncertainty as supporting experiments
- Drop consistency training (negative result, mention in discussion)

---

## 12. Presentation Plan

### Slide Outline (20-25 min talk)

```
Slide 1-2:   Motivation -- what are SPADs, why 1-bit, why it matters
Slide 3-4:   Architecture overview -- FLUX + ControlNet + LoRA diagram
Slide 5:     Training setup -- scene-aware split, FP8 quantization
Slide 6-8:   Results gallery -- qualitative examples (diverse scenes)
Slide 9:     Quantitative results table (baseline + multi-seed)
Slide 10:    img2img ablation -- why ControlNet is needed (catastrophic without it)
Slide 11:    Frame ablation -- how multi-frame helps
Slide 12:    DPS physics guidance results + best-of-K tradeoff
Slide 13:    "What does the model know?" -- probing motivation
Slide 14-16: Probing results (heatmap, AC3D curve, delta)
Slide 17:    Depth from 1-bit -- the spatial probing headline
Slide 18:    Cross-frame variance -- model encodes frame sensitivity
Slide 19:    Object recognition from binary data
Slide 20:    SD1.5 vs FLUX -- cross-architecture comparison
Slide 21:    No-ControlNet ablation -- validation
Slide 22:    Uncertainty awareness -- model knows where it guesses
Slide 23:    Discussion + limitations
Slide 24:    Future work
```

### Key Points to Emphasize

1. **The depth result is the headline**: R^2=0.685 from 1-bit binary data, comparable to AC3D's RGB-conditioned depth probing. SD1.5 achieves 0.727 spatial depth.
2. **LoRA delta is the mechanistic insight**: LoRA teaches geometry conversion, not just input preservation
3. **No-ControlNet ablation is the validation**: Confirms all information flows through ControlNet
4. **img2img ablation is the justification**: Without ControlNet, PSNR ~7.5 dB (catastrophic). ControlNet is essential.
5. **Cross-architecture generality**: SD1.5 (1.2B) is surprisingly competitive with FLUX (12B). ControlNet encoding pattern is architecture-general.
6. **Cross-frame variance is a new contribution**: Model encodes frame-dependent sensitivity in early blocks. LoRA amplifies this by 32%.
7. **Uncertainty awareness is the unexpected finding**: Model internally represents where it's guessing
8. **DPS modesty is a feature**: The ControlNet already internalizes the physics

---

## 13. Codebase Map

### 13.1 DiffSynth-Studio-SPAD (Primary Repo)

```
/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/

Core Library (diffsynth/):
  diffsynth/diffusion/
    spad_forward.py          # Bernoulli forward model + NLL (fixed: softplus, log1mexp, sRGB)
    flow_dps.py              # Pixel-space DPS (OOM, implemented but not used)
    latent_dps.py            # Latent-space DPS (used for physics ablation)
    consistency_loss.py      # Per-frame consistency training loss
    flow_match.py            # FlowMatchScheduler (step, add_noise, training_target)
    base_pipeline.py         # BasePipeline (step wrapper)
    loss.py                  # FlowMatchSFTLoss and other losses
  diffsynth/pipelines/
    flux_image.py            # FluxImagePipeline, FluxImageUnit_ControlNet, model_fn
  diffsynth/models/
    flux_controlnet.py       # FluxControlNet, MultiControlNet
    flux_vae.py              # VAE encoder/decoder
    flux_dit.py              # FLUX DiT backbone
  diffsynth/core/
    data/operators.py        # load_spad_image() -- 16-bit fix
  diffsynth/utils/
    lora/flux.py             # FluxLoRALoader
    controlnet.py            # ControlNetInput

Training Scripts:
  train_lora.py              # Base LoRA training (FluxTrainingModule)
  train_consistency.py       # Consistency loss training (extends FluxTrainingModule)
  paired_spad_dataset.py     # Dataset for paired SPAD frames
  train_scene_aware_raw.sh   # RAW baseline training script
  train_consistency.sh       # Consistency training script
  train_od03_finetune.sh     # OD03 fine-tune script
  train_od07_finetune.sh     # OD07 fine-tune script
  train_od03_scratch.sh      # OD03 from-scratch script

Validation/Inference Scripts:
  validate_lora.py           # Standard FLUX inference
  validate_dps.py            # Inference with latent-space DPS
  validate_flow_dps.py       # Inference with pixel-space DPS (OOM)
  validate_crossframe.py     # Cross-frame consistency evaluation
  inference_lora.py          # Single-image inference

Analysis Scripts:
  linear_probing.py          # 3-phase FLUX probing pipeline (MOST COMPLEX)
  probing_analysis.py        # Cross-condition figure generation (7 figures)
  update_probing_report.py   # Auto-update probing report with latest results
  run_metrics.py             # PSNR/SSIM/LPIPS/FID/CFID computation
  metrics.py                 # CFID implementation
  aggregate_metrics.py       # Multi-seed aggregation
  compute_variance_maps.py   # Per-pixel variance analysis
  calibration_analysis.py    # ECE computation
  save_intermediate_latents.py # Latent snapshots
  best_of_k.py              # Best-of-K reranking by NLL
  generate_montages.py       # Comparison montage generation
  frame_vs_seed_variance.py  # Variance decomposition

Orchestration Scripts:
  run_all_experiments.sh       # Master pipeline: SD1.5 probing → img2img → consistency → FLUX spatial
  run_multiseed_validation.sh
  run_frame_ablation.sh
  run_physics_ablation.sh
  run_od_ablation.sh
  run_probing_comprehensive.sh
  run_img2img_ablation.sh      # img2img denoising strength sweep
  train_img2img_ablation.sh    # img2img LoRA-on-DiT training

Validation Scripts (additional):
  validate_img2img.py          # img2img (no ControlNet) validation

Documentation:
  agent/                     # All project context (this directory)
    THESIS_CONTEXT.md        # THIS FILE
    HANDOFF.md               # Complete project state handoff
    DECISIONS.md             # Key technical decisions
    TODO.md                  # Prioritized task list
    TEST_STATUS.md           # Experiment tracking
    INDEX.md                 # Agent directory index
    reports/                 # Analysis reports
    external_plans/          # Cursor/Antigravity plans
    chat_exports/            # Conversation logs
    logs/                    # Training/experiment logs
  EXPERIMENTS.md             # Results tables + reproduction commands
```

### 13.2 spad-diffusion (SD1.5 Baseline Repo)

```
/home/jw/engsci/thesis/spad/spad-diffusion/

Core:
  cldm/cldm.py              # ControlLDM model (SD1.5 ControlNet + ControlledUnetModel)
  cldm/ddim_hacked.py       # Modified DDIM sampler (50 steps, numpy schedule)
  ldm/modules/diffusionmodules/openaimodel.py  # UNetModel (class at line 412)
  ldm/modules/attention.py   # MemoryEfficientCrossAttention (xformers)
  sim_train.py              # Training script (PyTorch Lightning)
  sim_inference.py           # Inference script
  config.py                  # Shared configuration
  models/cldm_v15.yaml       # Config: model_channels=320, channel_mult=[1,2,4,4]
  spad_dataset/
    dataloader.py            # SPAD data loading
    datasets.py              # Dataset classes
    measurement_simulation.py # SPAD measurement simulation
    processors.py            # Image processing pipeline
    extract_binary_images.py # Raw SPAD extraction
    create_RGB.py            # GT RGB creation pipeline

Linear Probing:
  sd15_linear_probing.py     # 3-phase SD1.5 probing pipeline (mirrors FLUX linear_probing.py)
  run_sd15_probing.sh        # SD1.5 probing runner script

Evaluation:
  evaluate_metrics.py        # Metric computation
  metrics.py                 # Metric implementations
  run_metrics.py             # Batch metric runner
  run_sd15_scene_aware_eval.sh # Scene-aware evaluation script

Models:
  lightning_logs/spad_controlnet/ # Trained checkpoints
    two_stage_best/best-epoch=14-val_loss=0.1057.ckpt

Probing Results:
  probing_results_sd15/       # SD1.5 probing output (complete)
    targets.json              # Reused from FLUX (same val set)
    activations/              # Extracted activation features
    probes/                   # Trained probes, heatmaps, results JSONs
```

### 13.3 Output Directories

```
DiffSynth-Studio-SPAD/
  models/train/
    FLUX-SPAD-LoRA-SceneAware-RAW/     # Primary model (40 epochs, best=epoch-15)
    FLUX-SPAD-LoRA-Consistency/         # Consistency model (30 epochs)
    FLUX-SPAD-LoRA-Img2Img-Ablation/   # img2img LoRA-on-DiT (20 epochs)
    FLUX-SPAD-LoRA-On-ControlNet/      # LoRA-on-ControlNet (40 epochs)
    FLUX-SPAD-LoRA_no_conditioning/    # No conditioning ablation (40 epochs)
    FLUX-SPAD-LoRA-SceneAware-OD03-FT/ # OD03 fine-tune (4 epochs, partial)
    FLUX-SPAD-LoRA-SceneAware-OD07-FT/ # OD07 fine-tune (4 epochs, partial)
    FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/ # OD03 from scratch (40 epochs)

  validation_outputs_scene_aware/seed_42/        # Baseline
  validation_outputs_multiseed/seed_{0..123}/    # 10-seed eval
  validation_outputs_physics_ablation/           # DPS eta sweep
  validation_outputs_consistency/epoch-0/        # Consistency model
  validation_outputs_consistency_dps/eta1.0/     # Consistency + DPS
  validation_outputs_frame_ablation/             # Multi-frame inputs (1-1000 frames)
  validation_outputs_crossframe/baseline/        # Cross-frame generation (7 frames)
  validation_outputs_flow_dps/                   # FlowDPS variants
  validation_outputs_best_of_k/                  # Best-of-10 NLL reranking
  validation_outputs_img2img_ablation/           # img2img strength sweep (0.3-0.9)

  probing_results_allblocks/   # Main FLUX probing (776 samples, all blocks, spatial)
  probing_results_control/     # Control probing (no LoRA)
  probing_results_no_cn/       # No-ControlNet ablation
  probing_analysis_output/     # 7 comparison figures (PNG + PDF)

  variance_analysis/           # 776 variance images
  calibration_analysis/        # ECE results
  intermediate_latents/        # 20 samples x 8 steps
```

---

## 14. Current Status and Next Steps

### 14.1 What's Done

- [x] Dataset audit + scene-aware stratified split (zero leakage)
- [x] FLUX RAW baseline training (40 epochs, best=epoch-15)
- [x] K=10 multi-seed validation + aggregated metrics
- [x] Linear probing — all 3 FLUX conditions (main, control, no-CN), 776 samples, all 72 blocks
- [x] Spatial streaming probing — FLUX (bit_density, depth, variance), SD1.5 (all 4 targets)
- [x] Object presence probing (24 objects) for FLUX main + control conditions
- [x] Cross-frame variance probing — global, all 3 FLUX conditions + SD1.5
- [x] SD1.5 linear probing — complete end-to-end (38 blocks × 7 timesteps, global + spatial + objects)
- [x] Physics ablation (DPS eta sweep 0.01-1.0)
- [x] Frame-count ablation (1-1000 frames)
- [x] Consistency training + epoch-0 evaluation (30 epochs)
- [x] img2img ablation — training complete (20 epochs), strength sweep {0.3-0.9} complete
- [x] Best-of-K NLL reranking (K=10, perception-consistency tradeoff confirmed)
- [x] Cross-frame generation (7 frames × 776 images)
- [x] FlowDPS with detection-based guidance (eta=0.1, mid-4 steps)
- [x] Variance analysis (776 images)
- [x] Calibration analysis (ECE=0.269)
- [x] Intermediate latents (20 samples x 8 steps)
- [x] CFID metric with numerical stability
- [x] DPS/consistency code audit and fixes
- [x] Probing analysis figures (7 main figures + SD1.5 heatmaps)

### 14.2 What's In Progress

| Item | Status | ETA |
|------|--------|-----|
| img2img ablation sweep | Strengths 0.3-0.9 done; 1.0 pending | ~1h |
| Consistency epoch sweep | In pipeline (Step 3), epochs 5/10/15/20/25/29 | ~2h |
| ~~FLUX spatial crossframe re-extraction~~ | **Complete** — R²=0.359 | Done |
| OD filter training (OD03-FT, OD07-FT) | Partial (4 epochs each) | Check tmux |

### 14.3 What's Not Started (Priority Order)

| Item | Priority | Effort | Value |
|------|----------|--------|-------|
| Thesis/paper writing | **Critical** | High | Deliverable (due Apr 7) |
| Presentation slides | **Critical** | Medium | Practice Mon Mar 31, actual Wed Apr 2 |
| Curated comparison montages | **High** | Low | Essential for presentation |
| SD1.5 vs FLUX probing composite figure | **High** | Low | Cross-architecture headline |
| Architecture diagram (FLUX + CN + LoRA + DPS) | **High** | Medium | Essential for method section |
| Spatial depth prediction visualization | **High** | Low | Headline result figure |
| Frame ablation curve plot | **High** | Low | Key figure |
| SD1.5 baseline re-evaluation (scene-aware split) | Medium | Low | Fair reconstruction comparison |
| Higher LoRA rank (64 or 128) | Medium | 17h retrain | Cheap ablation |
| Consistency training with lower lambda | Low | Medium | Negative result already interesting |

### 14.4 Completed Architectural Ablations

| # | Experiment | Result | Status |
|---|-----------|--------|--------|
| A1 | **img2img + LoRA-on-DiT (no CN)** | **Catastrophic failure** (PSNR ~7.5 dB). VAE can't encode binary SPAD. Strongly justifies ControlNet. | **Complete** |
| A2 | **SD1.5 cross-architecture probing** | SD1.5 (1.2B) surprisingly competitive. Spatial depth R²=0.727 > FLUX 0.685. Architecture-general ControlNet encoding. | **Complete** |
| A3 | **Cross-frame variance probing** | Global R²=0.292 (DiT), 0.360 (CN); Spatial R²=0.359 (DiT). LoRA amplifies frame sensitivity. Info in early joint blocks at late timesteps; spatial info in late single blocks at mid-denoising. | **Complete** |

### 14.5 Remaining Ablation Experiments

| # | Experiment | What It Tests | Priority |
|---|-----------|---------------|----------|
| A4 | **Higher LoRA rank** (64 or 128) | Is rank-16 under-expressive? | Medium — cheapest experiment |
| A5 | **Dual LoRA** (ControlNet + DiT) | Does DiT benefit from LoRA when CN is also adapted? | Medium |
| A6 | **Full ControlNet fine-tuning** (no LoRA) | Removes LoRA bottleneck | Low |
| A7 | **Custom SPAD encoder** | Is pre-processing 1-bit data beneficial? | Low (NeurIPS stretch) |

---

## 15. Key Decisions Log

| # | Decision | Rationale | Impact |
|---|----------|-----------|--------|
| 1 | LoRA on ControlNet (not DiT) | Empirically better; adapts SPAD interpretation, not prior | All training scripts |
| 2 | Scene-aware stratified split | Original had 94/101 locations leaking | All metrics |
| 3 | epoch-15 as best RAW checkpoint | Lowest val loss (0.3083) | OD fine-tuning base |
| 4 | 16-bit image loading fix | PIL clamps >255; custom normalization | Multi-frame correctness |
| 5 | Latent-space DPS (not pixel-space) | Pixel-space OOM on 32GB | DPS approximation |
| 6 | Metrics in grayscale | SPAD is monochrome; avoids color bias | All evaluation |
| 7 | CFID with Ridge + float64 | Rank-deficient covariance (N=776 << D=2048) | CFID stability |
| 8 | Stop-gradient on F2 in consistency | Two full forward passes OOM | Asymmetric loss |
| 9 | Adaptive ridge lambda for probing | Fixed lambda fails at D/n=38 | Probing stability |
| 10 | All-blocks probing (57 DiT + 15 CN) | Sparse probing missed patterns | Complete information flow |
| 11 | Cross-frame variance as probing target | Captures frame-dependent sensitivity, distinct from seed variance | New contribution (Appendix E) |
| 12 | SD1.5 probing uses `diffsynth` env (not `control2`) | PyTorch 2.4 in control2 doesn't support RTX 5090 sm_120 | Monkey-patch xformers → native SDPA |
| 13 | Spatial streaming with lazy per-key accumulation | SD1.5 has variable feature dims (320/640/1280) vs FLUX uniform 3072 | Correct spatial probing for both architectures |
| 14 | img2img ablation with rank-32 LoRA-on-DiT | Higher rank than CN LoRA (rank-16) to give img2img every advantage | Still catastrophic — validates ControlNet |

See `agent/DECISIONS.md` for full rationale.

---

## 16. Competitor Landscape

| Competitor | Approach | Key Difference from Ours |
|------------|----------|--------------------------|
| **gQIR** (arxiv 2602.20417) | 3-stage: VAE fine-tune → LoRA+distillation → FusionViT burst merging | See detailed analysis below |
| **Quanta Diffusion** (Chennuri et al., 2025) | Embed forward model in diffusion loop, multi-frame video | Forward model in loop, video focus |
| **bit2bit** (NeurIPS 2024) | Self-supervised 1-bit reconstruction via Bernoulli lattice | Self-supervised, video reconstruction |
| **FlowDPS** (Kim et al., ICCV 2025) | DPS for rectified flow models | General method; we apply to SPAD |
| **AC3D** (Bhatt et al., 2024, includes Lindell) | Probed video DiT for camera pose | Our supervisor's paper; we extend to SPAD probing |

### 16.1 Detailed gQIR Analysis (arxiv 2602.20417)

gQIR is the closest competitor — also uses latent diffusion for SPAD reconstruction. Their 3-stage pipeline:

**Stage 1 — "Predegradation Removal" (VAE Encoder Fine-Tuning)**:
- Fine-tune only VAE encoder so degraded SPAD inputs map to clean-image latent manifold
- **Deterministic mean encoding**: Use `μ_φ(x)` only (no sampling from `N(μ,σ²)`). SPAD binary statistics have heavy tails → stochastic sampling amplifies variance
- **Latent Space Alignment (LSA) loss**: `L_lsa = ||μ_φ*(x_lq) - μ_φ(x_gt)||²` — forces fine-tuned encoder outputs to align with frozen encoder's clean-image latents
- **Critical finding**: Without LSA, encoder collapses — PSNR drops 24.78 → 10.30 (Table 4). Naive VAE encoder fine-tuning on SPAD is destructive.
- Decoder frozen throughout
- 600k steps on 8×A100

**Stage 2 — LoRA + Adversarial Distillation**:
- LoRA on SD 2.1 U-Net (not DiT)
- Adversarial distillation to single-step generator (100k iterations)

**Stage 3 — FusionViT**:
- Spatio-temporal transformer for burst latent merging (3-bit nano-bursts from 7 averaged frames)
- 20k steps

**Key Differences from Our Approach**:

| Aspect | gQIR | Ours |
|--------|------|------|
| SPAD entry point | VAE encoder (fine-tuned) | ControlNet (separate encoder) |
| VAE fine-tuning | Required (Stage 1, critical) | Not needed |
| Base model | SD 2.1 (865M UNet) | FLUX 12B DiT + SD1.5 860M UNet |
| Input | 3-bit nano-bursts (7 avg frames) | Single 1-bit binary frame |
| Training data | 2.81M simulated images | ~776 real captures |
| Multi-frame | Yes (FusionViT Stage 3) | Zero-shot frame accumulation |
| Distillation | Yes (1-step adversarial) | No (28-step Euler) |
| Analysis | Reconstruction metrics only | Probing, uncertainty, physics guidance |

**Why we don't need gQIR's Stage 1**: Our ControlNet bypasses the VAE entirely for SPAD conditioning. SPAD → ControlNet → DiT conditioning, never touching the VAE encoder. This is an architectural advantage — simpler pipeline, no risk of encoder collapse, no 600k-step fine-tuning needed.

**Our img2img ablation validates gQIR's finding**: When we *do* force SPAD through the VAE (img2img pathway), we get PSNR ~7.5 dB — even worse than gQIR's "no LSA" baseline (10.30 dB), because we don't fine-tune the encoder at all. This independently confirms the VAE domain gap is fundamental.

**Thesis framing**: *"gQIR demonstrates that VAE fine-tuning is essential when SPAD data enters through the encoder (their Stage 1). Our ControlNet approach avoids this bottleneck entirely, enabling plug-and-play conditioning on frozen foundation models without multi-stage training."*

### 16.2 Our Unique Positioning

- **Single 1-bit frame** (much harder than burst/multi-frame)
- **Real large-scale dataset** (2,500 views, not synthetic)
- **12B parameter model** (largest SPAD reconstruction model to date)
- **Analysis-first approach**: probing + uncertainty + physics, not just reconstruction
- **Extends supervisor's own methodology** (AC3D probing) to new domain
- **Architectural simplicity**: Single-stage ControlNet+LoRA vs gQIR's 3-stage pipeline

---

## 17. References

### Core References

- Chung et al., "Diffusion Posterior Sampling" (ICLR 2023) -- DPS
- Song et al., "PaDIS: Pseudoinverse-Guided Diffusion" (ICLR 2023) -- PaDIS
- Kim et al., "FlowDPS" (ICCV 2025, arxiv 2503.08136) -- DPS for flow matching
- Bhatt, Bahmani, ..., **Lindell**, Tulyakov, "AC3D" (2024) -- probing video DiT
- Cohen et al., "Looks Too Good To Be True" (NeurIPS 2024) -- uncertainty-perception tradeoff
- Chen et al., "Beyond Surface Statistics" (NeurIPS 2023 workshop) -- probing diffusion for depth/saliency

### SPAD/Sensor References

- gQIR (arxiv 2602.20417) -- 3-stage latent diffusion for SPAD: VAE predegradation removal + LoRA distillation + FusionViT burst merging. Key result: LSA loss essential for VAE encoder fine-tuning (see Section 16.1)
- Chennuri et al., "Quanta Diffusion" (2025) -- forward model in diffusion loop
- bit2bit (NeurIPS 2024) -- self-supervised 1-bit reconstruction

### Architecture References

- FLUX.1-dev (Black Forest Labs) -- 12B rectified-flow transformer
- ControlNet (Zhang & Agrawala, 2023) -- conditioning injection
- LoRA (Hu et al., 2022) -- low-rank adaptation

### Additional

- FlowChef (Patel et al., ICCV 2025) -- gradient-free flow steering
- Intrinsic LoRA (Feng et al., 2024) -- extracting scene intrinsics
- PaDIS-MRI (arxiv 2509.21531) -- physics-informed diffusion for MRI
- PIRF (NeurIPS 2025, arxiv 2509.20570) -- physics-informed reward fine-tuning
- Wang et al., "Traversing Distortion-Perception Tradeoff" (CVPR 2025)

---

## Appendix A: Environment Setup

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Working directories
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD  # Primary repo
cd /home/jw/engsci/thesis/spad/spad-diffusion          # SD1.5 baseline

# GPU
nvidia-smi  # RTX 5090, 32GB

# Key conda environments
# diffsynth -- FLUX pipeline + SD1.5 probing (primary, PyTorch 2.9.1 with sm_120)
# control2  -- SD1.5 training/inference (PyTorch 2.4, no sm_120 — use diffsynth instead for RTX 5090)
# sam3      -- SAM3 segmentation
```

## Appendix B: Reproduction Commands

```bash
# Train RAW baseline
bash train_scene_aware_raw.sh

# Validate (single seed)
python validate_lora.py \
  --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
  --val_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
  --output_dir ./validation_outputs_scene_aware/seed_42 --seed 42

# Compute metrics
python run_metrics.py <output_dir> --save

# Multi-seed validation
bash run_multiseed_validation.sh

# Physics ablation
bash run_physics_ablation.sh

# Frame ablation
bash run_frame_ablation.sh

# Linear probing (full pipeline)
python linear_probing.py --prepare-targets
python linear_probing.py --extract --all --save-spatial
python linear_probing.py --train

# Probing analysis figures
python probing_analysis.py

# DPS validation
python validate_dps.py \
  --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
  --dps_eta 1.0 --dps_schedule linear_decay
```

---

*This document was last updated on 2026-03-29. It should be continuously maintained as experiments progress, results are analyzed, and paper/presentation writing advances.*

