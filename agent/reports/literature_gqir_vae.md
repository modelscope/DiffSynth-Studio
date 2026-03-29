# gQIR Paper Analysis & VAE Domain Gap in SPAD Reconstruction

**Date**: 2026-03-29
**Source**: arXiv 2602.20417 ("gQIR: Generative Quantized Image Restoration for SPAD")
**Relevance**: Closest competitor; their findings validate our architectural decisions

---

## 1. gQIR Overview

gQIR is a 3-stage latent diffusion framework for SPAD → RGB reconstruction. Unlike our ControlNet approach, gQIR routes SPAD data directly through the VAE encoder, necessitating a multi-stage training pipeline.

### Stage 1: Predegradation Removal (VAE Encoder Fine-Tuning)

The core contribution. The term "predegradation removal" comes from the DiffBIR/SUPIR literature — making the VAE encoder degradation-aware so degraded inputs map to the same latent manifold as clean inputs.

**Problem**: Standard VAE encoder (SD 2.1) produces OOD latents from SPAD input because:
- Binary {0,1} images are bimodal — VAE trained on continuous-tone natural images (mean ~128)
- Sparse SPAD frames (~5-10% active pixels) produce near-zero-energy latents
- Stochastic sampling `z ~ N(mu, sigma^2)` amplifies heavy-tailed SPAD statistics

**Solution**: Two-part fix:
1. **Deterministic mean encoding**: Use only `mu_phi(x)`, bypass the `sigma` branch entirely. This prevents variance explosion from SPAD's extreme binary statistics.
2. **Latent Space Alignment (LSA) loss**:
   ```
   L_lsa = ||mu_phi*(x_lq) - mu_phi(x_gt)||^2
   ```
   Forces the fine-tuned encoder (`phi*`) to produce latents for degraded SPAD inputs that are aligned with the frozen encoder's clean-image latents. Without this, the encoder collapses — fine-tuned latents drift off-manifold.

**Critical ablation (gQIR Table 4)**:

| Encoder Config | PSNR | Notes |
|---------------|------|-------|
| Frozen encoder (no fine-tune) | ~7-8 | Comparable to our img2img result |
| Fine-tuned w/o LSA | 10.30 | Encoder collapse |
| Fine-tuned w/ LSA (deterministic) | 24.78 | Working — LSA is essential |

**Training**: 600k steps on 8x A100 GPUs. Decoder stays frozen throughout.

### Stage 2: LoRA + Adversarial Distillation

- LoRA adapters on SD 2.1 U-Net
- Adversarial distillation to single-step generator
- 100k iterations
- Not directly comparable to our approach (we use 28-step Euler sampling)

### Stage 3: FusionViT (Burst Merging)

- Spatio-temporal transformer that merges latents from multiple SPAD frames
- Input: 3-bit "nano-bursts" (7 averaged binary frames)
- 20k training steps
- Achieves multi-frame reconstruction via latent fusion rather than pixel accumulation

---

## 2. Comparison with Our Pipeline

### Fundamental Architectural Difference

```
gQIR:    SPAD --> [VAE Encoder*] --> latent --> [U-Net+LoRA] --> [VAE Decoder] --> RGB
                  ^^ must be fine-tuned (Stage 1)

Ours:    SPAD --> [ControlNet] --> conditioning --|
         RGB  --> [VAE Encoder] --> latent -------|-- [FLUX DiT] --> [VAE Decoder] --> RGB
                  ^^ only sees clean RGB                (frozen)
```

The key insight: **ControlNet acts as a domain-specific encoder** that translates SPAD → conditioning signals without touching the VAE. This is architecturally superior for several reasons:

1. **No VAE fine-tuning needed**: The VAE encoder only ever sees natural RGB images (on-manifold). No risk of encoder collapse, no LSA loss needed, no 600k-step Stage 1.
2. **Frozen foundation model**: FLUX DiT stays frozen in FP8 — we never modify the generative prior.
3. **Single-stage training**: LoRA-on-ControlNet in one pass, vs gQIR's 3 separate stages.
4. **Modular conditioning**: ControlNet can be swapped, composed, or removed without affecting the base model.

### Where gQIR Has Advantages

1. **Multi-frame input**: FusionViT in latent space is potentially more principled than our zero-shot frame accumulation (average in pixel space → re-encode)
2. **Single-step inference**: Adversarial distillation enables 1-step generation (ours: 28 steps)
3. **Scale of training data**: 2.81M simulated images (ours: ~776 real captures)

### Where We Have Advantages

1. **Single 1-bit frame**: Much harder setting — no burst averaging. Tests the prior's limits.
2. **Real data**: 2,500 views from actual SPAD sensor, not simulated Poisson processes.
3. **12B parameter model**: FLUX DiT is 14x larger than SD 2.1 — stronger generative prior.
4. **Analysis depth**: Probing, uncertainty quantification, cross-architecture comparison — gQIR reports only reconstruction metrics.
5. **Architectural simplicity**: 1 stage vs 3. No adversarial training, no encoder surgery.

---

## 3. The VAE's Role in Our Pipeline

The FLUX VAE (AutoencoderKL, 16-channel latent space) is still load-bearing but never processes SPAD:

### During Training
- **Encoder**: GT RGB → latent targets. The flow matching loss operates in this latent space. The encoder defines the target distribution the DiT+ControlNet must produce.
- **Decoder**: Not used during training.

### During Inference
- **Encoder**: Not used.
- **Decoder**: Converts denoised latents → final RGB pixels. Every reconstruction image comes out of the frozen FLUX decoder. This is the final pixel generator.

### What This Means
- The VAE defines the **geometry of the latent space** — what the DiT learns to generate.
- The VAE decoder's quality sets an **upper bound on output fidelity** — if the decoder can't represent a texture or color, neither can our reconstruction.
- But neither component ever needs to understand SPAD data. The ControlNet handles that translation.

### Would VAE Fine-Tuning Help?
No. The VAE is **completely frozen** (`freeze_except` sets `requires_grad_(False)` on everything; VAE is never listed as trainable). It only translates between pixel space and latent space for well-formed RGB:
- **Encoder**: Only ever sees clean GT RGB → latent targets. On-manifold by definition.
- **Decoder**: Only ever sees DiT-generated latents → RGB pixels.
- SPAD data never enters the VAE at any point. There is nothing to fix.

### Future Work: Custom SPAD Encoder
A potentially valuable experiment: add a learned front-end before ControlNet:
```
SPAD binary {0,255} --> [Learned SPAD Encoder (lightweight CNN)] --> richer features --> [ControlNet] --> DiT
```
Currently ControlNet receives raw binary frames. A domain-specific encoder could map SPAD → more informative representations (e.g., local photon density estimates, edge features). This is gQIR's "predegradation removal" concept applied to the ControlNet input path instead of the VAE. Low implementation cost, clean experiment for future work.

---

## 4. img2img Ablation Explained by gQIR

Our img2img experiment (Section 6.6 of THESIS_CONTEXT.md) independently confirms gQIR's core finding:

**Our img2img setup**: SPAD → FLUX VAE encoder (frozen, unfine-tuned) → blend with noise → DiT+LoRA denoises → VAE decoder → RGB

**Result**: PSNR ~7.5 dB, LPIPS >1.0 — catastrophic failure at all denoising strengths.

**gQIR's explanation**: Without fine-tuning + LSA loss, the VAE encoder produces OOD latents from SPAD input. The bimodal binary distribution maps to latents representing "very dark images" rather than structured scenes. The DiT cannot recover meaningful content from these latents regardless of LoRA quality.

**Additional compounding factor in our case**: The img2img LoRA was trained on RGB data only (`--data_file_keys "image"`). It never saw SPAD during training — so even if the VAE produced reasonable latents, the LoRA has no SPAD→RGB mapping.

**Thesis argument**: Our img2img ablation + gQIR Table 4 together constitute strong evidence that ControlNet is the right architectural choice for SPAD conditioning on frozen diffusion models. The VAE domain gap is fundamental, not a hyperparameter issue.

---

## 5. Framing for Thesis Related Work Section

Suggested Related Work paragraph on VAE domain adaptation:

> Recent work on image restoration with latent diffusion models has highlighted the challenge of encoding degraded inputs into the pretrained VAE's latent space. DiffBIR and SUPIR introduce "predegradation removal" modules to bridge this domain gap. For SPAD reconstruction specifically, gQIR [ref] demonstrates that naive VAE encoding of binary SPAD frames produces out-of-distribution latents, and proposes deterministic mean encoding with a latent space alignment (LSA) loss to prevent encoder collapse (PSNR improves from 10.30 to 24.78 dB with LSA). Our approach sidesteps this challenge entirely: by conditioning through ControlNet rather than the VAE encoder, SPAD data never enters the latent encoding pathway. This architectural choice eliminates the need for multi-stage training while maintaining the full pretrained generative prior.

---

## 6. Key Takeaways

1. **We don't need gQIR's Stage 1** — ControlNet is our "predegradation removal" by architecture.
2. **Our img2img ablation validates gQIR** — unfine-tuned VAE + SPAD = catastrophic failure, independently confirming their core finding.
3. **gQIR is complementary, not competitive** — they solve a different input regime (multi-frame bursts + simulated data) with a different architecture. Our contribution is the analysis framework (probing, uncertainty) on single-frame real data.
4. **The VAE still matters for us** — as the decoder that produces final pixels and the encoder that defines the training target space — but it never needs to understand SPAD.
5. **Cite gQIR in both related work AND the img2img discussion** — their Table 4 is direct evidence supporting our ControlNet justification.
