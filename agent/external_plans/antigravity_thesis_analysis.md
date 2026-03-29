# SPAD Thesis Analysis: Training Feasibility, Performance, & Presentation Planning

## 1. Full ControlNet Training — Will Removing LoRA Help?

### Short Answer: Likely yes, but modestly

**Your current setup (ControlNet + LoRA)** trains a rank-32 LoRA adapter on the 15-block ControlNet Union Alpha while keeping both the ControlNet and DiT frozen in FP8. This is a parameter-efficient approach — LoRA adapts the ControlNet's conditioning pathway without modifying the base weights.

**Full ControlNet training** (fine-tuning all ControlNet parameters in bf16/fp32) has these theoretical advantages:

| Aspect | LoRA-on-ControlNet | Full ControlNet Fine-tuning |
|--------|-------------------|---------------------------|
| **Trainable params** | ~2-5M (rank-32 LoRA on 15 blocks) | ~500M+ (all ControlNet weights) |
| **Expressiveness** | Limited to low-rank subspace | Full parameter space |
| **Risk of overfitting** | Low (constrained) | Higher (needs careful LR + regularization) |
| **Expected gain** | Baseline | +1-3 dB PSNR (estimated) |

**Why I think the gain will be moderate, not dramatic:**

1. **Your LoRA already captures the key signal.** The probing results show R²=0.998 for bit density — the ControlNet is already near-perfectly preserving SPAD measurements. LoRA's main contribution is depth encoding (+0.308 R² delta).

2. **The bottleneck may not be ControlNet capacity.** Your PSNR of 17.99 is limited by the inherent ambiguity of 1-bit binary→RGB reconstruction. Full ControlNet training gives better expressiveness but can't resolve fundamental information loss.

3. **The ControlNet Union Alpha was already pre-trained well.** It's a general-purpose architecture designed for many conditioning types. Full fine-tuning lets it specialize for SPAD, but the starting point is already strong.

4. **Overfitting risk.** With only 1850 training views, full ControlNet fine-tuning may overfit if not carefully regularized. Your LoRA approach naturally constrains the effective parameter count.

> [!IMPORTANT]
> **Recommendation**: Full ControlNet training should give a noticeable but not revolutionary improvement — expect **+0.5-2 dB PSNR** and improved perceptual quality. The probing analysis would be very interesting to re-run on the full ControlNet model to see if the depth/uncertainty encoding improves further.

---

## 2. Can Full ControlNet Training Fit on RTX 5090 (32GB)?

### Memory Budget Analysis

The FLUX.1-dev ControlNet training requires:

| Component | FP8 | BF16 | FP32 |
|-----------|-----|------|------|
| **DiT backbone** (12B params) | ~12 GB | ~24 GB | ~48 GB |
| **Text encoder T5-XXL** (~4.7B) | ~4.7 GB | ~9.4 GB | ~18.8 GB |
| **Text encoder CLIP** | ~0.5 GB | ~1 GB | ~2 GB |
| **ControlNet** (~500M params) | ~0.5 GB | ~1 GB | ~2 GB |
| **VAE** | ~0.2 GB | ~0.4 GB | ~0.8 GB |
| **Total model weights** | ~18 GB | — | — |

For training (forward pass only isn't enough — need gradients + optimizer states):

| Component | Memory |
|-----------|--------|
| Frozen models in FP8 (DiT + T5 + CLIP + VAE) | ~17.5 GB |
| ControlNet weights in BF16 (trainable) | ~1 GB |
| ControlNet gradients (BF16) | ~1 GB |
| AdamW optimizer states (2× FP32 = 4 bytes/param) | ~2 GB |
| Activations/computation graph | ~4-8 GB |
| **Total estimate** | **~25-30 GB** |

### The DiffSynth FP8 Limitation

The core problem: **DiffSynth Studio's FP8 quantization applies to entire model files as specified in `--fp8_models`.** You currently list the ControlNet in `FP8_MODELS`:

```
FP8_MODELS="...,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"
```

If you want to train the ControlNet (not just LoRA), you'd need to:
1. **Remove ControlNet from `FP8_MODELS`** (keep it in BF16 for gradients)
2. Keep DiT + text encoders in FP8 (frozen)

### Can It Fit?

> [!WARNING]
> **Tight but potentially feasible with aggressive optimization.**

With gradient checkpointing + BF16 mixed precision + FP8 frozen models:
- Frozen FP8 models: ~17.5 GB
- ControlNet BF16 + gradients + Adam: ~4 GB
- Activations with gradient checkpointing: ~4-6 GB
- **Total: ~25-28 GB** ← technically fits in 32 GB but leaves very little headroom

**Options to make it work:**

1. **Reduce `MAX_PIXELS`**: Go from 512×512 (262144) → 384×384 (147456) to cut activation memory by ~45%
2. **Gradient accumulation**: Train with `batch_size=1` + `grad_accum=4` to reduce peak memory
3. **Offload optimizer states to CPU**: This is ~2GB saved on GPU but slows training
4. **Use `torch.cuda.amp` aggressively**: Ensure all non-trainable computation stays in FP8/BF16

> [!TIP]
> **The H100 approach is safer.** The H100 has 80GB VRAM — the same training would use ~30GB, leaving ample headroom. If the cluster job is already running, let it complete there.

---

## 3. Presentation & Report Planning

### 3.1 Thesis Structure (Suggested)

Based on the three pillars from HANDOFF.md and the complete experimental results:

```
1. Introduction
   - SPAD sensors: motivation (low-power, high-speed, but 1-bit binary)
   - Problem: binary sensor → RGB reconstruction 
   - Our approach: diffusion transformer with ControlNet conditioning

2. Related Work
   - SPAD reconstruction methods (traditional + learning-based)
   - Diffusion models for inverse problems (DPS, PnP)
   - Model interpretability / probing (AC3D, etc.)

3. Method
   3.1 Architecture: FLUX.1-dev + ControlNet Union Alpha + LoRA
   3.2 Training: FP8 quantization, scene-aware split, LoRA-on-ControlNet
   3.3 Latent-space DPS guidance (Bernoulli likelihood in latent space)
   3.4 Multi-frame temporal consistency

4. Experiments & Results
   4.1 Baseline reconstruction quality (PSNR/SSIM/LPIPS/FID/CFID)
   4.2 OD filter ablation (if results available)
   4.3 Frame-count ablation (zero-shot multi-frame transfer)
   4.4 Physics guidance (DPS) ablation
   4.5 Multi-seed uncertainty analysis
   4.6 SD1.5 vs FLUX comparison (if available)

5. "What Does the Model Know?" — Linear Probing Analysis 
   5.1 Bit density preservation (R²=0.998)
   5.2 Emergent depth encoding (R²=0.685 spatial)
   5.3 Uncertainty self-awareness (R²=0.506 spatial)
   5.4 Object recognition from 1-bit data
   5.5 LoRA delta analysis + No-ControlNet ablation
   5.6 Information flow through denoising

6. Discussion
   - Why consistency training didn't help
   - Limitations of latent-space DPS
   - Generalization to other sparse sensors

7. Conclusion
```

### 3.2 Key Figures for Presentation (Priority Order)

| # | Figure | Source | Why It's Compelling |
|---|--------|--------|-------------------|
| 1 | **Input → Output grid** (SPAD / Reconstruction / GT) | `validation_outputs_scene_aware/` | First-impression "wow factor" |
| 2 | **Probing heatmap** (fig1) | `probing_analysis_output/fig1_main_heatmap.png` | Shows WHERE in network knowledge lives |
| 3 | **Main vs Control delta** (fig3) | `probing_analysis_output/fig3_delta_heatmap.png` | Shows LoRA's specific contribution |
| 4 | **Spatial depth prediction** | `probing_results_allblocks/probes/` | R²=0.685 from 1-bit data → depth map |
| 5 | **Information flow** (fig4) | `probing_analysis_output/fig4_best_timestep_lineplot.png` | AC3D-style "when does understanding emerge" |
| 6 | **Object recognition** (fig7) | `probing_analysis_output/fig7_object_probing.png` | Semantic understanding from binary data |
| 7 | **Frame ablation curve** | Generate from EXPERIMENTS.md table | Diminishing returns analysis |
| 8 | **Multi-seed variance map** | `variance_analysis/` | Uncertainty visualization |
| 9 | **No-ControlNet ablation** | fig2/fig3 | The "it's all zeros without ControlNet" null result |

### 3.3 Presentation Flow (20-25 min talk)

```
Slide 1-2:  Motivation — what are SPADs, why 1-bit, why it matters
Slide 3-4:  Architecture overview — FLUX + ControlNet + LoRA diagram
Slide 5:    Training setup — scene-aware split, FP8 quantization
Slide 6-8:  Results gallery — qualitative examples (diverse scenes)
Slide 9:    Quantitative results table (baseline + DPS + multi-seed)
Slide 10:   Frame ablation — how multi-frame helps
Slide 11-12: OD filter results (if available)
Slide 13:   "What does the model know?" — probing motivation
Slide 14-16: Probing results (heatmap, delta, AC3D curve)
Slide 17:   Depth from 1-bit — the spatial probing "headline" result
Slide 18:   Object recognition — the surprise finding
Slide 19:   No-ControlNet ablation — validation + narrative
Slide 20:   Uncertainty awareness — model knows where it guesses
Slide 21:   Discussion + limitations
Slide 22:   Future work (full ControlNet, better DPS, active sensing)
```

### 3.4 What's Still Missing for Report/Presentation

| Item | Status | Priority for Report |
|------|--------|-------------------|
| **OD filter results** | Training on H100 / was on RTX 5090 | High — core ablation |
| **SD1.5 comparison** | Script ready, not run | High — shows FLUX advantage |
| **Full ControlNet results** | Training on H100 | Medium — shows LoRA limitation |
| **Frame-vs-seed variance decomp.** | Not started | Medium — insight into uncertainty |
| **Qualitative figure curation** | Not done | High — first impression matters |
| **Results tables from EXPERIMENTS.md** | Data exists, needs LaTeX | Low — mechanical |

> [!IMPORTANT]
> The **probing analysis is your strongest, most novel contribution**. The depth-from-1-bit-data result (R²=0.685) and the LoRA delta analysis are genuinely NeurIPS-worthy findings. Make these the centerpiece of both presentation and report.
