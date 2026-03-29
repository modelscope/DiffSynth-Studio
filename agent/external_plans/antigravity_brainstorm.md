# SPAD Thesis Brainstorm: Experiments, Presentation, Report

## 1. What's Done vs What's Missing

### ✅ Completed Experiments

| Experiment | Key Finding |
|---|---|
| **Baseline** (single-frame → RGB) | PSNR 17.99, SSIM 0.596, LPIPS 0.415 |
| **Multi-seed** (K=10) | Low variance: ±0.09 PSNR — model is stable |
| **DPS ablation** (latent-space, eta sweep) | +0.16 PSNR, -0.87 FID at eta=1.0 — modest |
| **Frame-count ablation** (1→1000 frames) | LPIPS improves (0.415→0.347), PSNR degrades (model trained on single-frame) |
| **Consistency training** | Hurt performance (epoch-0 best) — negative result |
| **Linear probing** | Bit density R²=0.99, Depth R²=0.64, Uncertainty R²=0.41 — **strongest result** |
| **Variance analysis** | Mean σ²=0.0061, bit-density correlation -0.226 |
| **Calibration** | ECE=0.269 (underconfident) |
| **Intermediate latents** | 20 samples × 8 steps decoded |
| **OD filter training** | Was running (OD03-FT, OD07-FT, OD03-scratch) |

### ❌ Missing / Not Yet Run

| Experiment | Status | Effort |
|---|---|---|
| **Frame-vs-seed variance decomposition** | Script exists (`frame_vs_seed_variance.py`), but inference not run | **Heavy** — needs K=10 seeds × 7 frame folders × 776 images |
| **Best-of-K reranking** | Not implemented | **Light** — pick from existing 10 seeds by NLL |
| **NLL as evaluation metric** | Not implemented | **Light** — compute NLL on all outputs |
| **Organized montage comparisons** | Not done | **Medium** — scripting needed |
| **SD1.5 fair re-evaluation** | Script ready, not run | **Medium** — needs GPU time |
| **OD ablation evaluation** | Blocked on training completion | Depends |

---

## 2. How DPS Works — Your Questions Answered

### DPS is inference-only

Yes — **DPS does not modify training at all.** It works by injecting gradient-based corrections during the denoising loop at inference time:

```
For each denoising step t:
  1. Model predicts velocity v_θ(x_t, t)
  2. Estimate clean image: x̂₀ = x_t - σ·v_θ
  3. Compute measurement loss: L = -log p(y | x̂₀)
  4. Backprop to get ∂L/∂x_t
  5. Add correction to velocity: v_corrected = v_θ + η·∇L
  6. Step: x_{t+1} = x_t + v_corrected·(σ_{t+1} - σ_t)
```

### Latent vs Pixel DPS — they ARE different

You have **two distinct implementations**:

| | `latent_dps.py` (what's been evaluated) | `flow_dps.py` (pixel-space, untested at scale) |
|---|---|---|
| **Loss** | `‖x̂₀ - z_spad‖²` (L2 in latent space) | Bernoulli NLL: `-[y·log(p) + (1-y)·log(1-p)]` |
| **Physics model** | None — just latent consistency | Full: sRGB→linear→exposure→Bernoulli |
| **Backprop** | Trivial (closed-form gradient) | Through VAE decoder at every step |
| **VRAM** | ~0 extra | **OOM on 32GB** |
| **Quality** | Heuristic approximation | Physically correct |

> [!IMPORTANT]
> The latent DPS is comparing the predicted latent against the *VAE-encoded SPAD binary image*. This has no physical meaning — it's just saying "make the output look more like the input in latent space." That's very different from enforcing Bernoulli consistency.

### Training with multiple frames?

Your consistency training (`train_consistency.py`) already does something related — it processes two different single-frame realizations through ControlNet and penalizes their noise predictions being different. But it uses a stop-gradient on frame 2 (VRAM constraint) and lambda=0.1 was too strong.

What you're describing ("leverage multiple frames as conditioning during training") could be:
1. **Multi-frame input during training**: Feed accumulated frames as ControlNet input during training (you already evaluated this zero-shot in the frame ablation)
2. **Contrastive/consistency between frames**: What consistency training does
3. **Average multiple single-frame predictions**: Best-of-K or ensemble at inference

---

## 3. Analysis of What's Worth Running

### 🟢 High Value, Low Effort

1. **Best-of-K Reranking**: You already have K=10 seeds generated. For each image, compute NLL under the SPAD forward model and pick the best one. This is a pure post-processing experiment — needs a script but no GPU inference.

2. **NLL as an Evaluation Metric**: Add NLL to your metrics table. This tells a story: "how measurement-consistent is each method?" Even if DPS doesn't improve PSNR, if it improves NLL, that's meaningful.

3. **Organized Comparison Montages**: Script to generate side-by-side grids: `[SPAD input | Baseline | DPS | Consistency | Ground Truth]` for curated examples. Essential for thesis figures.

### 🟡 Medium Value, Medium Effort

4. **Frame-vs-Seed Variance Decomposition**: The script exists but needs the inference outputs first. This is the experiment you said was missing. **Estimated GPU time**: 7 frames × 10 seeds × 776 images × ~2s/image ≈ 30 hours. That's very heavy. Consider:
   - Subsample: 50-100 images instead of 776
   - Fewer seeds: 3-5 instead of 10
   - This could be a good figure: "variance pie chart" showing measurement vs seed contribution

5. **Formal Frame Consistency Analysis**: You said you see some differences between DPS/consistency frames but need formal analysis. Compute:
   - Per-pixel absolute difference between outputs from different frame realizations (same seed)
   - SSIM between outputs from different frames
   - t-test or effect size (Cohen's d) vs baseline

### 🔴 Lower Priority / Questionable Value

6. **Re-exploring Consistency Training** with lower lambda (0.01, 0.001): The negative result is already interesting for the paper narrative.

7. **Pixel-space DPS**: OOM on 32GB. Not worth pursuing unless you can get a larger GPU.

---

## 4. Paper/Presentation Narrative

### DPS Story — You DON'T need it to work spectacularly

The latent DPS shows a modest improvement (+0.16 PSNR, -0.87 FID). The paper framing should be:

> "We find that the ControlNet's learned conditioning already captures the measurement information so effectively that explicit physics-based guidance provides only marginal additional benefit. The linear probing analysis supports this: the model's internal activations demonstrate near-perfect encoding of photon density (R²=0.99) and strong implicit 3D geometry understanding (R²=0.64), suggesting that the physics is already internalized through training."

### Key Figures for Thesis

1. **Hero figure**: SPAD → reconstruction examples (cherry-picked good ones)
2. **Linear probing heatmaps**: Already at `probing_results/probes/` — these are your strongest results
3. **Frame ablation curve**: LPIPS vs # frames (shows graceful degradation)
4. **Multi-seed variance**: Show that the model is stable across seeds
5. **DPS comparison montage**: Show subtle measurement-consistency improvements
6. **Architecture diagram**: FLUX + ControlNet + LoRA + optional DPS
7. **Variance decomposition** (if you run it): Pie chart of measurement vs seed variance

---

## 5. Montage / Figure Organization Plan

> [!TIP]
> Don't delete anything! Just create a new organized output structure.

Proposed script: `generate_thesis_figures.py` that:
1. Picks N curated examples (good, bad, interesting cases)
2. For each example, creates a horizontal montage: `[Input | Output | GT]`
3. Creates comparison grids across methods: `[Baseline | DPS | Consistency]`
4. Saves variance overlay images (per-seed variance as heatmap on reconstruction)
5. Outputs to a new `thesis_figures/` directory

---

## 6. Concrete Next Steps (Priority Order)

1. **[Quick Win]** Write `compute_nll_metric.py` — evaluate NLL on all existing outputs
2. **[Quick Win]** Write `best_of_k_reranking.py` — pick best of 10 seeds by NLL, compute metrics
3. **[Quick Win]** Write `generate_thesis_figures.py` — organized montages
4. **[If GPU free]** Run frame-vs-seed variance (subsampled)
5. **[Paper writing]** Use the organized figures + probing results to draft thesis chapters
6. **[If time]** Check OD training status and run `run_od_ablation.sh`

### Questions for you:
1. Do you want me to implement the Best-of-K and NLL metric scripts now?
2. For montages, how many curated examples do you want? (I'd suggest 8-12 diverse ones)
3. For frame-vs-seed variance: should I modify the script to subsample (e.g., 100 images, 5 seeds) to make it tractable?
4. Is the OD training finished? Should I check its status?
