# Code Review: Linear Probing (AC3D-Inspired)

## Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| [linear_probing.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/linear_probing.py) | 723 | 3-phase pipeline: target prep → activation extraction → probe training |
| [probing_results.json](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/probing_results/probes/probing_results.json) | 1694 | Results for 6 target types × 70 (block, timestep) combos |
| [AC3D paper](https://arxiv.org/abs/2411.18673) | — | Reference methodology (Section 3.4) |

## AC3D Methodology Summary (from Paper Section 3.4)

AC3D's linear probing:
- **Task**: Predict camera pose (rotation pitch/yaw/roll + translation) from internal activations
- **Dataset**: **1,000 videos** from RealEstate10K
- **Activations**: Per-block outputs of the video DiT (CogVideoX)
- **Probe**: Linear **ridge regression** (the paper says "linear regression model")
- **Granularity**: Each block probed independently; produces R² vs layer index plot (Figure 5)
- **Key finding**: Camera knowledge peaks in middle layers (#13–21), starts from block 9. This motivated limiting ControlNet injection to first 30% of architecture.

---

## Faithfulness Assessment

### 🟢 What's Faithful to AC3D

1. **Ridge regression probe** — Correct. AC3D uses linear regression; your code uses ridge with adaptive λ. This is a strict improvement (regularization helps when n<D).
2. **Per-block, per-timestep probing** — Correct. Probes 5 joint blocks × 7 timesteps + 5 single blocks × 7 timesteps = 70 probes.
3. **Hook-based activation extraction** — Correct. Forward hooks on DiT joint and single blocks capture intermediate representations without modifying the model.
4. **Joint vs. single block separation** — Good. The plot distinguishes joint blocks (J0, J4, J9, J14, J18) from single blocks (S0, S9, S19, S28, S37), with a dividing line.
5. **Heatmap + line plot visualization** — Faithful to AC3D Figure 5 style.

### 🔴 Critical: Global Mean-Pooling Probes Are Broken

The global probing R² values are **catastrophically negative** (all between -8 and -50). This is not a subtle issue — it means the probes predict *worse than predicting the mean*.

**Root cause: n ≪ D overfitting with 3072-dim features and only 100 samples.**

| Setup | n_train | n_test | D (features) | D/n ratio |
|-------|---------|--------|---------------|-----------|
| Your global | 80 | 20 | 3072 | 38.4× |
| AC3D | ~800 | ~200 | varies | ~varies |

With D/n = 38, even ridge regression overfits massively. The probe memorizes the 80 training points perfectly but produces garbage on the 20 test points. The Pearson r values look better (0.7–0.99 for bit density) because Pearson r is scale-invariant and some correlation direction is preserved, but the magnitude is completely wrong.

> [!CAUTION]
> **The global probing results are not publishable.** R² of -10 to -50 means the probe is far worse than a constant predictor. You cannot report these in a paper. The spatial probing results (discussed below) are correct and publishable.

**AC3D avoids this because:**
1. They use 1000 videos (→ ~800 train samples), not 100
2. Their probing target (camera pose) is 6D (3 rotation + 3 translation), not a scalar — so overfitting is less severe
3. They likely probe at the spatial-token level (per-frame features), not global mean-pooled

### 🟢 Spatial Probing Works Well

The spatial probing results are sensible:

| Target | Best R² | Best (block, timestep) | Interpretation |
|--------|---------|----------------------|----------------|
| spatial_bit_density | **0.991** | joint_14, t=0 | Model near-perfectly encodes SPAD input |
| spatial_depth | **0.641** | single_9, t=14 | Depth is linearly decodable — strong result! |
| spatial_variance | **0.411** | single_9, t=14 | Some uncertainty encoded — moderate signal |

This works because spatial probing has n = 100 images × 1024 tokens = ~80,000 training tokens, and D = 3072 features — a healthy D/n ratio of 0.038.

> [!IMPORTANT]
> **The spatial results are the ones you should report in the paper.** They have valid R² and show clear patterns across layers and timesteps. The global probes should be dropped or fixed.

---

## Issue-by-Issue Breakdown

### 🟡 Issue 1: Depth Model Mismatch

**Plan says**: Use ml-depth-pro (Apple's model).
**Code uses**: DPT-Hybrid from Intel/Hugging Face (`Intel/dpt-hybrid-midas`).

```python
# line 165
dm = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").cuda().eval()
```

DPT-Hybrid is a good monocular depth model but not state-of-the-art. ml-depth-pro is Apple's latest and likely more accurate. For probing, this matters — you want the best possible pseudo-GT depth to avoid introducing noise that masks the signal.

**Also**: If DPT fails (exception handler at line 184), it silently falls back to **luminance as depth proxy** — which is a completely different signal and would confuse the probe results.

### 🟡 Issue 2: Not Enough Samples for Global Probes

AC3D uses 1000 videos; you use 100 (`--max_samples 100`). For spatial probing this is fine (100 × 1024 = 102,400 tokens). For global probing, 100 samples with D=3072 features is insufficient.

**Fix**: Either:
1. Run with `--max_samples 776` (full val set) — this gives a D/n ratio closer to 5, still high but ridge may handle it
2. Add PCA dimensionality reduction before ridge (reduce D from 3072 to e.g. 50–100) — this is standard practice in probing papers
3. **Drop global probing entirely** and only report spatial — the spatial results tell a richer story anyway

### 🟡 Issue 3: Ridge Lambda May Be Too Small

```python
# line 375
lam_scaled = lam * XtX.trace() / D
```

With `lam=1e-3`, the effective regularization is `1e-3 × trace(XTX)/D`. In the n≪D regime, `XTX` is rank-deficient (rank ≤ 80 for 80 training samples), so `trace(XTX)/D` is small. The regularization may not be strong enough.

For the n≪D case, you want λ proportional to the largest eigenvalue, not the trace divided by D. Try `lam=0.1` or `1.0`.

### 🟡 Issue 4: No Input Standardization (Spatial Probing)

The spatial ridge regression at [line 466–484](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/linear_probing.py#L466-L484) uses streaming XTX accumulation, which is correct for memory efficiency. However, unlike the global probe (which normalizes X via `(X - mu) / sd` at line 369), the spatial probe does **not normalize the features**. This can cause numerical issues and make the regularization strength inconsistent across blocks.

### 🟡 Issue 5: Block Coverage Is Sparse

You probe 5 out of 19 joint blocks and 5 out of 38 single blocks:

```python
JOINT_BLOCK_IDS = [0, 4, 9, 14, 18]      # 5/19 (sampled)
SINGLE_BLOCK_IDS = [0, 9, 19, 28, 37]    # 5/38 (sampled)
```

AC3D probes **all** blocks. With 10 out of 57 total blocks, you might miss the peak — especially if depth knowledge peaks at, say, joint block 11 or single block 15.

For a paper figure, probing all blocks is worth the cost (each probe is just a ridge solve, which takes milliseconds; the expensive part is activation extraction, which you've already done). You could potentially hook all blocks during extraction and then probe all of them.

### 🟢 Issue 6: Timestep Selection

```python
TIMESTEP_INDICES = [0, 4, 9, 14, 19, 24, 27]  # 7 out of 28 steps
```

This is reasonable coverage (early → mid → late denoising). AC3D uses all timesteps for their temporal analysis. Probing all 28 would be richer but the 7-point sample should capture the temporal dynamics.

### 🟡 Issue 7: No Segmentation Probing

The plan specifies three probing targets: **depth, segmentation, bit density**. Plus an optional variance target. Your implementation has:
- ✅ Bit density
- ✅ Depth
- ✅ Variance (uncertainty)
- ❌ **Segmentation** — not implemented

The plan says: "Can a linear head predict which object class each spatial token belongs to?" This would use SAM3 masks as supervision. This is arguably the most interesting target beyond depth (does the model "see" objects in a 1-bit measurement?). Its absence is a gap.

---

## Results Interpretation

### Spatial Bit Density (R² up to 0.99)

Near-perfect linear decodability from early timesteps. This makes sense — the ControlNet directly injects SPAD information via conditionings, so the model should encode the input measurement in its activations. The early-timestep high performance (R²=0.99 at t=0) confirms that measurement information is immediately available.

**Interesting pattern**: R² decreases at later timesteps (~0.88 at t=27). This suggests the model gradually replaces raw measurement information with higher-level scene understanding as denoising proceeds.

### Spatial Depth (R² up to 0.64)

Depth is linearly decodable from mid-to-late layers, peaking in the **single blocks** (single_9 at t=14: R²=0.64). This is a genuinely interesting finding:
- The model was **never given depth supervision**, yet encodes depth implicitly
- The peak is in mid-depth single blocks, not the final blocks — similar to AC3D's finding that camera knowledge peaks in middle layers
- Depth encoding improves from early to mid timesteps (t=0: R²=0.18 → t=14: R²=0.64) and then slightly decreases — the model "commits" to scene geometry during mid-denoising

### Spatial Variance (R² up to 0.41)

Moderate linear decodability of prediction uncertainty from a single pass. This is the weakest signal but still meaningful — the model partially encodes its own uncertainty in single-pass activations.

---

## Summary Table

| Component | Status | Severity | Faithfulness to AC3D |
|-----------|--------|----------|----|
| Ridge regression probe | ✅ Correct | — | Faithful |
| Per-block per-timestep design | ✅ Correct | — | Faithful |
| Heatmap + line plot visualization | ✅ Correct | — | Faithful |
| Global mean-pool probes: R² all negative | 🔴 Broken | Critical | AC3D uses more data + spatial |
| n=100 samples (should be 776+) | 🟡 Insufficient | Medium | AC3D uses 1000 |
| 10/57 blocks (should be all) | 🟡 Sparse | Medium | AC3D probes all |
| Depth model (DPT vs ml-depth-pro) | 🟡 Mismatch | Low | — |
| No feature normalization (spatial) | 🟡 Missing | Medium | — |
| No segmentation probing | 🟡 Missing target | Medium | Plan specifies it |
| Spatial probing results | 🟢 Valid & interesting | — | Properly executed |

---

## Recommendations (Priority Order)

### 1. Drop global probing or fix it (Critical for paper)
- **Option A**: Drop global, report only spatial — the spatial results are richer anyway
- **Option B**: Run with all 776 val samples + add PCA (reduce 3072 → 100 dims)
- **Option C**: Increase ridge λ substantially (try `lam=1.0`)

### 2. Probe all blocks (medium effort, high impact for paper figure)
- Re-run extraction with all 19 joint + 38 single blocks hooked
- The probing itself is cheap; extraction is the bottleneck
- This gives the smooth curve across layers that AC3D has in Figure 5

### 3. Add segmentation probing (per plan)
- Run SAM3 on GT images, extract masks
- Use per-token mask labels as classification target for spatial probing
- Report segmentation IoU vs layer/timestep

### 4. Switch to ml-depth-pro for depth targets
- Replace DPT-Hybrid with ml-depth-pro as specified in the plan
- Re-run target preparation

### 5. Add feature normalization to spatial probing
- Accumulate feature mean/std in a first pass, normalize during XTX accumulation

---

## Addendum: GPT Suggestions — Status & Judgement

### Answering the Question: Are Spatial Results Per-Token?

**Yes, the spatial probing is genuinely per-token (per-patch).** Here's the data flow:

1. **Activation extraction** (hooks at lines 82–99): Each DiT block outputs image tokens `[B, img_len, D]`. The hooks save the image tokens directly (joint blocks sep img from txt, single blocks slice off text prefix). Each token corresponds to a patch in the latent grid.
2. **Token ↔ spatial mapping**: FLUX processes 512px → 64px latent → 32×32 = **1024 patches** (each patch = 16×16 pixels). The code assumes `PATCH_H=32, PATCH_W=32` (line 45).
3. **Target preparation**: Depth/bit-density/variance maps are downsampled to 32×32 = 1024 values via `Image.resize((PATCH_W, PATCH_H), BILINEAR)` (lines 152–158).
4. **Probe**: A single linear weight `w ∈ ℝ^{D×1}` maps each token's D=3072 features to a scalar target. All 1024 tokens across all images share the same linear head — this is the standard AC3D approach for dense probing.

So the spatial results are **per-token ridge regression on the actual DiT patch grid** — exactly what GPT's suggestion #3 recommends.

---

### GPT Suggestion Assessment

| # | Suggestion | Already Done? | Worth Adding? | Verdict |
|---|-----------|--------------|---------------|---------|
| 1 | Use single forward passes at selected t, not full 50-step trajectory | ✅ Partially | 🟡 Refine | See below |
| 2 | Split targets into dense and global | ✅ Done | — | Already have both modes |
| 3 | Token-wise probes for dense tasks | ✅ Done | — | Spatial probing is per-token |
| 4 | Add control comparison (base FLUX vs +ControlNet vs +LoRA) | ❌ Not done | ✅ **Yes** | High impact |
| 5 | Probe both layer index and timestep (AC3D-style) | ✅ Done | — | Heatmap + lineplot exist |
| 6 | Add reconstruction residual / uncertainty proxy target | ✅ Done | 🟡 Refine | "variance" target exists |
| 7 | Frame bit density as "measurement evidence" | N/A (framing) | ✅ Yes | Paper narrative only |

---

### Detailed Assessment Per Suggestion

#### 1. "Don't make 50-step full trajectory the main setup"

**Status**: Already partially addressed — the code uses 28-step denoising (not 50), and hooks fire at only 7 selected steps: `[0, 4, 9, 14, 19, 24, 27]`.

**However**, the code still runs the **full denoising trajectory** (all 28 steps) and just hooks at selected points. GPT suggests doing independent single-forward-pass probes at selected noise levels instead (inject noise at level t, run one forward pass, capture activations).

**My judgement**: **Not worth changing.** The full trajectory approach is:
- What naturally happens during image generation
- Gives you activations conditioned on the cumulative denoising state (not just a random noisy latent)
- More realistic — you care about what the model knows during actual generation, not during a contrived single-pass
- AC3D also probes during generation, not isolated forward passes

The only downside is compute cost (28 steps instead of 7), but you've already extracted all 100 samples. This is a non-issue.

#### 2. "Split targets into dense and global"

**Status**: ✅ Already done. The code has both global (mean-pooled) and spatial (per-token) probing. Results are stored separately:
- `bit_density`, `depth`, `variance` → global scalar probes
- `spatial_bit_density`, `spatial_depth`, `spatial_variance` → dense per-token probes

No changes needed.

#### 3. "Token-wise probes for dense tasks"

**Status**: ✅ Already done, as confirmed above. Spatial probing uses per-token features `[1024, D]` with a shared linear head `D→1`. This is exactly what GPT recommends.

No changes needed.

#### 4. "Add control comparison (base FLUX vs +ControlNet vs +LoRA)" ⭐

**Status**: ❌ **Not done.** Only the final trained model (FLUX + ControlNet + LoRA) is probed.

**My judgement**: **This is the single most impactful suggestion.** Without a control comparison, you can't distinguish:
- What FLUX already knows from ImageNet-scale pretraining (depth, objects)
- What the ControlNet adds (SPAD measurement → scene understanding)
- What the LoRA adaptation adds (task-specific fine-tuning)

**Recommended comparisons** (in order of importance):
1. **Base FLUX** (no ControlNet, no LoRA) — run denoising with same seeds, hook activations. This shows the prior's built-in scene understanding.
2. **FLUX + ControlNet, no LoRA** — ControlNet is there but LoRA weights are zero. This shows what the pretrained ControlNet-Union already gives.
3. **FLUX + ControlNet + LoRA** (current) — the full trained model.

The delta between 1→2 shows "what ControlNet conditioning adds," and 2→3 shows "what SPAD-specific LoRA adds." This is directly analogous to AC3D's analysis that led to constraining ControlNet injection to early blocks.

**Implementation**: Minimal code change — just run `--extract` with different model loading configs. The probing and plotting infrastructure can be reused.

> [!IMPORTANT]
> This comparison is what turns "the model encodes depth" (expected) into "SPAD conditioning causes the model to encode depth *differently*" (novel finding). Without it, a reviewer will say "of course FLUX encodes depth, it was trained on millions of images."

#### 5. "Probe both layer index and timestep"

**Status**: ✅ Already done. The code probes 10 blocks × 7 timesteps and produces both heatmaps (block × timestep) and lineplots (R² vs block, one curve per timestep). This directly mirrors AC3D Figure 5.

No changes needed.

#### 6. "Add reconstruction residual / uncertainty proxy target"

**Status**: ✅ Already done (partially). The `variance` target uses per-pixel RGB variance across multi-seed outputs as a proxy for model uncertainty. The per-token spatial variance probing produces R² up to 0.41.

**Could be refined**: GPT suggests also probing reconstruction *error* (|generated - GT|), not just cross-seed variance. This would test whether the model internally represents "how wrong it is" about a region. Easy to add — just compute per-pixel error maps during target preparation.

**My judgement**: The variance target is sufficient for now. Adding error maps is a marginal improvement that could be done later.

#### 7. "Frame bit density as measurement evidence"

**Status**: N/A — this is a paper narrative suggestion, not a code change.

**My judgement**: ✅ Agree completely. "Measurement evidence" or "photon support" is much more compelling framing than "bit density." The interesting story is:
- Early layers: high measurement-evidence signal (model retains raw SPAD input)
- Middle layers: depth/structure emerges (model abstracts from measurement to scene)
- Late layers: appearance/texture dominates (measurement signal fades)

Your spatial results already show this pattern! Bit density R² decreases from 0.99→0.88 as denoising proceeds, while depth R² peaks mid-denoising.

---

### Missing from GPT's Suggestions: Segmentation

GPT correctly flags that SAM masks are **instance masks, not semantic labels**. They suggest:
- **Option A** (instance-agnostic): foreground/background, boundary maps, mask occupancy — very doable with SAM
- **Option B** (semantic): needs a separate semantic segmentation model

**My judgement**: Option A is the right path. A binary foreground/background probe per token is clean and interpretable. Use SAM3 masks, assign any-mask-present → foreground, no-mask → background. This is cheap to implement and adds a meaningful target.

### Bottom Line

The existing implementation is more faithful to AC3D than GPT gives it credit for — the spatial per-token probing and multi-timestep heatmaps are solid. The one high-impact addition is the **control comparison** (#4). Everything else is either already done or marginal improvement.
