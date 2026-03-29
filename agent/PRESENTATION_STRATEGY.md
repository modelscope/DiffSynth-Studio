# Presentation & Thesis Figure Strategy

**Updated**: 2026-03-29
**Deadlines**: Practice Mon Mar 31 | Final Wed Apr 2 | Thesis Mon Apr 7

---

## Heilmeier Catechism Framing

### 1. What are you trying to do?
Reconstruct full RGB images from single 1-bit SPAD binary frames using a 12B-parameter generative prior, and understand what the model internally learns from such extreme sparse inputs.

### 2. How is it done today, and what are the limits?
- Classical: Multi-frame accumulation + ISP pipeline (needs 100s-1000s of frames)
- Learning-based: SD1.5 ControlNet (smaller prior, weaker results), burst-based methods (gQIR, Quanta Diffusion)
- Limits: Single-frame reconstruction is massively ill-posed (1 bit → 24 bits per pixel). No prior work probes what generative models learn from binary sensor data.

### 3. What is new in your approach?
- **Largest generative model** applied to SPAD (12B FLUX.1-dev vs ~1B SD1.5)
- **First probing analysis** of a diffusion model conditioned on binary sensor data
- **Trustworthy benchmark**: scene-aware stratified split fixing data leakage
- **Physics-guided posterior sampling**: Bernoulli likelihood DPS at inference

### 4. Who cares?
- Computational imaging: enables low-power, high-speed SPAD cameras
- ML interpretability: uses SPAD as a clean testbed for probing generative priors
- Uncertainty quantification: single-sample uncertainty estimation from activations

### 5. If successful, what difference will it make?
- Shows large priors implicitly encode 3D geometry from 1-bit data (depth R²=0.685)
- Enables single-sample uncertainty estimation (variance R²=0.506) without multi-seed sampling
- Establishes the first leak-free SPAD→RGB reconstruction benchmark

### 6. What are the risks?
- Method novelty alone is insufficient for NeurIPS (ControlNet+LoRA is known)
- DPS guidance is modest (+0.16 PSNR); consistency training hurt performance
- Probing is the strongest card; without it, this is "solid engineering, not science"

### 7. How much will it cost / how long will it take?
- Training: ~17h per model on RTX 5090 (already done for RAW, consistency)
- Probing: ~3h extraction + 30min training (already done, 42GB activations saved)
- Remaining: OD training running, SD1.5 re-eval ready, paper writing is the bottleneck

### 8. What are the midterm and final "exams" to check for success?
- Midterm: Trustworthy benchmark + strong baseline + probing figures (all done)
- Final: NeurIPS submission with probing as core novelty, architecture ablation, clean story

---

## Figure Assignment to Presentation Slides

### Opening Hook (Slides 1-2)
**Figure**: `fig_hero_spad_to_rgb.{png,pdf}`
- Show the "wow factor": sparse binary noise → plausible RGB scene
- 6 diverse examples spanning dark to bright, indoor to outdoor
- **Key message**: "From 1 bit per pixel, a 12B model reconstructs plausible scenes"

### Problem Statement (Slide 3)
**Figure needed**: Architecture diagram (TikZ or drawn)
- SPAD → VAE → ControlNet → FLUX DiT → VAE → RGB
- Show LoRA placement on ControlNet
- Mark the probing hooks on DiT blocks

### Data Leakage Fix (Slide 4)
**Figure**: `fig_leakage_fix.{png,pdf}`
- Before: 94 train / 101 val / 94 leaked locations (red)
- After: 77 train / 20 val, zero leakage (green badge)
- **Key message**: "We found and fixed severe data leakage in the benchmark"

### Baseline Results (Slide 5)
**Figure**: `fig_baseline_summary_table.{png,pdf}`
- 10-seed mean±std for PSNR/SSIM/LPIPS/FID/CFID
- Color-coded rows for baseline, DPS, consistency, frame ablation

### Qualitative Comparison (Slide 6-7)
**Figure**: `fig_master_comparison_grid.{png,pdf}`
- All methods side-by-side: Input | Baseline | DPS | Consistency | Consist+DPS | Variance | GT
- 4-6 rows of diverse scenes
- The variance column shows uncertainty visually

### Frame Ablation (Slides 8-9)
**Figures**:
1. `fig_frame_ablation_strip.{png,pdf}` — visual progression
2. `fig_frame_ablation_curves.{png,pdf}` — metric curves
- **Key message**: "More frames improve perception (LPIPS↓) and consistency (CFID↓) but hurt pixel fidelity (PSNR↓) — perception-distortion-consistency tradeoff"

### DPS Physics Guidance (Slide 10)
**Figures**:
1. `fig_pairwise_baseline_vs_dps.{png,pdf}` — visual comparison
2. `fig_dps_ablation_bars.{png,pdf}` — quantitative eta sweep
- **Key message**: "Physics guidance provides modest improvement; ControlNet already internalizes the measurement information"

### Probing Motivation (Slide 11)
- Reference AC3D (Lindell's own paper): "We adapt their probing methodology to SPAD"
- Explain: hooks on 57 DiT blocks × 7 timesteps, ridge regression

### Probing Results — Main Heatmap (Slide 12)
**Figure**: `probing_analysis_output/fig1_main_heatmap.{png,pdf}`
- 3-panel heatmap: bit density (preserved everywhere), depth (emerges in single blocks), variance (crystallizes late)
- **Key message**: "The model preserves input (bit density R²=0.99), builds geometry (depth R²=0.685), and knows its own uncertainty (variance R²=0.506)"

### Probing Results — LoRA Delta (Slide 13)
**Figure**: `probing_analysis_output/fig3_delta_heatmap.{png,pdf}`
- R²(Main) - R²(Control): red = LoRA helps
- **Key message**: "LoRA teaches SPAD-to-geometry conversion (+0.232 R² for depth)"

### Probing Results — Information Flow (Slide 14)
**Figure**: `probing_analysis_output/fig4_best_timestep_lineplot.{png,pdf}`
- AC3D-style curve: R² vs block depth
- Main vs Control comparison
- **Key message**: "Information flows: conditioning → geometry → commitment"

### Object Recognition (Slide 15)
**Figure**: `probing_analysis_output/fig7_object_probing.{png,pdf}`
- 24 objects, Main vs Control balanced accuracy
- **Key message**: "The model recognizes specific objects from 1-bit data"

### Uncertainty (Slide 16)
**Figure**: `fig_variance_overlay.{png,pdf}`
- Reconstruction paired with variance heatmap
- **Key message**: "The model knows where it's guessing — uncertainty from activations without multi-seed sampling"

### Contributions Slide (Slide 17)
**Figure**: `fig_contribution_boundary.{png,pdf}`
- 8-row color-coded table separating Data Capture / Curation / Benchmark / Method / Physics / Consistency / Analysis / Evaluation
- **Key message**: Clear delineation of who did what

### Conclusion (Slide 18)
- **Headline**: Large generative priors encode geometry and uncertainty from 1-bit photon measurements
- **Three takeaways**:
  1. Trustworthy benchmark with leak-free split
  2. Probing reveals depth from binary data (R²=0.685)
  3. Model knows its own uncertainty (R²=0.506)

---

## Figures Status

| Figure | Priority | Status |
|--------|----------|--------|
| Hero figure (SPAD → Reconstruction → GT) | **Critical** | **DONE** — `fig_hero_spad_to_rgb` |
| Master comparison grid | **Critical** | **DONE** — `fig_master_comparison_grid` |
| Frame ablation strip (visual) | **Critical** | **DONE** — `fig_frame_ablation_strip` (fixed GT column) |
| Frame ablation curves (metrics) | **Critical** | **DONE** — `fig_frame_ablation_curves` |
| DPS ablation bars | **Critical** | **DONE** — `fig_dps_ablation_bars` |
| Variance overlay | **Critical** | **DONE** — `fig_variance_overlay` |
| Pairwise comparisons (4x) | **Critical** | **DONE** — `fig_pairwise_*` |
| Leakage fix visualization | **Critical** | **DONE** — `fig_leakage_fix` |
| Baseline summary table | **Critical** | **DONE** — `fig_baseline_summary_table` |
| Contribution boundary table | **Critical** | **DONE** — `fig_contribution_boundary` |
| Depth comparison montage | **High** | **DONE** — `fig_depth_comparison` |
| Segmentation comparison montage | **High** | **DONE** — `fig_segmentation_comparison` |
| Probing summary panel | **High** | **DONE** — `fig_probing_summary` |
| Architecture diagram (FLUX+CN+LoRA+probes) | **Critical** | NOT DONE — manual TikZ/draw.io |
| Spatial depth prediction viz (probing) | Medium | NOT DONE — data exists |
| SD1.5 vs FLUX comparison | Medium | NOT DONE — blocked on SD1.5 re-eval |
| OD filter results | Medium | NOT DONE — blocked on OD training |

---

## Presentation Do's and Don'ts

### Do
- Lead with the hero figure (SPAD → RGB "wow")
- Center the talk on probing (slides 12-16 are the scientific core)
- Show the leakage fix as a credibility strength
- Use the frame ablation as a secondary "interesting finding"
- Keep DPS brief — it's modest and that's fine

### Don't
- Don't spend more than 1 slide on DPS
- Don't show consistency training results prominently (negative result)
- Don't claim "first SPAD dataset" (gQIR has one too)
- Don't put too many numbers on slides — 1 table, rest are figures
- Don't mix presentation and paper stories — keep the talk simple

### Slide Budget (20 min)
- Hook + motivation: 3 slides (3 min)
- Method: 2 slides (3 min)
- Results: 4 slides (5 min)
- Probing (core): 5 slides (7 min)
- Discussion + conclusion: 2 slides (2 min)

---

## Thesis Writing Priority

1. **Section 5** (Probing): Write first, this is the strongest section
2. **Section 4** (Dataset/Benchmark): Write second, establishes credibility
3. **Section 3** (Method): Fairly standard, can be written quickly
4. **Section 6** (Results): Tables and figures already exist
5. **Section 1-2** (Intro/Related Work): Write last, easier once results are clear
