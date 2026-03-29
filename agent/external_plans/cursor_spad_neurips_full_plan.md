---
name: SPAD NeurIPS Full Plan
overview: Complete research plan for SPAD-to-RGB reconstruction NeurIPS paper. Amalgamated from both plan files. Includes data infrastructure overhaul (scene-aware stratified train-test split, dataset audit, retraining), all experimental phases, paper positioning, and execution details.
todos:
  - id: dataset-audit
    content: "Phase 0a: Write audit_dataset.py -- scan all bits_* and RGB folders, build presence matrix, identify common subset, exclude 5 malformed scene IDs, output dataset_inventory.json"
    status: completed
  - id: scene-aware-split
    content: "Phase 0b: Write prepare_dataset_scene_aware.py -- stratified split by location (indoor/outdoor balanced), generate metadata_{train,val}.csv for EVERY bits_* folder using the same split"
    status: completed
  - id: retrain-flux-raw
    content: "Phase 0c: Retrain FLUX ControlNet LoRA on RAW_empty (bits/) from scratch with corrected stratified split -- 40 epochs completed, checkpoints at models/train/FLUX-SPAD-LoRA-SceneAware-RAW/"
    status: completed
  - id: reeval-sd15
    content: "Phase 0d: Re-run SD1.5 inference + metrics on the corrected scene-aware val set for fair comparison"
    status: pending
  - id: metrics-baseline
    content: "Phase 1a: Run validate_lora.py + run_metrics.py on retrained FLUX val outputs (seed 42), compare against SD1.5 on same corrected val set"
    status: pending
  - id: multi-seed-gen
    content: "Phase 1b: Generate K=10 seeds via validate_lora.py on corrected val set, run full metrics pipeline on each"
    status: pending
  - id: aggregate-script
    content: "Phase 1c: Run aggregate_metrics.py for cross-seed mean/std computation and paper tables"
    status: pending
  - id: variance-maps
    content: "Phase 2a: Per-pixel RGB variance maps across K=10 seeds, correlation with bit density and depth"
    status: pending
  - id: frame-vs-seed
    content: "Phase 2b: Frame-vs-seed variance decomposition using 7 bits_frame_* folders on common subset views"
    status: pending
  - id: downstream-stability
    content: "Phase 2c: Segmentation entropy maps + depth variance across seeds for downstream task stability"
    status: pending
  - id: linear-probing
    content: "Phase 2d: Train linear probes on FLUX DiT activations per layer/timestep (AC3D-inspired key experiment)"
    status: pending
  - id: diffusion-steps
    content: "Phase 2e: Save/decode intermediate latents at selected denoising steps, analyze convergence"
    status: pending
  - id: calibration
    content: "Phase 2f: Build calibration analysis -- empirical confidence intervals vs GT coverage rates"
    status: pending
  - id: spad-forward
    content: "Phase 3a: Implement differentiable SPAD forward model (Bernoulli likelihood) -- code written at diffsynth/diffusion/spad_forward.py"
    status: completed
  - id: dps-guidance
    content: "Phase 3b: Implement FlowDPS inference-time guidance in FluxImagePipeline -- code written at diffsynth/diffusion/flow_dps.py"
    status: completed
  - id: frame-consistency-loss
    content: "Phase 3c: Implement per-frame consistency loss in train_lora.py -- code written at diffsynth/diffusion/consistency_loss.py"
    status: completed
  - id: physics-ablation
    content: "Phase 3e: Ablation matrix -- baseline vs FlowDPS vs consistency loss vs combined -- script written at run_physics_ablation.sh"
    status: pending
  - id: finetune-od3
    content: "Phase 4a-i: Fine-tune the RAW_empty checkpoint on OD_03 data (bits_RAW_OD_03/), separate output folder"
    status: pending
  - id: train-od3-scratch
    content: "Phase 4a-ii: Train a separate model from scratch on OD_03 only (bits_RAW_OD_03/), for comparison"
    status: pending
  - id: frame-od-ablation
    content: "Phase 4a: OD filter + frame-count ablations using new bits_* folders on corrected val set -- scripts written"
    status: pending
  - id: paper-writing
    content: "Phase 5: Write NeurIPS paper (9 pages), begin drafting intro/method by Week 3 -- initial draft at paper/main.tex"
    status: in_progress
isProject: false
---

# SPAD-to-RGB Reconstruction via Diffusion Priors: NeurIPS Paper Plan

## Current State Assessment

**What exists** (from thesis interim report and repo):

- Working FLUX.1-dev ControlNet + LoRA pipeline ([DiffSynth-Studio-SPAD](DiffSynth-Studio-SPAD/)) with LoRA-on-ControlNet (the better placement per your finding)
- SD1.5 ControlNet baseline ([spad-diffusion](spad-diffusion/)) with multi-stage training (10-frame -> 1-frame -> OD3)
- Dataset: ~2,500 views, 512x512, up to 20k binary frames/view, R/G/B + OD filters
- GT pipeline with log-inversion flux estimation (Eq. 2.5), hot-pixel suppression, white balance, gamma
- Validation outputs for 3 seeds (42, 67, 88) with depth (ml-depth-pro) and segmentation (SAM3) analysis
- Existing metrics from Table 3.1: SD1.5 best = PSNR 16.81 / SSIM 0.540 / LPIPS 0.360 / FID 58.88; FLUX = PSNR 17.44 / SSIM 0.613 / LPIPS 0.408 / FID 70.49
- Montages comparing GT vs generated for both depth and segmentation

**What is missing**:

- `run_metrics.py` (PSNR/SSIM/LPIPS/FID) has NOT been run on FLUX validation directories (only SD1.5 has metrics.txt)
- Only 3 seeds -- insufficient for the distributional analysis described in Eq. 4.1-4.4 of the report
- The variance decomposition (Eq. 4.4: measurement-driven vs seed-driven) is designed but not implemented
- Segmentation entropy maps (Eq. 4.5-4.6) are designed but not implemented
- No CFID implementation
- No physics consistency loss or DPS guidance -- the Bayesian formulation (Eq. 2.8-2.9) is theoretical only
- No forward-model simulation/augmentation (Eq. 4.7)
- No OOD stress testing across filter conditions
- No intermediate diffusion step visualization for FLUX

**Key competitor landscape**:

- gQIR (arxiv 2602.20417) -- latent diffusion + Bernoulli statistics for color SPAD bursts; targets photon-starved regime with synthetic+real data
- Quanta Diffusion (Chennuri et al., 2025) -- embeds SPAD/QIS forward model in diffusion loop; multi-frame focus
- bit2bit (NeurIPS 2024) -- self-supervised 1-bit reconstruction via Bernoulli lattice process

---

## Critical Discovery: Data Leakage in Current Split

The current train-test split uses `sklearn.train_test_split` at the **view level** with `random_state=42`. This is a multi-view dataset with 2631 views across 101 physical locations (sessions). The current split puts views from **94 out of 101 locations in BOTH train and val**, inflating all reported metrics.

Example: `0724-dgp` (the DGP lab) has 224 views in train and 55 in val. The model has seen this room 224 times during training, then is "tested" on 55 other angles of the same room.

**This must be fixed before any metrics are meaningful.**

**Resolution**: Implemented stratified scene-aware split (Phase 0b). Indoor and outdoor locations are split independently at 20% val rate. Final split: 77 train locations (1850 views), 20 val locations (776 views). 14 indoor + 6 outdoor in val. Zero location leakage.

---

## Paper Positioning and Framing

### Why the current work is NOT yet NeurIPS-worthy (honest assessment)

Training ControlNet + LoRA on FLUX for SPAD-to-RGB is solid engineering, but by itself it is an application of existing methods (ControlNet, LoRA, FLUX) to a new domain. NeurIPS reviewers would say "the method is well-known, and the novelty is only the application domain." The LoRA-on-ControlNet placement finding is interesting but incremental.

### What elevates it: Three complementary pillars

For NeurIPS, the paper needs to make reviewers say "I learned something new about how generative models work, how to evaluate them, or how physics interacts with learned priors."

**Pillar 1 -- "What does the model know?" (Probing Analysis, AC3D-inspired)**

This is what Prof. Lindell clearly wants and is the most likely to produce a surprising finding. Precedents:

- AC3D (Bahmani, ..., **Lindell**, Tulyakov, 2024): probed video DiT for camera pose -- found it peaks in middle layers
- "Beyond Surface Statistics" (Chen et al., NeurIPS 2023 workshop): probed Stable Diffusion for depth/saliency via linear probes

Nobody has probed a diffusion model conditioned on SPAD measurements. Novel questions:

- Does the model encode depth even though it was never given depth supervision?
- Can you predict from *single-seed activations* whether a region will be high-variance across seeds? (internal uncertainty signal without multi-seed sampling)

**Pillar 2 -- Uncertainty-Hallucination-Consistency Analysis**

"Looks Too Good To Be True" (Cohen et al., NeurIPS 2024) proved that perfect perceptual quality requires >= 2x the inherent uncertainty. Single-frame SPAD is a *perfect test case*:

- 1 bit per pixel = enormous inherent uncertainty, quantifiable via multi-seed sampling
- Measure whether hallucinations (high-variance regions) correlate with downstream task failures
- Plot a novel three-way tradeoff: **distortion vs. perception vs. measurement consistency**

**Pillar 3 -- Physics-Consistent Posterior Sampling (FlowDPS + Bernoulli Likelihood)**

The physics step from your slide = DPS guidance = gradient ascent on log p(y|x) at each denoising step. Since FLUX uses rectified flow, use **FlowDPS** (Kim et al., ICCV 2025) rather than vanilla DPS. FlowDPS adapts posterior sampling for flow-matching models via a flow-version of Tweedie's formula.

- ControlNet provides *implicit* measurement conditioning (learned mapping)
- FlowDPS provides *explicit* conditioning (physics gradient of Bernoulli likelihood)
- Comparing these two information pathways is methodologically interesting

### Combined paper framing (the strongest story)

> "We study generative reconstruction from single-photon measurements as a testbed for understanding how large diffusion priors interact with extreme inverse problems. Using a 12B-parameter rectified-flow transformer conditioned on SPAD binary frames, we: (1) probe the model's internal representations to show it implicitly encodes scene geometry despite never receiving depth supervision, (2) characterize the uncertainty-hallucination tradeoff in the single-photon regime where the measurement provides minimal constraint, and (3) introduce physics-consistent posterior sampling that enforces the Bernoulli photon-detection likelihood at inference time."

This framing is not "we applied ControlNet to SPAD" -- it is "we used SPAD as a lens to understand generative priors."

### Additional ideas to strengthen the paper

1. **Internal uncertainty prediction**: Train a linear probe on single-seed intermediate activations to predict per-pixel variance (computed from K=10 seeds). If this works, you have a way to estimate uncertainty without expensive multi-seed sampling -- a genuine contribution to UQ for generative models.
2. **Perception-distortion-consistency triangle**: Three evaluation axes -- distortion (PSNR/SSIM), perception (LPIPS/FID), measurement consistency (NLL under forward model). The physics guidance should improve consistency at the cost of distortion or perception. Plotting this three-way tradeoff is novel.
3. **Uncertainty vs. measurement information**: Plot total output variance as a function of SPAD bit density. As bit density increases (more photons), uncertainty should decrease monotonically. Validates that the model correctly uses measurement information.
4. **ControlNet vs. FlowDPS as information pathways**: ControlNet injects measurement *implicitly* (learned). FlowDPS injects it *explicitly* (physics gradient). Does FlowDPS reduce multi-seed variance? Does it reduce hallucination? This comparison is methodologically interesting.
5. **Cross-architecture probing (SD1.5 vs FLUX)**: Probe both architectures for the same properties. Does the larger model encode depth more accurately? Does it produce more calibrated uncertainty? This leverages your existing SD1.5 baseline for analysis.

### Differentiation from competitors

- **vs. gQIR**: They use burst data (multiple frames) + color SPADs + synthetic-heavy training. You: single 1-bit frame (much harder), real large-scale dataset (2,500 views), distributional + probing analysis, downstream tasks.
- **vs. Quanta Diffusion**: They embed forward model in diffusion loop for multi-frame video. You: inference-time guidance on a frozen large-scale prior, analysis-first approach.
- **vs. bit2bit**: Self-supervised video reconstruction. You: generative prior approach with uncertainty quantification and interpretability.

### Key references to cite (beyond those in your report)

- **FlowDPS** (Kim et al., ICCV 2025, arxiv 2503.08136) -- DPS for rectified flow models, exactly your setup for the physics step
- FlowChef (Patel et al., ICCV 2025) -- gradient-free steering of rectified flow, alternative approach
- "Beyond Surface Statistics" (Chen et al., NeurIPS 2023 workshop) -- linear probing of diffusion internals for depth/saliency
- "Looks Too Good To Be True" (Cohen et al., NeurIPS 2024) -- uncertainty-perception tradeoff theory
- Traversing Distortion-Perception Tradeoff (Wang et al., CVPR 2025) -- navigating the tradeoff at inference time
- Intrinsic LoRA (Feng et al., 2024) -- extracting scene intrinsics from generative models
- gQIR (arxiv 2602.20417) -- direct SPAD reconstruction competitor
- PaDIS-MRI (arxiv 2509.21531) -- patch-based diffusion + physics for inverse problems
- PIRF (NeurIPS 2025, arxiv 2509.20570) -- physics-informed reward fine-tuning

---

## Phase 0: Data Infrastructure Overhaul (BLOCKING -- do first)

### 0a. Dataset Audit ✅ COMPLETED

**RGB/ is the source of truth for pairing.** Any SPAD image without a matching RGB ground truth is dropped. Cross-check all `bits_`* folders against `RGB/` to build a master view inventory.

**Available folders** (from extraction log `extraction_log_20260315_190624.txt`):


| Folder                     | Filter    | Config           | Count | Notes        |
| -------------------------- | --------- | ---------------- | ----- | ------------ |
| `bits/`                    | RAW_empty | frame 0, N=1     | 2637  | Baseline     |
| `bits_frame_1000/`         | RAW_empty | frame 1000, N=1  | 2632  | -5 views     |
| `bits_frame_4000/`         | RAW_empty | frame 4000, N=1  | 2631  | -6 views     |
| `bits_frame_5000/`         | RAW_empty | frame 5000, N=1  | ~2630 | Pre-existing |
| `bits_frame_8000/`         | RAW_empty | frame 8000, N=1  | 2630  | -7 views     |
| `bits_frame_12000/`        | RAW_empty | frame 12000, N=1 | 2630  | -7 views     |
| `bits_frame_16000/`        | RAW_empty | frame 16000, N=1 | 2625  | -12 views    |
| `bits_multi_4/`            | RAW_empty | frame 0, N=4     | 2633  |              |
| `bits_multi_16/`           | RAW_empty | frame 0, N=16    | 2633  |              |
| `bits_multi_64/`           | RAW_empty | frame 0, N=64    | 2633  |              |
| `bits_multi_256/`          | RAW_empty | frame 0, N=256   | 2633  |              |
| `bits_multi_1000/`         | RAW_empty | frame 0, N=1000  | 2633  |              |
| `bits_RAW_OD_01/`          | OD 0.1    | frame 5000, N=1  | 2633  |              |
| `bits_RAW_OD_03/`          | OD 0.3    | frame 5000, N=1  | 2633  |              |
| `bits_RAW_OD_07/`          | OD 0.7    | frame 5000, N=1  | 2631  | -2 views     |
| `bits_RAW_OD_01_multi_16/` | OD 0.1    | frame 0, N=16    | 2633  |              |
| `bits_RAW_OD_01_multi_64/` | OD 0.1    | frame 0, N=64    | 2633  |              |
| `bits_RAW_OD_03_multi_16/` | OD 0.3    | frame 0, N=16    | 2633  |              |
| `bits_RAW_OD_03_multi_64/` | OD 0.3    | frame 0, N=64    | 2633  |              |
| `bits_RAW_OD_07_multi_16/` | OD 0.7    | frame 0, N=16    | 2633  |              |
| `bits_RAW_OD_07_multi_64/` | OD 0.7    | frame 0, N=64    | 2633  |              |


**Ignore** `bits_od3/` (2966 images, inconsistent with other folders, likely from older extraction).

**Missing views** (from extraction log warnings):

- `0731-sfbcor07-14`: Only 3 frames total (broken capture) -- missing from most frame-index folders
- `0807-outdoor-bahen-vehicle04-06`: 3197 frames -- missing from frame_4000+
- `0814-outdoor-galbraith-end05-11`: 4400 frames -- missing from frame_8000+
- Several views with ~13-14k frames: missing from frame_16000 only
- `0801-bahcor-cor02-10` and `0802-ba7252-desk01-26`: OD_07 only 3 frames

**Exclude 5 malformed scene IDs** entirely: `0814-outdoo9`, `0814-outdoo983`, `0814-outdoor`, `0814-outdoor-galbraith`, `0814-outdoor-galbraith-end05` (no view index suffix, likely corrupted captures).

**Action**: `audit_dataset.py` written and run. Output: `dataset_inventory.json` (2626 RGB scenes, 2616 common subset).

### 0b. Implement Scene-Aware Train-Test Split ✅ COMPLETED

**Problem**: Current split in [prepare_dataset.py](DiffSynth-Studio-SPAD/prepare_dataset.py) uses `train_test_split(metadata, test_size=0.2, random_state=42)` -- random at view level, no session grouping. Result: 94 out of 101 locations leak into both train and val.

**Fix**: `prepare_dataset_scene_aware.py` written and run. Uses **stratified** splitting:

1. Parses session prefix from each scene ID using `re.match(r'^(.+)-(\d{1,3})$', sid)` (strips last view index)
2. Classifies each location as indoor or outdoor from its name
3. Splits indoor and outdoor locations **independently** at the target 20% rate
4. All views from a given location go to either train OR val, never both
5. Generates `metadata_train.csv` and `metadata_val.csv` for **every** `bits`_* folder, all using the **same location-level split**
6. Generates `split_manifest.json` documenting which locations are train vs val
7. **Verifies SPAD-to-RGB pairing** for every entry

**Result**: 77 train locations (1850 views), 20 val locations (776 views). 14 indoor + 6 outdoor in val. Zero location leakage.

**Old random split backed up** to `spad_dataset/old_random_split/`.

### 0c. Retrain FLUX with Corrected Split (RAW_empty baseline) ✅ COMPLETED

Used [train_controlnet.sh](DiffSynth-Studio-SPAD/train_controlnet.sh) via `train_scene_aware_raw.sh`. Key settings:

- **Input**: `bits/` (RAW_empty, single frame at index 0) paired with `RGB/`
- **Metadata**: New scene-aware stratified `metadata_train.csv` and `metadata_val.csv`
- **Output path**: `./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/`
- **From scratch**: No checkpoint resume (clean start to avoid leaky-split contamination)
- **Hyperparams**: 40 epochs, lr=1e-4, LoRA rank 32, LoRA on ControlNet
- **Result**: All 40 epoch checkpoints saved (epoch-0.safetensors through epoch-39.safetensors)

This checkpoint is the base for fine-tuning on other exposure conditions (OD_03, OD_07) in Phase 4.

### 0d. Re-evaluate SD1.5 on Corrected Val Set ⏳ PENDING

- Run SD1.5 inference on the new val set views using `spad-diffusion/run_sd15_scene_aware_eval.sh`
- Checkpoint: `lightning_logs/spad_controlnet/two_stage_best/best-epoch=14-val_loss=0.1057.ckpt`
- Val data: `spad_dataset/val_only/bits` and `val_only/RGB` (776 symlinked images each)
- Run metrics pipeline (PSNR/SSIM/LPIPS/FID, depth, segmentation) on SD1.5 outputs
- This ensures FLUX vs SD1.5 comparison is on identical, leak-free val views

---

## Phase 1: Fill Metric Gaps and Multi-Seed Generation (Week 1)

These experiments are prerequisites for everything else. They are blocking.

### 1a. Run standard image metrics on all existing FLUX validation outputs ⏳ PENDING

Run `validate_lora.py` with seed 42 using the retrained FLUX checkpoint on the corrected val set, then run [run_metrics.py](DiffSynth-Studio-SPAD/run_metrics.py) on the outputs. This produces PSNR, SSIM, LPIPS, FID for FLUX to directly compare against SD1.5 from Table 3.1 (best SD1.5: PSNR 16.81, SSIM 0.540, LPIPS 0.360, FID 58.88).

### 1b. Generate K=10 seeds for distributional analysis ⏳ PENDING

Run `validate_lora.py` with 10 seeds (0, 13, 23, 42, 55, 67, 77, 88, 99, 123) on the corrected val set. For each seed:

- Generate all validation outputs
- Run `run_metrics.py`, `analysis_depth.py`, `sam3_eval.py`

This gives 10 complete evaluation runs -- sufficient for the distributional analysis in Sections 4.1. Script: [run_multiseed_validation.sh](DiffSynth-Studio-SPAD/run_multiseed_validation.sh).

### 1c. Write `aggregate_metrics.py` ✅ SCRIPT WRITTEN

Script written at [aggregate_metrics.py](DiffSynth-Studio-SPAD/aggregate_metrics.py). It should:

- Read all `depth_metrics_50.csv`, `sam3_metrics_50.csv`, and `metrics.txt` across all seed dirs
- Compute per-image statistics across seeds: mean, std, min, max for each metric
- Compute per-image pixel-level variance (needed for Eq. 4.3)
- Output: `aggregate_results.csv` (paper Table 1), per-image variance summaries
- Also compute the frame-vs-seed decomposition if multiple SPAD frames per view are tested (Eq. 4.4)

---

## Phase 2: Distributional and Downstream Analysis (Week 2-3)

This implements the core analysis from your thesis Section 4.1, which is the strongest part of the paper narrative regardless of whether the physics step works. All scripts written, waiting for multi-seed outputs.

### 2a. Per-pixel variance maps (Eq. 4.2-4.3) -- script: [compute_variance_maps.py](DiffSynth-Studio-SPAD/compute_variance_maps.py)

For each of the 50 validation images, compute per-pixel mean and variance across all K=10 seed outputs. Create:

- Variance heatmaps (per color channel and luminance) overlaid on the mean reconstruction
- Scatter plots: pixel variance vs. SPAD bit density (do dark regions have more uncertainty?)
- Scatter plots: pixel variance vs. GT depth (are distant objects more uncertain?)
- Aggregate: mean variance per image vs. overall bit density of the SPAD input

This directly implements Eq. 4.2-4.3 from the thesis and produces Figure-ready panels.

### 2b. Frame-vs-seed variance decomposition (Eq. 4.4) -- script: [frame_vs_seed_variance.py](DiffSynth-Studio-SPAD/frame_vs_seed_variance.py)

This is the strongest analytical contribution for the uncertainty story. For selected views where multiple SPAD frames exist:

- Sample F=5 different 1-bit frames from the same view
- Generate K=10 reconstructions per frame
- Decompose total variance into measurement-driven (across frames) and sampling/seed-driven (across seeds) components per Eq. 4.4
- Report which component dominates -- this tells us whether the bottleneck is photon randomness or model ambiguity

Uses the 7 `bits_frame`_* folders (frame 0, 1000, 4000, 5000, 8000, 12000, 16000). Only use views present in ALL 7 frame folders (the "common subset" from Phase 0a audit).

### 2c. Downstream task stability (Eq. 4.5-4.6) -- script: [downstream_stability.py](DiffSynth-Studio-SPAD/downstream_stability.py)

**Segmentation stability**: For each view, run SAM3 on all K=10 reconstructions. Compute:

- Per-pixel mask probability (Eq. 4.5) and entropy maps (Eq. 4.6)
- IoU variance across seeds for each detected object
- Correlation between RGB variance maps and segmentation entropy maps
- Does high-entropy segmentation spatially overlap with high-variance RGB?

**Depth stability**: For each view, run ml-depth-pro on all K=10 reconstructions. Compute:

- Per-pixel depth variance across seeds
- Correlation between depth variance and RGB variance
- Depth consistency metric: do all seeds agree on relative depth ordering?

### 2d. Linear probing of FLUX internal representations (AC3D-inspired) -- script: [linear_probing.py](DiffSynth-Studio-SPAD/linear_probing.py)

**This is a key experiment Prof. Lindell wants.** The idea comes directly from AC3D (Bahmani, Skorokhodov, Qian, Siarohin, Menapace, Tagliasacchi, **Lindell**, Tulyakov, 2024) Section 3.4, where they train linear probes on the internal activations of a video DiT to discover that the model implicitly performs camera pose estimation, and that this knowledge peaks in specific middle layers.

**The analogy for your SPAD work**: Does the FLUX DiT, when conditioned on a SPAD bit-plane via ControlNet, implicitly encode scene properties (depth, segmentation, object identity) in its intermediate layer activations? And at which layers and denoising steps does this information emerge?

**Concrete experiment**:

1. **Extract activations**: For each validation image, run the FLUX denoising process (50 steps) and save the intermediate activations from each of the DiT's transformer blocks at selected timesteps (e.g., t=0.9, 0.7, 0.5, 0.3, 0.1). The FLUX DiT has multiple blocks -- save the output of each block.
2. **Train linear probes**: For each (block, timestep) combination, train a lightweight linear regression/classifier to predict:
  - **Depth**: Can a linear head on the activations predict the GT depth map? (Use ml-depth-pro GT depth as supervision)
  - **Segmentation labels**: Can a linear head predict which object class each spatial token belongs to? (Use SAM3 GT masks as supervision)
  - **SPAD bit density**: Can the model's internals predict how many photons the input SPAD frame received per region?
3. **Analysis** (produces a key figure like AC3D's Figure 5):
  - Plot probe accuracy (e.g., depth Pearson r, segmentation IoU) vs. layer index for each timestep
  - Identify WHERE and WHEN scene understanding emerges in the network
  - Does depth knowledge peak in early/middle/late layers? Does it change across timesteps?
  - Does the ControlNet injection point correlate with where SPAD-specific information appears?
4. **Implications**:
  - If depth/segmentation is linearly decodable from intermediate activations, it means the model "understands" the scene at those layers -- not just producing pixels
  - This provides evidence that generative reconstruction preserves semantic structure, supporting downstream task utility (O3)
  - It also informs whether the ControlNet LoRA is placed optimally -- if SPAD information only affects certain layers, maybe LoRA should target those specifically
  - If probes at early timesteps already encode depth/segmentation, it means the model commits to scene layout early (like AC3D found for camera pose)

This experiment is relatively cheap (linear probes are fast to train) and produces a highly publishable figure. It directly connects your work to an analysis methodology from your supervisor's own paper.

### 2e. Latent-space analysis at diffusion steps -- script: [save_intermediate_latents.py](DiffSynth-Studio-SPAD/save_intermediate_latents.py)

Modify `FluxImagePipeline.__call__()` in [DiffSynth-Studio-SPAD](DiffSynth-Studio-SPAD/) to save intermediate latents at steps [1, 5, 10, 20, 30, 40, 50]. Decode each. Analyze:

- At which step does structure (layout, large objects) lock in vs. fine detail (texture, color)?
- LPIPS between consecutive decoded steps -- convergence rate
- Compare two different seeds at the same step -- when does divergence appear?
- **Combine with probing results**: correlate when probes become accurate with when decoded images become recognizable

### 2f. Calibration analysis -- script: [calibration_analysis.py](DiffSynth-Studio-SPAD/calibration_analysis.py)

- For each pixel, compute the empirical [5%, 95%] interval from K=10 seeds
- Check what fraction of GT pixels fall within those intervals (should be ~90% if calibrated)
- Plot a calibration curve: nominal coverage vs. empirical coverage
- Correlate calibration quality with SPAD bit density (are high-signal regions better calibrated?)

---

## Phase 3: Physics-Consistent Posterior Sampling (Week 3-4)

This is where the technical contribution lives. Your thesis already has the math (Eq. 2.8-2.9); what's missing is the implementation and evaluation. Prof. Lindell flagged this may or may not work well -- so treat it as an experiment with contingency plans.

### 3a. Implement the differentiable SPAD forward model ✅ CODE WRITTEN

Code at [diffsynth/diffusion/spad_forward.py](DiffSynth-Studio-SPAD/diffsynth/diffusion/spad_forward.py).

Using your thesis Eq. 2.2-2.3 and Eq. 2.9, implement a differentiable forward model in PyTorch:

```python
def spad_log_likelihood(x_rgb, y_spad, alpha):
    """
    x_rgb: generated RGB image (after inverse gamma to get linear)
    y_spad: observed 1-bit SPAD frame {0,1}^(H x W)
    alpha: intensity scaling parameter (learnable or fixed)
    
    H_i(x) = alpha * lin(x)_i
    p_i = 1 - exp(-H_i)
    log p(y|x) = sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    """
```

Key implementation details:

- The mapping from sRGB output to linear intensity requires inverse gamma (you have the Linear16 representation in the GT pipeline for reference)
- The `alpha` parameter absorbs quantum efficiency, exposure time, etc. (Eq. 2.3) -- estimate from data by fitting to GT pairs, or treat as a hyperparameter
- Numerical stability: clamp H_i to avoid log(0) at saturation

### 3b. FlowDPS inference-time guidance ✅ CODE WRITTEN

Code at [diffsynth/diffusion/flow_dps.py](DiffSynth-Studio-SPAD/diffsynth/diffusion/flow_dps.py).

**Use FlowDPS (Kim et al., ICCV 2025, arxiv 2503.08136) not vanilla DPS.** Since FLUX uses rectified flow, the original DPS (designed for DDPM) doesn't directly apply. FlowDPS decomposes the flow ODE using a flow-version of Tweedie's formula into clean image estimation + noise estimation components, and integrates likelihood gradients into the clean component. This is exactly the same physics step you described in your slide -- taking a gradient step on log p(y|x) -- but properly formulated for the rectified-flow framework.

At each denoising step t in the FLUX rectified-flow sampling:

1. Estimate clean image x_hat via the flow-Tweedie formula from the current latent z_t
2. Compute grad_z log p(y|x_hat) by backpropagating through the VAE decoder and the SPAD Bernoulli likelihood (Eq. 2.9)
3. Inject the gradient into the flow ODE per the FlowDPS formulation

Design choices to sweep:

- `eta(t)` schedule: constant, linearly increasing, or cosine
- Apply guidance at all steps vs. only last N steps (AC3D found camera info locks in during the first 10% of denoising -- test if SPAD structure similarly locks in early)
- Whether to backprop through the full VAE decoder or use a lightweight approximation
- FlowDPS supports stochastic injection -- test deterministic vs. stochastic variants

Inference script: [validate_dps.py](DiffSynth-Studio-SPAD/validate_dps.py).

### 3c. Per-frame consistency loss (Benjamin Attal idea, IC-Light inspired) ✅ CODE WRITTEN

Code at [diffsynth/diffusion/consistency_loss.py](DiffSynth-Studio-SPAD/diffsynth/diffusion/consistency_loss.py).

**Core insight**: For the same scene, you have many binary SPAD frames (different Bernoulli realizations of the same clean image). Two different binary frames F1 and F2 should produce the same denoised output. This can be enforced directly on the predicted noise during training.

**How it works**:

1. During training, for a given scene, sample two different binary SPAD frames y_F1 and y_F2 from the same view (they are different photon realizations of the same underlying flux)
2. Create the same noisy latent z_t from the same clean image at the same timestep t
3. Run the model conditioned on y_F1 to get predicted noise eps_F1
4. Run the model conditioned on y_F2 to get predicted noise eps_F2
5. Since both should denoise to the same clean image, enforce: **L_consistency = ||eps_F1 - eps_F2||**

```
L_total = L_flow_match + lambda_c * ||eps_F1 - eps_F2||
```

**Why this is powerful**:

- It directly teaches the model that different binary frames from the same scene are equivalent -- the model learns the *invariance* to photon randomness
- It doesn't require the model to match a specific GT pixel-for-pixel; it only requires self-consistency
- Your dataset is perfect for this: each view has up to 20,000 binary frames, providing unlimited pairs of different realizations
- This is analogous to IC-Light (Zhang, Rao, Agrawala, ICLR 2025), which enforces light transport consistency by relating predicted noise under two different illumination conditions. They showed this prevents the model from becoming a "structure-guided random image generator" and enables stable training at scale (>10M samples on SDXL/Flux)

**Key difference from IC-Light**: IC-Light's consistency comes from physics of linear light transport (blending illuminations). Your consistency comes from the fact that different Bernoulli samples of the same Poisson intensity should produce the same reconstruction. Both use the same noise-prediction consistency mechanism.

**Implementation notes**:

- Modify the dataloader to return pairs of binary frames for each scene
- Run two forward passes through the ControlNet (one per frame) but share the same noisy latent z_t and timestep t
- Add the consistency loss to the existing flow-matching loss
- This approximately doubles the per-step compute but provides a strong self-supervisory signal

### 3d. Training-time physics loss (optional)

Add a physics term to the flow-matching loss in [train_lora.py](DiffSynth-Studio-SPAD/train_lora.py):

```
L_total = L_flow_match + lambda * L_physics
L_physics = -log p(y | D(z_predicted))
```

where D is the VAE decoder and z_predicted is the flow-match clean prediction. This is more expensive (requires decoding during training) but could learn to be physics-consistent rather than relying on inference-time correction.

### 3e. Ablation matrix -- script: [run_physics_ablation.sh](DiffSynth-Studio-SPAD/run_physics_ablation.sh) ⏳ PENDING

Run the following configurations on the validation set (K=3 seeds each for speed):

- Baseline: ControlNet LoRA only (current)
- +FlowDPS guidance (constant eta, all steps)
- +FlowDPS guidance (late-only, last 10 steps)
- +FlowDPS guidance (tuned eta schedule)
- +Per-frame consistency loss (retrained)
- +Per-frame consistency loss + FlowDPS guidance (combined)
- +Physics training loss (if time permits)

Report: PSNR, SSIM, LPIPS, FID, depth MAE, segmentation mIoU, and a new metric: **measurement consistency** -- how well does the generated image explain the observed SPAD frame under the forward model?

Also measure: **cross-frame consistency** -- given two different binary frames from the same scene, how similar are the reconstructions? The per-frame consistency loss should directly improve this metric.

---

## Phase 4: Ablations, Simulation, and Paper Completeness (Week 4-5)

### 4a. OD filter and frame combination ablations

Your dataset has no-filter, OD1, OD3, OD7 settings and you've already trained SD1.5 on OD3 (Table 3.1).

**4a-i. Fine-tune from RAW_empty checkpoint** -- script: [train_od03_finetune.sh](DiffSynth-Studio-SPAD/train_od03_finetune.sh)

Take the RAW_empty checkpoint from Phase 0c and fine-tune on each OD condition. Each gets a separate output folder:

- `./models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/` -- fine-tune on `bits_RAW_OD_03/`
- `./models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/` -- fine-tune on `bits_RAW_OD_07/`

Script for OD07: [train_od07_finetune.sh](DiffSynth-Studio-SPAD/train_od07_finetune.sh).

**4a-ii. Train OD_03 from scratch (comparison experiment)** -- script: [train_od03_scratch.sh](DiffSynth-Studio-SPAD/train_od03_scratch.sh)

Train a separate model purely on OD_03 from scratch (no RAW_empty pretraining):

- `./models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/`
- This tests whether transfer from RAW_empty helps or hurts for darker inputs

**4a-iii. Frame-count ablation (inference only, no retraining)** -- script: [run_frame_ablation.sh](DiffSynth-Studio-SPAD/run_frame_ablation.sh)

Run the RAW_empty model on multi-frame inputs it wasn't trained on:

- 1 frame (baseline), 4, 16, 64, 256, 1000 accumulated frames
- This tests zero-shot generalization to higher-SNR inputs
- All evaluated on the corrected val set using the same scene-aware split

**All OD evaluation**: [run_od_ablation.sh](DiffSynth-Studio-SPAD/run_od_ablation.sh)

**All training runs use [train_controlnet.sh](DiffSynth-Studio-SPAD/train_controlnet.sh)** with only `DATASET_BASE_PATH`, `OUTPUT_PATH`, `LOG_DIR`, and `LORA_CHECKPOINT` changed per run.

### 4b. Forward-model simulation augmentation (Eq. 4.7)

Implement the Bernoulli simulation from your thesis Eq. 4.7:

- Given GT Linear16 images, generate synthetic 1-bit SPAD frames with randomized alpha (mimicking different OD filters)
- Mix synthetic pairs with real pairs during training
- Evaluate whether this improves generalization to held-out filter conditions (OOD stress test from Section 4.1)

### 4c. CFID implementation

Implement Conditional FID (from your thesis Section 4.1) to measure conditioning fidelity rather than just marginal distribution match. This addresses the specific concern you raised: standard FID can reward outputs that look real but ignore the SPAD measurement.

### 4d. Comprehensive comparison table

Target format for the paper (all values as mean +/- std across K=10 seeds):

- **Row methods**: SD1.5 (best config from Table 3.1), FLUX LoRA-on-DiT, FLUX LoRA-on-ControlNet, FLUX + DPS guidance, (optionally) FLUX + physics loss
- **Column metrics**: PSNR, SSIM, LPIPS, FID, CFID, Depth MAE, Depth Pearson r, Seg mIoU, Seg Recall, Measurement Consistency (NLL under forward model)
- **Additional rows**: multi-frame ablation (4/16/64 frames), OD filter ablation

### 4e. Failure case and qualitative analysis

Curate examples for the paper:

- Cases where high RGB variance correlates with segmentation entropy (uncertainty propagation)
- Cases where the model hallucinates objects not in GT
- Cases where FLUX outperforms SD1.5 structurally but has worse pixel metrics (perception-distortion trade-off from Section 2.6.1)
- If DPS guidance is implemented: before/after comparisons showing measurement consistency improvement

---

## Phase 5: Paper Writing (Week 5-7)

Initial draft at [paper/main.tex](paper/main.tex) and [paper/references.bib](paper/references.bib).

### Proposed NeurIPS structure (9 pages + references)

1. **Introduction** (1 page): SPAD sensors enable extreme low-light imaging but produce sparse binary data. Single-frame reconstruction is severely underconstrained. We present: (1) a ControlNet-LoRA framework on a rectified-flow transformer prior, (2) physics-consistent posterior sampling via a differentiable Bernoulli likelihood, (3) the first systematic distributional and downstream-task analysis for generative SPAD reconstruction on real data at scale.
2. **Related Work** (1 page): Quanta sensing and reconstruction (Ma et al., bit2bit, gQIR, Quanta Diffusion); diffusion for inverse problems (DPS, DDRM, PaDIS-MRI); conditioning mechanisms (ControlNet, LoRA, CtrLoRA); uncertainty in generative reconstruction.
3. **Method** (2.5 pages):
  - 3.1 Problem formulation: Bayesian inverse problem (your Eq. 2.8-2.9)
  - 3.2 Prior: FLUX.1-dev + ControlNet with LoRA-on-ControlNet (why this placement matters)
  - 3.3 Likelihood: differentiable SPAD forward model (Eq. 2.2-2.3)
  - 3.4 Physics-consistent sampling: DPS-style gradient guidance at inference
  - 3.5 Probing internal representations: linear probes on DiT activations for depth/segmentation (AC3D-inspired, Section 2d of Phase 2)
  - 3.6 Distributional evaluation framework: variance decomposition (Eq. 4.4), downstream task stability
4. **Experiments** (4 pages):
  - 4.1 Dataset (2,500 views, real SPAD hardware)
  - 4.2 Quantitative comparison table (SD1.5 vs FLUX vs FLUX+DPS, all metrics)
  - 4.3 **What does the model know?** Linear probing analysis -- depth/segmentation/bit-density decodability vs layer and timestep (key figure, analogous to AC3D Figure 5)
  - 4.4 Distributional analysis: variance maps, calibration, frame-vs-seed decomposition
  - 4.5 Downstream tasks: segmentation stability (entropy maps), depth consistency
  - 4.6 Ablations: LoRA placement, DPS guidance schedules, frame count, OD filters
5. **Discussion and Conclusion** (1 page)

---

## Risk Assessment and Contingency

- **Physics consistency (DPS) may not improve metrics**: Biggest risk. Mitigation: the paper can pivot to Angle B (uncertainty-first) where the distributional analysis IS the contribution. The physics step is discussed as principled but limited, which is still valuable analysis.
- **gQIR overlap**: Differentiate on: single-frame regime (vs. burst), real large-scale data (vs. synthetic-heavy), downstream task evaluation, distributional analysis. LoRA-on-ControlNet finding is also novel.
- **Compute for retraining**: DPS guidance (Approach A) is inference-only. Only pursue Approach B if A shows clear signal.
- **NeurIPS 2026 deadline**: Based on NeurIPS 2025 (abstract May 11, paper May 15), target similar for 2026. Start writing by late March.
- **Insufficient variance across seeds**: If all seeds produce nearly identical outputs, pivot narrative to "diffusion priors are surprisingly consistent for single-frame SPAD" -- still publishable.

---

## Key File Paths

- **FLUX training script** (USE THIS): [DiffSynth-Studio-SPAD/train_controlnet.sh](DiffSynth-Studio-SPAD/train_controlnet.sh) -- LoRA on ControlNet (best results)
- FLUX training code: [DiffSynth-Studio-SPAD/train_lora.py](DiffSynth-Studio-SPAD/train_lora.py)
- FLUX inference: [DiffSynth-Studio-SPAD/validate_lora.py](DiffSynth-Studio-SPAD/validate_lora.py)
- Metrics: [DiffSynth-Studio-SPAD/run_metrics.py](DiffSynth-Studio-SPAD/run_metrics.py), [DiffSynth-Studio-SPAD/metrics.py](DiffSynth-Studio-SPAD/metrics.py)
- Dataset prep (OLD, leaky): [DiffSynth-Studio-SPAD/prepare_dataset.py](DiffSynth-Studio-SPAD/prepare_dataset.py)
- Dataset prep (NEW, scene-aware stratified): `spad_dataset/prepare_dataset_scene_aware.py`
- SPAD extraction: [spad-diffusion/spad_dataset/extract_binary_images.py](spad-diffusion/spad_dataset/extract_binary_images.py)
- Depth analysis: [analysis_depth.py](analysis_depth.py)
- Segmentation: [sam3/sam3_eval.py](sam3/sam3_eval.py)
- Dataset metadata: [spad_dataset/metadata.csv](spad_dataset/metadata.csv) (regenerated with stratified scene-aware split)
- Extraction log: [spad_dataset/extraction_log_20260315_190624.txt](spad_dataset/extraction_log_20260315_190624.txt)
- Old leaky split backup: [spad_dataset/old_random_split/](spad_dataset/old_random_split/)

## Execution Environment

All long-running jobs (training, multi-seed inference, metrics) should be launched inside **tmux** sessions so they persist if the terminal disconnects. Use descriptive session names:

```bash
tmux new-session -d -s flux-train-raw      # Phase 0c: RAW_empty training
tmux new-session -d -s flux-train-od3-ft   # Phase 4a-i: OD_03 fine-tune
tmux new-session -d -s flux-train-od3-scratch # Phase 4a-ii: OD_03 from scratch
tmux new-session -d -s flux-multiseed      # Phase 1b: K=10 seed generation
tmux new-session -d -s sd15-reeval         # Phase 0d: SD1.5 re-evaluation
```

Monitor with `tmux ls` and attach with `tmux attach -t <name>`.

## Training Run Naming Convention

Each training run gets its own clearly labeled output folder:


| Run                 | Output Path                                               | Input Data              | Resume From           |
| ------------------- | --------------------------------------------------------- | ----------------------- | --------------------- |
| RAW_empty (primary) | `models/train/FLUX-SPAD-LoRA-SceneAware-RAW/`             | `bits/`                 | Fresh (no checkpoint) |
| OD_03 fine-tuned    | `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/`         | `bits_RAW_OD_03/`       | RAW_empty best epoch  |
| OD_07 fine-tuned    | `models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/`         | `bits_RAW_OD_07/`       | RAW_empty best epoch  |
| OD_03 from scratch  | `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/`    | `bits_RAW_OD_03/`       | Fresh (no checkpoint) |
| + Consistency loss  | `models/train/FLUX-SPAD-LoRA-SceneAware-ConsistencyLoss/` | `bits/` + paired frames | Fresh (Phase 3c)      |


## Recommended Execution Order

Ordered by: (1) what unblocks downstream work, (2) likely paper-worthy results, (3) compute efficiency.

1. ~~**Phase 0a-0c**: Dataset audit, scene-aware split, retrain FLUX~~ ✅ DONE
2. **Phase 0d**: SD1.5 re-evaluation on corrected val set
3. **Phase 1a**: Run FLUX validation + `run_metrics.py` on seed 42 -- produces Table 1 baseline
4. **Phase 1b**: Launch 10 seed generation jobs -- overnight batch, unblocks distributional analysis
5. **Phase 1c**: Run `aggregate_metrics.py` -- produces paper tables
6. **Phase 2e**: Intermediate diffusion step visualization + save activations -- unblocks probing
7. **Phase 2d**: **Linear probing of FLUX internals** -- high-impact AC3D-inspired experiment, produces key figure
8. **Phase 2a**: Per-pixel variance maps -- key figures
9. **Phase 2c**: Downstream task stability (seg entropy, depth variance)
10. **Phase 2b**: Frame-vs-seed variance decomposition
11. **Phase 2f**: Calibration analysis
12. **Phase 3e**: Physics ablation matrix (FlowDPS + consistency loss)
13. **Phase 4a**: OD filter + frame ablations + training
14. **Phase 5**: Paper writing -- fill experiments as results arrive
