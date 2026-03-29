# Project Handoff: SPAD-to-RGB Reconstruction via Diffusion Priors

**Date**: 2026-03-23
**Repo**: `/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD`
**Branch**: `main`
**Last commit**: `edf6f1d` (2026-03-19) — bulk commit of all LoRA/ControlNet work, training scripts, analysis, metrics
**14 files modified since last commit** (not yet committed), see `agent/git_diff.patch`

---

## Objective

Build a NeurIPS-worthy paper on **SPAD single-photon sensor to RGB image reconstruction** using a 12B-parameter FLUX.1-dev rectified-flow transformer conditioned via ControlNet + LoRA. The paper has three pillars:

1. **"What does the model know?"** — Linear probing of DiT internals (AC3D-inspired, Prof. Lindell's priority)
2. **Uncertainty-Hallucination-Consistency** — Multi-seed distributional analysis of generative reconstruction
3. **Physics-Consistent Posterior Sampling** — Latent-space DPS guidance with Bernoulli likelihood

---

## Current Status

### ACTIVELY RUNNING NOW

| Job | tmux session | Started | ETA |
|-----|-------------|---------|-----|
| OD03 fine-tune training (20 epochs) | `od-training` | 2026-03-22 21:57 | ~8h remaining (epoch 17/20) |
| OD07 fine-tune (queued next) | same session | — | ~14h after OD03 finishes |
| OD03 from-scratch (queued last) | same session | — | ~28h after OD03 finishes |

**GPU**: NVIDIA RTX 5090 (32GB VRAM), currently at 30/32GB used by training.
**Conda env**: `diffsynth` (Python 3.x, PyTorch with CUDA)

### What Has Been Completed

| Phase | Description | Output Location |
|-------|-------------|-----------------|
| **0a** Dataset audit | Scanned all 20+ `bits_*` folders, built inventory | `spad_dataset/dataset_inventory.json` |
| **0b** Scene-aware split | Stratified indoor/outdoor split, zero data leakage | `spad_dataset/metadata_{train,val}.csv` (1850/776 views) |
| **0c** FLUX RAW training | 40 epochs from scratch, best=epoch-15 (val loss 0.3083) | `models/train/FLUX-SPAD-LoRA-SceneAware-RAW/` |
| **1a-1c** Baseline + multi-seed | K=10 seeds, full metrics including CFID | `validation_outputs_multiseed/` + `aggregated_metrics.json` |
| **2a** Variance maps | 776 per-pixel variance images | `variance_analysis/` |
| **2d** Linear probing | 3-phase pipeline: targets, activation extraction, ridge regression probes | `probing_results/` (42GB activations + figures) |
| **2e** Intermediate latents | 20 samples × 8 denoising steps decoded | `intermediate_latents/` |
| **2f** Calibration | ECE=0.269, 776 calibration images | `calibration_analysis/` |
| **3a** SPAD forward model | Differentiable Bernoulli likelihood | `diffsynth/diffusion/spad_forward.py` |
| **3b** Latent-space DPS | Monkeypatched `pipe.step` for latent guidance | `diffsynth/diffusion/latent_dps.py`, `validate_dps.py` |
| **3c** Consistency loss | Per-frame consistency training with stop-gradient | `diffsynth/diffusion/consistency_loss.py`, `train_consistency.py` |
| **3e** Physics ablation | DPS eta sweep (0.01-1.0) + baseline | `validation_outputs_physics_ablation/` |
| **Frame ablation** | 1,4,16,64,256,1000 accumulated frames | `validation_outputs_frame_ablation/` |
| **Consistency eval** | epoch-0 + consistency+DPS combined | `validation_outputs_consistency/`, `validation_outputs_consistency_dps/` |
| **CFID metric** | Conditional FID with numerical stability fixes | `metrics.py` |

### Key Results Summary

**Baseline (RAW single-frame SPAD → RGB)**:
- PSNR: 17.99 ± 0.09 | SSIM: 0.596 ± 0.001 | LPIPS: 0.415 ± 0.001 | FID: 66.3 ± 0.7 | CFID: 152.0 ± 1.2

**Physics DPS (best eta=1.0)**: PSNR 18.05 (+0.16), FID 65.97 (-0.87), CFID 151.35 (-0.59) — modest improvement

**Consistency training**: Slight degradation (PSNR 17.72, epoch-0 was best)

**Frame ablation**: LPIPS monotonically improves with more frames (0.415→0.347), PSNR degrades for >4 frames (model trained on single-frame only, zero-shot transfer)

**Linear probing (HIGHEST PRIORITY RESULT)**:
- Spatial bit density: R²=0.99 — model perfectly preserves input photon density
- Spatial depth: R²=0.64 — model implicitly encodes 3D geometry without depth supervision
- Spatial uncertainty: R²=0.41 — activations predict where model will be uncertain
- All peak at mid-network (single blocks 0-9) and mid-denoising (step ~14/28)
- Figures: `probing_results/probes/heatmap_spatial_*.png`, `comparison_best_timestep.png`

### What Is Partially Done

| Item | Status | Details |
|------|--------|---------|
| OD filter training | Running | OD03-FT at epoch 17/20, OD07-FT and OD03-scratch queued |
| OD filter evaluation | Blocked | Needs trained OD checkpoints; script ready: `run_od_ablation.sh` |
| SD1.5 re-evaluation | Not started | Script ready: `run_sd15_scene_aware_eval.sh` in `spad-diffusion/` |
| Plan pending phases | Pending | See `agent/TODO.md` for full list |
| 14 modified files | Uncommitted | See `agent/git_diff.patch` |

### What Is Broken or Uncertain

1. **Consistency training hurt performance** — epoch-0 was best (barely fine-tuned), later epochs degraded. The per-frame consistency loss with `lambda=0.1` may be too strong, or the stop-gradient approach may be limiting. Worth re-exploring with lower lambda (0.01, 0.001).

2. **Global probing R² values are negative** — Expected for `n << D` (80 samples vs 3072 features). Pearson r is the meaningful metric for global probing. The spatial probing (1024 tokens × 100 samples = 102,400 samples) has proper R² values.

3. **Best checkpoint controversy** — `epoch-15` was selected as best by val loss (0.3083), but the user previously used `epoch-39`. May be worth doing an epoch sweep validation.

4. **16-bit image loading** — A critical bug was fixed where PIL's `convert('RGB')` clamped 16-bit values >255 to 255 instead of normalizing. The fix (`load_spad_image()` in `diffsynth/core/data/operators.py`) is applied to all scripts but NOT committed.

5. **OD training scripts reference `epoch-15.safetensors`** — If the user decides epoch-39 is better, the `--lora_checkpoint` in `train_od03_finetune.sh` / `train_od07_finetune.sh` needs updating.

6. **SD1.5 conda env** — The SD1.5 evaluation script expects `conda activate control2` (not `control`). Verify before running.

---

## Important Constraints and Assumptions

- **Conda environment**: Always activate `diffsynth` before running any script: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth`
- **FP8 offloading**: All FLUX scripts use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and FP8 model quantization for the 32GB VRAM constraint
- **Data leakage**: The original train-test split was leaky (94/101 locations in both sets). The corrected scene-aware stratified split is in `spad_dataset/metadata_{train,val}.csv`. Old split backed up at `spad_dataset/old_random_split/`
- **Metrics computed in grayscale** — all PSNR/SSIM/LPIPS/FID/CFID are computed on grayscale images (matching the single-channel SPAD input)
- **SPAD binary captures**: Raw captures are on a remote server (9TB), not local. Only extracted PNGs are local in `spad_dataset/bits_*/`
- **Probing activation data is 42GB** — stored at `probing_results/activations/`. Do not re-extract unless necessary.

---

## Files That Matter Most

### Core Scripts (Modified, uncommitted)
| File | Purpose |
|------|---------|
| `linear_probing.py` | 3-phase probing pipeline (prepare-targets, extract, train) — **most complex script** |
| `validate_lora.py` | FLUX inference with LoRA, handles 16-bit SPAD loading |
| `validate_dps.py` | Inference with latent-space DPS guidance |
| `run_metrics.py` | PSNR/SSIM/LPIPS/FID/CFID computation |
| `metrics.py` | CFID implementation with numerical stability |
| `diffsynth/core/data/operators.py` | `load_spad_image()` — 16-bit normalization fix |
| `diffsynth/diffusion/latent_dps.py` | Latent-space DPS guidance (new file) |
| `diffsynth/diffusion/consistency_loss.py` | Per-frame consistency loss |
| `train_consistency.py` | Consistency training script (new file) |
| `paired_spad_dataset.py` | Dataset class for paired SPAD frames (new file) |

### Shell Scripts
| Script | What it does |
|--------|-------------|
| `train_scene_aware_raw.sh` | Train RAW baseline (DONE) |
| `train_od03_finetune.sh` | Fine-tune on OD03 from RAW-15 (RUNNING) |
| `train_od07_finetune.sh` | Fine-tune on OD07 from RAW-15 (QUEUED) |
| `train_od03_scratch.sh` | Train OD03 from scratch (QUEUED) |
| `train_consistency.sh` | Train with consistency loss (DONE) |
| `run_multiseed_validation.sh` | K=10 seed validation (DONE) |
| `run_frame_ablation.sh` | Multi-frame zero-shot ablation (DONE) |
| `run_physics_ablation.sh` | DPS eta sweep (DONE) |
| `run_od_ablation.sh` | OD filter evaluation (BLOCKED on training) |

### Data and Outputs
| Path | Size | What |
|------|------|------|
| `spad_dataset/` | 13GB | All SPAD inputs + RGB ground truth |
| `spad_dataset/metadata_val.csv` | 776 lines | Scene-aware stratified val set |
| `models/train/FLUX-SPAD-LoRA-SceneAware-RAW/` | ~40 checkpoints | Primary trained model |
| `models/train/FLUX-SPAD-LoRA-Consistency/` | 30 checkpoints | Consistency-trained model |
| `validation_outputs_multiseed/` | 6.1GB | 10-seed outputs + aggregated metrics |
| `validation_outputs_physics_ablation/` | 3.7GB | DPS eta sweep (6 configs) |
| `validation_outputs_frame_ablation/` | 3.5GB | 6 frame-count configs |
| `probing_results/` | 42GB | Activations (200 .pt files) + figures + JSON |
| `EXPERIMENTS.md` | — | Full experiment guide with results tables |

### Key Config/Plan
| Path | What |
|------|------|
| `~/.cursor/plans/spad_neurips_full_plan_1cbbff23.plan.md` | Master research plan (666 lines) |
| `EXPERIMENTS.md` | Experiment results + reproduction commands |
| `accelerate_config.yaml` | HuggingFace accelerate config for training |

---

## Exact Next Steps (in priority order)

### 1. Wait for OD training to complete (~24-36h)
```bash
tmux attach -t od-training  # Check progress
```
Checkpoints will appear at:
- `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/`
- `models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/`
- `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/`

### 2. Run OD evaluation (after training)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth
bash run_od_ablation.sh
```

### 3. Commit all outstanding changes
14 modified files + 10+ new files need to be committed. See `agent/git_status.txt`.

### 4. SD1.5 re-evaluation for fair comparison
```bash
cd /home/jw/engsci/thesis/spad/spad-diffusion
bash run_sd15_scene_aware_eval.sh
```

### 5. Remaining analysis tasks (CPU-friendly)
- Frame-vs-seed variance decomposition
- Downstream task stability (segmentation entropy, depth variance)

### 6. Paper writing
- Linear probing figures are ready for the paper
- Experiment tables can be generated from `EXPERIMENTS.md`
- Initial draft structure outlined in the plan

---

## Risks / Gotchas

1. **16-bit image loading**: The `load_spad_image()` fix in `operators.py` is critical. Without it, multi-frame SPAD inputs appear white. This fix is NOT committed.

2. **Memory**: Linear probing extracted 42GB of activations. If you need to re-extract, ensure enough disk space. Use `--save-spatial` flag only if needed.

3. **CFID numerical stability**: CFID uses `float64` and Ridge regularization internally because the Inception features are high-dimensional (2048) vs few samples (~776). If you get negative CFID values, increase `reg` in `compute_cfid()`.

4. **DPS guidance is latent-space only**: The original pixel-space DPS approach caused OOM on 32GB. The implemented version operates entirely in latent space with an L2 consistency gradient, which is a simplification of the full Bernoulli likelihood.

5. **Consistency training was underwhelming**: The per-frame consistency loss didn't help. This is acknowledged in the paper framing as "principled but limited in practice."

6. **DNS resolution errors**: The machine occasionally fails to resolve `www.modelscope.cn` which is needed for first-time model downloads. Models are cached after first download.

7. **Git stash**: There are 2 stashed changesets (`stash@{0}`: misc temp files, `stash@{1}`: tracking docs). Apply with `git stash pop` if needed.

8. **Probing activations directory**: 42GB at `probing_results/activations/`. Contains 100 `global_*.pt` and 100 `spatial_*.pt` files. Do NOT delete — re-extraction takes ~3 hours on GPU.

---

## Commands Already Run (key ones)

```bash
# Training (completed)
tmux new -d -s flux-train 'bash train_scene_aware_raw.sh'  # 40 epochs, ~17h
tmux new -d -s consistency 'bash train_consistency.sh'      # 30 epochs, ~10h

# Inference (completed)
bash run_multiseed_validation.sh   # 10 seeds × 776 images
bash run_frame_ablation.sh         # 6 frame configs × 776 images
bash run_physics_ablation.sh       # 6 DPS configs × 776 images

# Metrics (completed)
python run_metrics.py validation_outputs_multiseed/seed_42 --save
python run_metrics.py validation_outputs_physics_ablation/dps_eta1.0 --save
# (run on all output dirs)

# Linear probing (completed)
python linear_probing.py --prepare-targets
python linear_probing.py --extract --all --save-spatial
python linear_probing.py --train

# Currently running
tmux new -d -s od-training 'bash train_od03_finetune.sh && bash train_od07_finetune.sh && bash train_od03_scratch.sh'
```

---

## Useful Outputs and Error Messages

**Probing results JSON**: `probing_results/probes/probing_results.json` — contains all R², Pearson r values for every (block, timestep, target) combination.

**Aggregated multi-seed metrics**: `validation_outputs_multiseed/aggregated_metrics.json` — cross-seed mean/std for all metrics.

**Variance summary**: `variance_analysis/variance_summary.json` — per-image variance statistics, mean σ²=0.0061, bit-density correlation=-0.226.

**Calibration**: `calibration_analysis/calibration_results.json` — ECE=0.269.

**Known error patterns**:
- `ModuleNotFoundError: No module named 'numpy'` → wrong conda env, activate `diffsynth`
- `TypeError: FluxImagePipeline.from_pretrained() got multiple values` → check `model_configs` kwarg format
- `CUDA out of memory` during DPS → use `validate_dps.py` (latent-space), not the original pixel-space approach
- White multi-frame images → 16-bit loading bug, ensure `load_spad_image()` is used
