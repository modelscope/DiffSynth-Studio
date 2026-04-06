# Experiment Results & Usage Guide

**Model**: FLUX.1-dev ControlNet LoRA (SceneAware-RAW)
**Best checkpoint**: `models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors` (lowest val loss 0.3083)
**Val set**: 776 images (scene-aware stratified split, 20 val locations)
**All metrics**: grayscale mode, 776 paired images

---

## 1. Consolidated Results

### 1a. Baseline (single-frame binary SPAD → RGB)

| Metric | Seed 42 | 10-seed mean±std |
|--------|---------|-----------------|
| PSNR   | 17.89   | 17.99 ± 0.09    |
| SSIM   | 0.596   | 0.596 ± 0.001   |
| LPIPS  | 0.415   | 0.415 ± 0.001   |
| FID    | 66.84   | 66.29 ± 0.74    |
| CFID   | 151.94  | 152.04 ± 1.16   |

### 1b. Physics Ablation (Latent-Space DPS Guidance)

| Config    | PSNR  | SSIM   | LPIPS  | FID   | CFID   |
|-----------|-------|--------|--------|-------|--------|
| baseline  | 17.89 | 0.5962 | 0.4152 | 66.84 | 151.94 |
| eta=0.01  | 18.02 | 0.5957 | 0.4132 | 65.81 | 151.66 |
| eta=0.05  | 18.02 | 0.5958 | 0.4132 | 65.76 | 151.39 |
| eta=0.1   | 18.02 | 0.5958 | 0.4132 | 65.86 | 151.45 |
| eta=0.5   | 18.03 | 0.5961 | 0.4131 | 65.83 | 151.50 |
| **eta=1.0** | **18.05** | **0.5969** | **0.4131** | **65.97** | **151.35** |

### 1c. Consistency Training (per-frame consistency loss)

| Config              | PSNR  | SSIM   | LPIPS  | FID   | CFID   |
|---------------------|-------|--------|--------|-------|--------|
| baseline            | 17.89 | 0.5962 | 0.4152 | 66.84 | 151.94 |
| consistency epoch-0 | 17.72 | 0.5888 | 0.4215 | 66.51 | 154.99 |
| consistency + DPS   | 17.86 | 0.5898 | 0.4199 | 65.75 | 154.15 |

### 1d. Frame-Count Ablation (zero-shot multi-frame inputs)

| Frames | PSNR  | SSIM   | LPIPS  | FID   | CFID   |
|--------|-------|--------|--------|-------|--------|
| 1      | 17.89 | 0.5962 | 0.4152 | 66.84 | 151.94 |
| 4      | 17.82 | 0.6361 | 0.3763 | 71.44 | 138.05 |
| 16     | 16.91 | 0.6513 | 0.3586 | 74.85 | 131.04 |
| 64     | 15.47 | 0.6403 | 0.3461 | 74.50 | 120.90 |
| 256    | 14.12 | 0.6045 | 0.3388 | 70.66 | 110.11 |
| 1000   | 13.04 | 0.5507 | 0.3470 | 68.52 | 108.11 |

### 1e. Analysis Results

- **Variance**: mean σ² = 0.0061, bit-density correlation = -0.226
- **Calibration**: ECE = 0.269 (underconfident — empirical coverage < nominal)
- **Intermediate latents**: saved for 20 samples × 8 steps

---

## 2. Pending Experiments

| Experiment | Status | Notes |
|------------|--------|-------|
| OD03 fine-tune | NOT TRAINED | Data ready at `bits_RAW_OD_03/` |
| OD03 scratch   | NOT TRAINED | Same data, train from scratch |
| OD07 fine-tune | NOT TRAINED | Data ready at `bits_RAW_OD_07/` |
| OD ablation eval | BLOCKED | Needs OD models first |
| Linear probing | NOT RUN | Script ready: `linear_probing.py` |
| SD1.5 re-eval  | NOT RUN | Script ready: `run_sd15_scene_aware_eval.sh` |

---

## 3. How to Run Each Experiment

All commands assume `cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD` and `conda activate diffsynth`.

### 3a. Training

```bash
# RAW baseline (already done — 40 epochs, ~17h)
bash train_scene_aware_raw.sh

# Consistency loss fine-tune (already done — 30 epochs from epoch-15 base)
bash train_consistency.sh

# OD filter training (NOT YET RUN)
bash train_od03_finetune.sh   # fine-tune from RAW epoch-15
bash train_od03_scratch.sh    # from scratch on OD03 data
bash train_od07_finetune.sh   # fine-tune from RAW epoch-15
```

**Key hyperparams** (in each .sh file):
- `LEARNING_RATE`: 1e-4 (baseline), 5e-5 (fine-tune/consistency)
- `NUM_EPOCHS`: 40 (scratch), 30 (fine-tune)
- `MAX_PIXELS`: 262144 (512×512)
- `CONSISTENCY_WEIGHT`: 0.1 (only in `train_consistency.sh`)
- `--lora_checkpoint`: omit for scratch, set for fine-tune

### 3b. Single-Seed Validation

```bash
python validate_lora.py \
  --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
  --val_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
  --output_dir ./validation_outputs_scene_aware/seed_42 \
  --seed 42
```

**Tunable**: `--seed`, `--num_inference_steps` (default 20), `--guidance_scale` (default 3.5)

### 3c. Multi-Seed Validation + Aggregation

```bash
bash run_multiseed_validation.sh
# Runs seeds: 0 13 23 42 55 67 77 88 99 123
# Then calls aggregate_metrics.py
```

### 3d. Metrics (PSNR/SSIM/LPIPS/FID/CFID)

```bash
python run_metrics.py <output_dir> --save
# CFID computed automatically when input/ subdir exists
# Add --color to skip grayscale conversion
# Add --no-fid to skip FID (faster)
```

### 3e. Latent-Space DPS (Physics Guidance)

```bash
python validate_dps.py \
  --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
  --val_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
  --output_dir ./validation_outputs_physics_ablation/dps_eta1.0 \
  --seed 42 \
  --dps_eta 1.0
```

**Tunable**: `--dps_eta` (guidance strength: 0.01–1.0, best=1.0), `--dps_schedule` (linear_decay)

### 3f. Frame-Count Ablation

```bash
bash run_frame_ablation.sh
# Uses RAW epoch-15 on bits_multi_{4,16,64,256,1000} folders
# Baseline (bits/) is symlinked to avoid re-running
```

### 3g. OD Filter Ablation (after training OD models)

```bash
bash run_od_ablation.sh
```

### 3h. Physics Ablation (full sweep)

```bash
bash run_physics_ablation.sh
# Sweeps eta: 0.01 0.05 0.1 0.5 1.0, plus baseline
```

### 3i. Analysis Scripts (CPU-friendly after inference)

```bash
python compute_variance_maps.py    # seed variance + bit-density correlation
python calibration_analysis.py     # ECE, calibration curves
python save_intermediate_latents.py  # latent snapshots at denoising steps
python linear_probing.py           # probe DiT activations (needs GPU)
```

---

## 4. Trained Model Checkpoints

| Model | Path | Epochs | Notes |
|-------|------|--------|-------|
| SceneAware-RAW | `FLUX-SPAD-LoRA-SceneAware-RAW/` | 0–39 | **Primary model**, best=epoch-15 |
| Consistency | `FLUX-SPAD-LoRA-Consistency/` | 0–29 | Fine-tuned from RAW-15, best=epoch-0 |
| Old ControlNet LoRA | `FLUX-SPAD-ControlNet-LoRA/` | 0–39 | Pre-stratified split (leaky) |
| Old LoRA-On-ControlNet | `FLUX-SPAD-LoRA-On-ControlNet/` | 0–39 | Pre-stratified split (leaky) |

---

## 5. Output Directory Map

```
validation_outputs_scene_aware/seed_42/       # Baseline (primary)
validation_outputs_multiseed/seed_{0..123}/    # 10-seed statistical eval
validation_outputs_physics_ablation/           # DPS guidance sweep
  baseline/
  dps_eta{0.01,0.05,0.1,0.5,1.0}/
validation_outputs_consistency/epoch-0/        # Consistency model
validation_outputs_consistency_dps/eta1.0/     # Consistency + DPS
validation_outputs_frame_ablation/             # Multi-frame inputs
  bits/                                        # symlink → scene_aware/seed_42
  bits_multi_{4,16,64,256,1000}/
```

Each directory contains: `input/`, `output/`, `ground_truth/`, `metrics.txt`, `metrics.json`
