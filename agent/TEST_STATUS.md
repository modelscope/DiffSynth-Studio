# Test & Validation Status

**Updated**: 2026-03-23

This project does not have a formal test suite. Validation is done via experiment runs with quantitative metrics.

---

## Experiments Run (All Passing)

### Baseline Validation (RAW single-frame → RGB)
- **Status**: PASS
- **Output**: `validation_outputs_scene_aware/seed_42/`
- **Images**: 776/776 generated
- **Metrics**: PSNR 17.89, SSIM 0.596, LPIPS 0.415, FID 66.84, CFID 151.94
- **Command**: `python validate_lora.py --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors --val_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv --output_dir validation_outputs_scene_aware/seed_42 --seed 42`

### Multi-Seed Validation (K=10)
- **Status**: PASS (all 10 seeds complete)
- **Output**: `validation_outputs_multiseed/seed_{0,13,23,42,55,67,77,88,99,123}/`
- **Images**: 776/776 per seed, all seeds verified
- **Aggregated**: `validation_outputs_multiseed/aggregated_metrics.json`
- **Command**: `bash run_multiseed_validation.sh`

### Physics Ablation (DPS eta sweep)
- **Status**: PASS (6 configurations)
- **Output**: `validation_outputs_physics_ablation/{baseline,dps_eta0.01,dps_eta0.05,dps_eta0.1,dps_eta0.5,dps_eta1.0}/`
- **Images**: 776/776 per config
- **Command**: `bash run_physics_ablation.sh`

### Frame-Count Ablation
- **Status**: PASS (6 frame counts)
- **Output**: `validation_outputs_frame_ablation/{bits,bits_multi_4,bits_multi_16,bits_multi_64,bits_multi_256,bits_multi_1000}/`
- **Images**: 776/776 per config
- **Note**: `bits/` is a symlink to `validation_outputs_scene_aware/seed_42/` to avoid redundant computation
- **Command**: `bash run_frame_ablation.sh`

### Consistency Training + Evaluation
- **Status**: PASS (training + inference)
- **Training**: 30 epochs, checkpoints at `models/train/FLUX-SPAD-LoRA-Consistency/epoch-{0..29}.safetensors`
- **Eval output**: `validation_outputs_consistency/epoch-0/` (776 images)
- **Combined DPS**: `validation_outputs_consistency_dps/eta1.0/` (776 images, consistency model + DPS eta=1.0)
- **Command**: `bash train_consistency.sh` then `python validate_lora.py ...`

### Linear Probing (3-phase pipeline)
- **Status**: PASS
- **Phase 1 (prepare-targets)**: Computed bit density, depth (DPT), variance for 100 val samples
- **Phase 2 (extract)**: 100 `global_*.pt` + 100 `spatial_*.pt` files (42GB total)
- **Phase 3 (train)**: Ridge regression probes for 6 targets × 70 (block, timestep) combinations
- **Output**: `probing_results/probes/probing_results.json`, 13 PNG/PDF figure pairs
- **Key result**: Spatial depth R²=0.64, spatial bit density R²=0.99
- **Command**: `python linear_probing.py --prepare-targets && python linear_probing.py --extract --all --save-spatial && python linear_probing.py --train`

### Variance Analysis
- **Status**: PASS
- **Output**: `variance_analysis/` (776 variance images + `variance_summary.json`)
- **Result**: Mean σ²=0.0061, bit-density correlation=-0.226

### Calibration Analysis
- **Status**: PASS
- **Output**: `calibration_analysis/calibration_results.json`
- **Result**: ECE=0.269 (underconfident)

### Intermediate Latents
- **Status**: PASS
- **Output**: `intermediate_latents/sample_{0000..0019}/` (20 samples × 8 steps + ground truth + input)

---

## Experiments Running

### OD03 Fine-tune Training
- **Status**: RUNNING (epoch 17/20 as of 2026-03-23)
- **tmux**: `od-training`
- **Output will be at**: `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/`
- **Check progress**: `tmux capture-pane -t od-training -p | tail -5`

### OD07 Fine-tune Training
- **Status**: QUEUED (will auto-start after OD03)
- **Output will be at**: `models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/`

### OD03 From-Scratch Training
- **Status**: QUEUED (will auto-start after OD07)
- **Output will be at**: `models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/`

---

## Experiments NOT Run Yet

| Experiment | Script/Command | Blocked By | Notes |
|------------|---------------|------------|-------|
| OD filter evaluation | `bash run_od_ablation.sh` | OD training completion | Script ready |
| SD1.5 re-evaluation | `bash run_sd15_scene_aware_eval.sh` (in `spad-diffusion/`) | Nothing (can run when GPU free) | Needs `conda activate control2` |
| Frame-vs-seed decomposition | Not yet scripted | Multi-seed runs per frame folder | Needs K=10 seeds for each `bits_frame_*` folder |
| Downstream stability | `analysis_depth.py`, `sam3/sam3_eval.py` | Nothing | May need script updates for new val set paths |
| Consistency with lower lambda | Edit `train_consistency.sh` | GPU availability | Try lambda=0.01, 0.001 |
| Epoch sweep validation | Run `validate_lora.py` for epochs 10,15,20,25,30,35,39 | GPU availability | Settle the epoch-15 vs epoch-39 question |

---

## How to Rerun Validation

### Prerequisites
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Single-seed inference
```bash
python validate_lora.py \
  --lora_path models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
  --val_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
  --output_dir ./validation_outputs_scene_aware/seed_42 \
  --seed 42
```

### Compute metrics on an output directory
```bash
python run_metrics.py <output_dir> --save
# Outputs: metrics.json, metrics.txt in <output_dir>
```

### Rerun linear probing (targets already prepared)
```bash
# Only re-train probes (uses cached activations):
python linear_probing.py --train
# Full re-extraction (slow, ~3h, GPU required):
python linear_probing.py --extract --all --save-spatial
```
