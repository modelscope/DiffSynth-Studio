# Execution Order for SPAD NeurIPS Experiments

All scripts are in `DiffSynth-Studio-SPAD/` unless otherwise noted.
Training uses the `diffsynth` conda environment. SD1.5 uses `control2`.

Best FLUX checkpoint: `epoch-15.safetensors` (val_loss=0.3083, best of 40 epochs)

```bash
export BEST_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
```

## Phase 0: Data Infrastructure ✅ COMPLETED

```bash
# 0a. Dataset audit ✅
cd /home/jw/engsci/thesis/spad/spad_dataset
python3 audit_dataset.py

# 0b. Stratified scene-aware split ✅ (indoor/outdoor balanced)
python3 prepare_dataset_scene_aware.py
# Result: 77 train (55 indoor + 22 outdoor), 20 val (14 indoor + 6 outdoor)

# 0c. RAW_empty training ✅ (40 epochs completed, best=epoch-15)
# Checkpoints at: models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-{0..39}.safetensors

# 0d. SD1.5 re-evaluation ⏳ (run after GPU is free)
cd /home/jw/engsci/thesis/spad/spad-diffusion
tmux new-session -d -s sd15-reeval 'bash run_sd15_scene_aware_eval.sh'
```

## Phase 1: Metrics + Multi-Seed

```bash
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

# 1a. Single-seed validation + metrics ✅
# Results: PSNR 17.89, SSIM 0.5962, LPIPS 0.4152, FID 66.84
# Output: validation_outputs_scene_aware/seed_42/

# 1b. Multi-seed generation (K=10) ⏳ RUNNING in tmux flux-multiseed
#   Monitor: tmux attach -t flux-multiseed
bash run_multiseed_validation.sh $BEST_CKPT ./validation_outputs_multiseed

# 1c. Aggregate after multi-seed completes
python aggregate_metrics.py ./validation_outputs_multiseed --seeds 0 13 23 42 55 67 77 88 99 123 --latex
```

## Phase 2: Analysis (after multi-seed completes)

CPU-only analyses (can run without GPU):
```bash
# 2a. Variance maps (CPU)
python compute_variance_maps.py \
  --seed-dirs ./validation_outputs_multiseed/seed_*/output \
  --gt-dir ./validation_outputs_multiseed/seed_0/ground_truth \
  --control-dir ./validation_outputs_multiseed/seed_0/input \
  --output-dir ./variance_analysis --save-maps

# 2f. Calibration (CPU)
python calibration_analysis.py \
  --seed-dirs ./validation_outputs_multiseed/seed_*/output \
  --gt-dir ./validation_outputs_multiseed/seed_0/ground_truth \
  --output-dir ./calibration_analysis
```

GPU analyses:
```bash
# 2d. Linear probing (GPU -- key experiment)
python linear_probing.py --lora_checkpoint "$BEST_CKPT" --extract --train \
  --max_samples 100 --output-dir ./probing_results

# 2e. Intermediate latents (GPU)
python save_intermediate_latents.py --lora_checkpoint "$BEST_CKPT" \
  --max_samples 20 --output-dir ./intermediate_latents

# 2b. Frame-vs-seed variance (GPU -- run inference on 7 frame folders)
python frame_vs_seed_variance.py \
  --lora_checkpoint "$BEST_CKPT" \
  --output-dir ./frame_vs_seed_analysis

# 2c. Downstream stability (GPU -- needs depth + seg models)
python downstream_stability.py \
  --seed-dirs ./validation_outputs_multiseed/seed_*/output \
  --output-dir ./downstream_analysis
```

## Phase 3: Physics Experiments (GPU)

```bash
# 3b. FlowDPS (inference-time, no retraining)
python validate_dps.py --lora_checkpoint "$BEST_CKPT" \
  --dps_guidance_scale 0.1 --dps_schedule linear_decay \
  --output_dir ./validation_outputs_dps

# 3e. Full ablation matrix
bash run_physics_ablation.sh "$BEST_CKPT"
```

## Phase 4: OD/Frame Ablations (GPU -- can run in parallel with Phase 2/3 if 2nd GPU)

```bash
# 4a-iii. Frame-count ablation (inference only, RAW model on multi-frame inputs)
bash run_frame_ablation.sh "$BEST_CKPT"

# 4a-i. Fine-tune on OD_03 from RAW checkpoint (~10-20 epochs)
tmux new-session -d -s flux-train-od3-ft 'bash train_od03_finetune.sh'

# 4a-ii. Train OD_03 from scratch
tmux new-session -d -s flux-train-od3-scratch 'bash train_od03_scratch.sh'

# 4a-i. Fine-tune on OD_07
tmux new-session -d -s flux-train-od7-ft 'bash train_od07_finetune.sh'

# After all OD training completes:
bash run_od_ablation.sh
```

## Phase 0d: SD1.5 Re-evaluation (can run anytime GPU is free)

```bash
cd /home/jw/engsci/thesis/spad/spad-diffusion
tmux new-session -d -s sd15-reeval 'bash run_sd15_scene_aware_eval.sh'
```
