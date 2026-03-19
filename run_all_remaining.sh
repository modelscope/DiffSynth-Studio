#!/bin/bash
# Master script: Run all remaining phases after multi-seed validation completes.
# Usage: tmux new-session -d -s run-all 'bash run_all_remaining.sh'

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

BEST_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
MULTISEED_DIR="./validation_outputs_multiseed"

echo "============================================================"
echo "SPAD NeurIPS -- Running All Remaining Phases"
echo "============================================================"
echo "Checkpoint: ${BEST_CKPT}"
echo "Multi-seed dir: ${MULTISEED_DIR}"
echo ""

# Phase 1c: Re-aggregate metrics (now with all 10 seeds)
echo "=== Phase 1c: Aggregate Metrics ==="
python aggregate_metrics.py "${MULTISEED_DIR}" --seeds 0 13 23 42 55 67 77 88 99 123 --latex
echo ""

# Phase 2a: Variance maps (CPU-only)
echo "=== Phase 2a: Variance Maps ==="
python compute_variance_maps.py \
  --seed-dirs ${MULTISEED_DIR}/seed_0/output ${MULTISEED_DIR}/seed_13/output \
              ${MULTISEED_DIR}/seed_23/output ${MULTISEED_DIR}/seed_42/output \
              ${MULTISEED_DIR}/seed_55/output ${MULTISEED_DIR}/seed_67/output \
              ${MULTISEED_DIR}/seed_77/output ${MULTISEED_DIR}/seed_88/output \
              ${MULTISEED_DIR}/seed_99/output ${MULTISEED_DIR}/seed_123/output \
  --gt-dir ${MULTISEED_DIR}/seed_0/ground_truth \
  --control-dir ${MULTISEED_DIR}/seed_0/input \
  --output-dir ./variance_analysis --save-maps
echo ""

# Phase 2f: Calibration (CPU-only)
echo "=== Phase 2f: Calibration Analysis ==="
python calibration_analysis.py \
  --seed-dirs ${MULTISEED_DIR}/seed_0/output ${MULTISEED_DIR}/seed_13/output \
              ${MULTISEED_DIR}/seed_23/output ${MULTISEED_DIR}/seed_42/output \
              ${MULTISEED_DIR}/seed_55/output ${MULTISEED_DIR}/seed_67/output \
              ${MULTISEED_DIR}/seed_77/output ${MULTISEED_DIR}/seed_88/output \
              ${MULTISEED_DIR}/seed_99/output ${MULTISEED_DIR}/seed_123/output \
  --gt-dir ${MULTISEED_DIR}/seed_0/ground_truth \
  --output-dir ./calibration_analysis
echo ""

# Phase 2e: Intermediate latents (GPU)
echo "=== Phase 2e: Intermediate Latents ==="
python save_intermediate_latents.py \
  --lora_checkpoint "${BEST_CKPT}" \
  --max_samples 20 \
  --output-dir ./intermediate_latents
echo ""

# Phase 2d: Linear probing (GPU -- key experiment)
echo "=== Phase 2d: Linear Probing ==="
python linear_probing.py \
  --lora_checkpoint "${BEST_CKPT}" \
  --extract --train \
  --max_samples 100 \
  --output-dir ./probing_results
echo ""

# Phase 3: Physics ablation (GPU)
echo "=== Phase 3e: Physics Ablation ==="
bash run_physics_ablation.sh "${BEST_CKPT}"
echo ""

# Phase 4a-iii: Frame-count ablation (inference only, GPU)
echo "=== Phase 4a-iii: Frame-Count Ablation ==="
bash run_frame_ablation.sh "${BEST_CKPT}"
echo ""

echo "============================================================"
echo "ALL PHASES COMPLETE!"
echo "============================================================"
echo ""
echo "Still pending (sequential GPU training):"
echo "  - Phase 4a-i:  bash train_od03_finetune.sh"
echo "  - Phase 4a-ii: bash train_od03_scratch.sh"
echo "  - Phase 4a-i:  bash train_od07_finetune.sh"
echo "  - Phase 0d:    SD1.5 re-eval (cd spad-diffusion && bash run_sd15_scene_aware_eval.sh)"
echo ""
echo "After OD training completes:"
echo "  - bash run_od_ablation.sh"
