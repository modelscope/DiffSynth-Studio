#!/bin/bash
# Overnight Pipeline v2 — 2026-03-30
# Runs: (1) Consistency epoch sweep, (2) OD ablation matrix, (3) Spatial crossframe re-extraction
# All sequential (single GPU)
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

echo "=========================================="
echo "STEP 1: Consistency Epoch Sweep (~2h)"
echo "=========================================="

for epoch in 5 10 15 20 25 29; do
    ckpt="models/train/FLUX-SPAD-LoRA-Consistency/epoch-${epoch}.safetensors"
    out_dir="./validation_outputs_consistency_epoch${epoch}"

    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] epoch $epoch: checkpoint not found"
        continue
    fi

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[SKIP] epoch $epoch: already done"
        continue
    fi

    echo ""
    echo "--- Consistency epoch $epoch ---"

    python scripts/validation/validate_lora.py \
        --lora_checkpoint "$ckpt" \
        --lora_target controlnet \
        --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
        --dataset_base /home/jw/engsci/thesis/spad/spad_dataset \
        --output_dir "$out_dir" \
        --steps 28 --max_samples 776 --seed 42

    python scripts/metrics/run_metrics.py "$out_dir" --save
    echo "[DONE] Consistency epoch $epoch"
done

echo ""
echo "=========================================="
echo "STEP 2: Spatial Crossframe Re-extraction (~1h) [TOP PRIORITY]"
echo "=========================================="

echo "--- Extracting spatial streaming with crossframe variance ---"
python scripts/analysis/linear_probing.py \
    --extract \
    --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
    --output-dir probing_results_allblocks \
    --hook-controlnet \
    --spatial-streaming \
    --pca-dim 0 --ridge-lambda 0.1 \
    --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
    --dataset_base /home/jw/engsci/thesis/spad/spad_dataset \
    --max_samples 776 --steps 28

echo "--- Training probes ---"
python scripts/analysis/linear_probing.py --train --output-dir probing_results_allblocks \
    --pca-dim 0 --ridge-lambda 0.1 --max_samples 776

echo "[DONE] Spatial crossframe probing"

echo ""
echo "=========================================="
echo "STEP 3: OD Ablation Matrix (~3h)"
echo "=========================================="

bash run_od_ablation.sh

echo ""
echo "=========================================="
echo "ALL STEPS COMPLETE"
echo "=========================================="
echo "Next: run metrics summary, update reports"
