#!/bin/bash
# No-LoRA ablation: frozen ControlNet + frozen DiT, no LoRA
# Tests pretrained ControlNet Union Alpha with different processor modes
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

DATASET_BASE="/home/jw/engsci/thesis/spad/spad_dataset"
VAL_CSV="${DATASET_BASE}/metadata_val.csv"

for mode in gray lq canny tile depth; do
    OUT_DIR="./validation_outputs_nolora_${mode}"

    if [ -f "${OUT_DIR}/metrics.json" ]; then
        echo "[SKIP] ${mode}: already done"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "No-LoRA ablation: processor_id=${mode}"
    echo "=========================================="

    python scripts/validation/validate_lora.py \
        --metadata_csv "${VAL_CSV}" \
        --dataset_base "${DATASET_BASE}" \
        --output_dir "${OUT_DIR}" \
        --processor_id "${mode}" \
        --lora_target controlnet \
        --steps 28 --max_samples 776 --seed 42

    python scripts/metrics/run_metrics.py "${OUT_DIR}" --save
    echo "[DONE] ${mode}: metrics saved"
done

echo ""
echo "=========================================="
echo "No-LoRA Ablation Complete!"
echo "=========================================="
echo ""
echo "Results:"
for mode in gray lq canny tile depth; do
    MF="./validation_outputs_nolora_${mode}/metrics.json"
    if [ -f "$MF" ]; then
        PSNR=$(python3 -c "import json; d=json.load(open('$MF')); print(f'{d[\"psnr\"]:.2f}')" 2>/dev/null || echo "?")
        LPIPS=$(python3 -c "import json; d=json.load(open('$MF')); print(f'{d[\"lpips\"]:.3f}')" 2>/dev/null || echo "?")
        echo "  ${mode}: PSNR=${PSNR}, LPIPS=${LPIPS}"
    else
        echo "  ${mode}: no metrics"
    fi
done
