#!/bin/bash
# Phase 4a-iii: Frame-count ablation (inference only, no retraining)
# Run the RAW_empty model on multi-frame inputs it wasn't trained on.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

LORA_CHECKPOINT="$1"
if [ -z "$LORA_CHECKPOINT" ]; then
    echo "Usage: $0 <lora_checkpoint_path>"
    exit 1
fi

DATASET_BASE="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_BASE="./validation_outputs_frame_ablation"

FRAME_FOLDERS=(
    "bits_multi_4"
    "bits_multi_16"
    "bits_multi_64"
    "bits_multi_256"
    "bits_multi_1000"
)

# Reuse existing baseline (seed 42, same checkpoint) via symlink
BASELINE_DIR="./validation_outputs_scene_aware/seed_42"
BASELINE_LINK="${OUTPUT_BASE}/bits"
if [ -d "${BASELINE_DIR}" ] && [ ! -e "${BASELINE_LINK}" ]; then
    echo "[symlink] bits -> existing baseline at ${BASELINE_DIR}"
    ln -s "../validation_outputs_scene_aware/seed_42" "${BASELINE_LINK}"
elif [ -L "${BASELINE_LINK}" ]; then
    echo "[reuse]  bits -> $(readlink ${BASELINE_LINK})"
fi

echo "=========================================="
echo "Frame-Count Ablation (Inference Only)"
echo "=========================================="
echo "Checkpoint: ${LORA_CHECKPOINT}"
echo "=========================================="

for FOLDER in "${FRAME_FOLDERS[@]}"; do
    VAL_CSV="${DATASET_BASE}/${FOLDER}/metadata_val.csv"
    OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER}"

    if [ ! -f "${VAL_CSV}" ]; then
        echo "[skip] ${FOLDER}: no val CSV found"
        continue
    fi

    if [ -f "${OUTPUT_DIR}/metrics.txt" ]; then
        echo "[skip] ${FOLDER}: already completed"
        continue
    fi

    echo ""
    echo "--- ${FOLDER} ---"

    python scripts/validation/validate_lora.py \
        --lora_checkpoint "${LORA_CHECKPOINT}" \
        --lora_target controlnet \
        --metadata_csv "${VAL_CSV}" \
        --dataset_base "${DATASET_BASE}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed 42 \
        --steps 28

    python scripts/metrics/run_metrics.py "${OUTPUT_DIR}" --save --output-file metrics.txt
    echo "${FOLDER} complete!"
done

echo ""
echo "=========================================="
echo "Frame-count ablation complete!"
echo "=========================================="
echo ""

# Print summary
echo "Results:"
for FOLDER in "${FRAME_FOLDERS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${FOLDER}"
    if [ -f "${OUTPUT_DIR}/metrics.txt" ]; then
        echo "--- ${FOLDER} ---"
        grep -E "PSNR|SSIM|LPIPS|FID" "${OUTPUT_DIR}/metrics.txt" 2>/dev/null
    fi
done
