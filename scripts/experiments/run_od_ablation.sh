#!/bin/bash
# Phase 4a: OD filter ablation evaluation
# Runs each trained OD model on its corresponding val set and on cross-OD val sets.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

DATASET_BASE="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_BASE="./validation_outputs_od_ablation"

RAW_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
OD03_FT_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/epoch-best.safetensors"
OD03_SCR_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/epoch-best.safetensors"
OD07_FT_CKPT="./models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/epoch-best.safetensors"

declare -A CHECKPOINTS
CHECKPOINTS=(
    ["raw"]="${RAW_CKPT}"
    ["od03_ft"]="${OD03_FT_CKPT}"
    ["od03_scratch"]="${OD03_SCR_CKPT}"
    ["od07_ft"]="${OD07_FT_CKPT}"
)

OD_FOLDERS=("bits" "bits_RAW_OD_01" "bits_RAW_OD_03" "bits_RAW_OD_07")

echo "=========================================="
echo "OD Filter Ablation Matrix"
echo "=========================================="

for MODEL_NAME in raw od03_ft od03_scratch od07_ft; do
    CKPT="${CHECKPOINTS[$MODEL_NAME]}"
    if [ ! -f "${CKPT}" ]; then
        echo "[skip] ${MODEL_NAME}: checkpoint not found at ${CKPT}"
        continue
    fi

    for FOLDER in "${OD_FOLDERS[@]}"; do
        VAL_CSV="${DATASET_BASE}/${FOLDER}/metadata_val.csv"
        OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/${FOLDER}"

        if [ ! -f "${VAL_CSV}" ]; then
            echo "[skip] ${MODEL_NAME} on ${FOLDER}: no val CSV"
            continue
        fi

        if [ -f "${OUTPUT_DIR}/metrics.txt" ]; then
            echo "[skip] ${MODEL_NAME} on ${FOLDER}: already done"
            continue
        fi

        echo ""
        echo "--- ${MODEL_NAME} on ${FOLDER} ---"

        python scripts/validation/validate_lora.py \
            --lora_checkpoint "${CKPT}" \
            --lora_target controlnet \
            --metadata_csv "${VAL_CSV}" \
            --dataset_base "${DATASET_BASE}" \
            --output_dir "${OUTPUT_DIR}" \
            --seed 42 --steps 28

        python scripts/metrics/run_metrics.py "${OUTPUT_DIR}" --save --output-file metrics.txt
    done
done

echo ""
echo "=========================================="
echo "OD Ablation Complete!"
echo "=========================================="
echo ""
echo "Cross-OD Results Matrix:"
printf "%-15s" "Model\\Data"
for FOLDER in "${OD_FOLDERS[@]}"; do
    printf "%-20s" "${FOLDER}"
done
echo ""
echo "$(printf '%.0s-' {1..80})"

for MODEL_NAME in raw od03_ft od03_scratch od07_ft; do
    printf "%-15s" "${MODEL_NAME}"
    for FOLDER in "${OD_FOLDERS[@]}"; do
        METRICS_FILE="${OUTPUT_BASE}/${MODEL_NAME}/${FOLDER}/metrics.txt"
        if [ -f "${METRICS_FILE}" ]; then
            PSNR=$(grep "PSNR" "${METRICS_FILE}" | head -1 | awk '{print $2}')
            printf "%-20s" "${PSNR}"
        else
            printf "%-20s" "-"
        fi
    done
    echo ""
done
