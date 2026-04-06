#!/bin/bash
# OD filter ablation — single image, all 4 models × 4 SPAD inputs
# Each run is a separate process to avoid OOM from LoRA fuse accumulation
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

DATASET_BASE="/home/jw/engsci/thesis/spad/spad_dataset"
OUT_BASE="./validation_outputs_od_single_image"

declare -A CKPTS
CKPTS=(
    ["raw"]="models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
    ["od03_ft"]="models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT/epoch-best.safetensors"
    ["od03_scratch"]="models/train/FLUX-SPAD-LoRA-SceneAware-OD03-Scratch/epoch-best.safetensors"
    ["od07_ft"]="models/train/FLUX-SPAD-LoRA-SceneAware-OD07-FT/epoch-best.safetensors"
)

SPAD_CSVS=("raw" "od01" "od03" "od07")

for MODEL in raw od03_ft od03_scratch od07_ft; do
    CKPT="${CKPTS[$MODEL]}"
    for SPAD in "${SPAD_CSVS[@]}"; do
        OUTDIR="${OUT_BASE}/${MODEL}_on_${SPAD}"
        CSV="/tmp/od_${SPAD}.csv"

        if [ -d "${OUTDIR}/output" ] && [ "$(ls ${OUTDIR}/output/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "[SKIP] ${MODEL} on ${SPAD}"
            continue
        fi

        echo ""
        echo "=== ${MODEL} on ${SPAD} ==="
        python scripts/validation/validate_lora.py \
            --lora_checkpoint "${CKPT}" \
            --lora_target controlnet \
            --metadata_csv "${CSV}" \
            --dataset_base "${DATASET_BASE}" \
            --output_dir "${OUTDIR}" \
            --steps 28 --seed 42 --overwrite
    done
done

echo ""
echo "=========================================="
echo "Single-image OD ablation complete!"
echo "=========================================="
echo "Outputs in ${OUT_BASE}/"
ls -d ${OUT_BASE}/*/output/ 2>/dev/null | while read d; do
    echo "  $(dirname $d | xargs basename): $(ls $d/*.png 2>/dev/null | wc -l) images"
done
