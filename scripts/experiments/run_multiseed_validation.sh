#!/bin/bash
# Phase 1b: Generate K=10 seeds via validate_lora.py on corrected val set
# Run full metrics pipeline on each seed.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

LORA_CHECKPOINT="$1"
if [ -z "$LORA_CHECKPOINT" ]; then
    echo "Usage: $0 <lora_checkpoint_path> [output_base_dir]"
    echo "  e.g.: $0 ./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-best.safetensors"
    exit 1
fi

OUTPUT_BASE="${2:-./validation_outputs_multiseed}"
METADATA_CSV="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"

SEEDS=(0 13 23 42 55 67 77 88 99 123)

echo "=========================================="
echo "Multi-Seed Validation"
echo "=========================================="
echo "Checkpoint: ${LORA_CHECKPOINT}"
echo "Output:     ${OUTPUT_BASE}"
echo "Seeds:      ${SEEDS[*]}"
echo "=========================================="
echo ""

for SEED in "${SEEDS[@]}"; do
    SEED_DIR="${OUTPUT_BASE}/seed_${SEED}"

    if [ -f "${SEED_DIR}/metrics.txt" ]; then
        echo "[skip] seed ${SEED}: metrics.txt already exists"
        continue
    fi

    echo ""
    echo "--- Seed ${SEED} ---"

    python scripts/validation/validate_lora.py \
        --lora_checkpoint "${LORA_CHECKPOINT}" \
        --lora_target controlnet \
        --metadata_csv "${METADATA_CSV}" \
        --output_dir "${SEED_DIR}" \
        --seed "${SEED}" \
        --steps 28

    echo "Computing metrics for seed ${SEED}..."
    python scripts/metrics/run_metrics.py "${SEED_DIR}" --save --output-file metrics.txt

    echo "Seed ${SEED} complete!"
done

echo ""
echo "=========================================="
echo "All seeds complete! Aggregating..."
echo "=========================================="

SEED_DIRS=""
for SEED in "${SEEDS[@]}"; do
    SEED_DIRS="${SEED_DIRS} ${OUTPUT_BASE}/seed_${SEED}"
done

python scripts/metrics/aggregate_metrics.py "${OUTPUT_BASE}" --seeds "${SEEDS[@]}" --latex

echo "Done!"
