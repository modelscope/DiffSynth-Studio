#!/bin/bash
# Sweep denoising_strength for img2img ablation (no ControlNet).
# Finds the best checkpoint automatically, then runs validation at each strength.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

CKPT_DIR="./models/train/FLUX-SPAD-LoRA-Img2Img-Ablation"
OUTPUT_ROOT="./validation_outputs_img2img_ablation"
METADATA="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"

STRENGTHS="0.3 0.5 0.7 0.8 0.9 1.0"
SEED=42
STEPS=28

# --- Find best checkpoint by lowest epoch number with valid file ---
# (Use last epoch if no val loss selection is available; user can override)
BEST_CKPT=""
if [ -n "$1" ]; then
    BEST_CKPT="$1"
    echo "Using user-specified checkpoint: ${BEST_CKPT}"
else
    # Default: use the last available epoch
    BEST_CKPT=$(ls "${CKPT_DIR}"/epoch-*.safetensors 2>/dev/null | sort -t'-' -k2 -n | tail -1)
    if [ -z "$BEST_CKPT" ]; then
        echo "ERROR: No checkpoints found in ${CKPT_DIR}/"
        echo "Run training first: bash train_img2img_ablation.sh"
        exit 1
    fi
    echo "Auto-selected checkpoint (last epoch): ${BEST_CKPT}"
fi

echo "=========================================="
echo "img2img Ablation — Denoising Strength Sweep"
echo "=========================================="
echo "  Checkpoint: ${BEST_CKPT}"
echo "  Strengths:  ${STRENGTHS}"
echo "  Output:     ${OUTPUT_ROOT}/"
echo "=========================================="
echo ""

for DS in ${STRENGTHS}; do
    OUT_DIR="${OUTPUT_ROOT}/strength_${DS}"
    echo ""
    echo "--- denoising_strength=${DS} ---"

    # Skip if already complete
    EXISTING=$(ls "${OUT_DIR}/output/"*.png 2>/dev/null | wc -l)
    TOTAL=$(wc -l < "${METADATA}")
    TOTAL=$((TOTAL - 1))  # subtract header
    if [ "${EXISTING}" -ge "${TOTAL}" ]; then
        echo "  Already complete (${EXISTING}/${TOTAL} images), skipping generation."
    else
        python validate_img2img.py \
            --lora_checkpoint "${BEST_CKPT}" \
            --metadata_csv "${METADATA}" \
            --output_dir "${OUT_DIR}" \
            --denoising_strength "${DS}" \
            --steps ${STEPS} \
            --seed ${SEED}
    fi

    # Compute metrics
    if [ ! -f "${OUT_DIR}/metrics.json" ] || [ "${EXISTING}" -lt "${TOTAL}" ]; then
        echo "  Computing metrics..."
        python scripts/metrics/run_metrics.py "${OUT_DIR}" --save
    else
        echo "  Metrics already computed."
    fi
done

# --- Print summary ---
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
printf "%-12s %8s %8s %8s %8s %8s\n" "Strength" "PSNR" "SSIM" "LPIPS" "FID" "CFID"
echo "------------------------------------------------------------------"

for DS in ${STRENGTHS}; do
    OUT_DIR="${OUTPUT_ROOT}/strength_${DS}"
    if [ -f "${OUT_DIR}/metrics.json" ]; then
        python3 -c "
import json, sys
m = json.load(open('${OUT_DIR}/metrics.json'))
print(f'${DS:12s} {m.get(\"psnr\",0):8.2f} {m.get(\"ssim\",0):8.4f} {m.get(\"lpips\",0):8.4f} {m.get(\"fid\",0):8.2f} {m.get(\"cfid\",0):8.2f}')
"
    else
        printf "%-12s %8s\n" "${DS}" "MISSING"
    fi
done

echo ""
echo "=========================================="
echo "All results in: ${OUTPUT_ROOT}/"
echo "=========================================="
