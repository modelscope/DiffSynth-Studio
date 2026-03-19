#!/bin/bash
# Phase 3e: Physics Ablation Matrix
# Baseline vs FlowDPS vs Consistency Loss vs Combined

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

LORA_CHECKPOINT_BASELINE="$1"
LORA_CHECKPOINT_CONSISTENCY="${2:-}"

if [ -z "$LORA_CHECKPOINT_BASELINE" ]; then
    echo "Usage: $0 <baseline_checkpoint> [consistency_loss_checkpoint]"
    exit 1
fi

METADATA_CSV="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
OUTPUT_BASE="./validation_outputs_physics_ablation"

echo "=========================================="
echo "Physics Ablation Matrix"
echo "=========================================="

# 1. Baseline (no DPS, standard checkpoint)
echo ""
echo "--- (A) Baseline ---"
python validate_lora.py \
    --lora_checkpoint "${LORA_CHECKPOINT_BASELINE}" \
    --lora_target controlnet \
    --metadata_csv "${METADATA_CSV}" \
    --output_dir "${OUTPUT_BASE}/baseline" \
    --seed 42 --steps 28
python run_metrics.py "${OUTPUT_BASE}/baseline" --save --output-file metrics.txt

# 2. FlowDPS with baseline checkpoint
echo ""
echo "--- (B) Baseline + FlowDPS ---"
for ETA in 0.01 0.05 0.1 0.5; do
    echo "  eta=${ETA}..."
    python validate_dps.py \
        --lora_checkpoint "${LORA_CHECKPOINT_BASELINE}" \
        --metadata_csv "${METADATA_CSV}" \
        --output_dir "${OUTPUT_BASE}/dps_eta${ETA}" \
        --seed 42 --steps 28 \
        --dps_guidance_scale ${ETA} \
        --dps_schedule linear_decay
    python run_metrics.py "${OUTPUT_BASE}/dps_eta${ETA}" --save --output-file metrics.txt
done

# 3. Consistency loss checkpoint (if provided)
if [ -n "$LORA_CHECKPOINT_CONSISTENCY" ]; then
    echo ""
    echo "--- (C) Consistency Loss ---"
    python validate_lora.py \
        --lora_checkpoint "${LORA_CHECKPOINT_CONSISTENCY}" \
        --lora_target controlnet \
        --metadata_csv "${METADATA_CSV}" \
        --output_dir "${OUTPUT_BASE}/consistency" \
        --seed 42 --steps 28
    python run_metrics.py "${OUTPUT_BASE}/consistency" --save --output-file metrics.txt

    # 4. Combined: Consistency + DPS
    echo ""
    echo "--- (D) Consistency + FlowDPS ---"
    python validate_dps.py \
        --lora_checkpoint "${LORA_CHECKPOINT_CONSISTENCY}" \
        --metadata_csv "${METADATA_CSV}" \
        --output_dir "${OUTPUT_BASE}/consistency_dps" \
        --seed 42 --steps 28 \
        --dps_guidance_scale 0.1 \
        --dps_schedule linear_decay
    python run_metrics.py "${OUTPUT_BASE}/consistency_dps" --save --output-file metrics.txt
fi

echo ""
echo "=========================================="
echo "Ablation complete!"
echo "=========================================="
echo ""
echo "Results:"
for DIR in baseline dps_eta0.01 dps_eta0.05 dps_eta0.1 dps_eta0.5 consistency consistency_dps; do
    if [ -f "${OUTPUT_BASE}/${DIR}/metrics.txt" ]; then
        echo "--- ${DIR} ---"
        grep -E "PSNR|SSIM|LPIPS|FID" "${OUTPUT_BASE}/${DIR}/metrics.txt" 2>/dev/null
    fi
done
