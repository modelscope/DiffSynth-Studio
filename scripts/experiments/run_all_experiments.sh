#!/bin/bash
# ===================================================================
# Master Experiment Pipeline (optimized order)
# ===================================================================
# Priority order (fast experiments first):
#   1. SD1.5 linear probing (~2-3h)
#   2. img2img ablation training + sweep (~12h)
#   3. Consistency epoch sweep (~2h)
#   4. FLUX spatial crossframe re-extraction (~15h) — last, since only
#      adds one spatial target to existing complete results
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="experiment_pipeline.log"
METADATA="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
DATASET="/home/jw/engsci/thesis/spad/spad_dataset"

echo "=== Experiment pipeline started: $(date) ===" | tee -a "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh

# ──────────────────────────────────────────────────────────────────────
# 1. SD1.5 linear probing (~2-3h) — cross-architecture comparison
# ──────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] Step 1: SD1.5 linear probing..." | tee -a "$LOG"

conda activate diffsynth

SD15_DIR="/home/jw/engsci/thesis/spad/spad-diffusion"
SD15_CKPT="${SD15_DIR}/lightning_logs/spad_controlnet/two_stage_best/best-epoch=14-val_loss=0.1057.ckpt"
SD15_OUTPUT="${SD15_DIR}/probing_results_sd15"
VAL_BITS="/home/jw/engsci/thesis/spad/spad_dataset/val_only/bits"
VAL_RGB="/home/jw/engsci/thesis/spad/spad_dataset/val_only/RGB"
FLUX_TARGETS="${SCRIPT_DIR}/probing_results_allblocks/targets.json"

# Copy targets from FLUX (bit_density + depth are GT properties, reusable)
if [ ! -f "${SD15_OUTPUT}/targets.json" ] && [ -f "${FLUX_TARGETS}" ]; then
    mkdir -p "${SD15_OUTPUT}"
    cp "${FLUX_TARGETS}" "${SD15_OUTPUT}/targets.json"
    echo "  Copied FLUX targets to SD1.5" | tee -a "$LOG"
fi

cd "$SD15_DIR"

python sd15_linear_probing.py --extract \
    --checkpoint "$SD15_CKPT" \
    --val_bits "$VAL_BITS" --val_rgb "$VAL_RGB" \
    --output-dir "$SD15_OUTPUT" \
    --max_samples 776 \
    --all-blocks \
    --hook-controlnet \
    --spatial-streaming \
    --seed 42 \
    2>&1 | tee -a "${SCRIPT_DIR}/$LOG"

python sd15_linear_probing.py --train \
    --output-dir "$SD15_OUTPUT" \
    --max_samples 776 \
    --ridge-lambda 0.1 \
    2>&1 | tee -a "${SCRIPT_DIR}/$LOG"

cd "$SCRIPT_DIR"
echo "[$(date +%H:%M)] Step 1 complete: SD1.5 probing done" | tee -a "$LOG"

# ──────────────────────────────────────────────────────────────────────
# 2. img2img ablation training + sweep (~12h)
# ──────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] Step 2: img2img ablation..." | tee -a "$LOG"

conda activate diffsynth
cd "$SCRIPT_DIR"

IMG2IMG_CKPT_DIR="./models/train/FLUX-SPAD-LoRA-Img2Img-Ablation"
if [ -z "$(ls ${IMG2IMG_CKPT_DIR}/*.safetensors 2>/dev/null)" ]; then
    echo "  Training img2img ablation LoRA..." | tee -a "$LOG"
    bash scripts/training/train_img2img_ablation.sh 2>&1 | tee -a "$LOG"
else
    echo "  img2img checkpoints exist, skipping training." | tee -a "$LOG"
fi

echo "  Sweeping denoising strengths..." | tee -a "$LOG"
bash scripts/experiments/run_img2img_ablation.sh 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M)] Step 2 complete: img2img ablation done" | tee -a "$LOG"

# ──────────────────────────────────────────────────────────────────────
# 3. Consistency epoch sweep (~2h)
# ──────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] Step 3: Consistency epoch sweep..." | tee -a "$LOG"

CONSISTENCY_DIR="models/train/FLUX-SPAD-LoRA-Consistency"
CONSISTENCY_EPOCHS=(5 10 15 20 25 29)

for epoch in "${CONSISTENCY_EPOCHS[@]}"; do
    ckpt="${CONSISTENCY_DIR}/epoch-${epoch}.safetensors"
    out_dir="./validation_outputs_consistency_epoch${epoch}"

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP epoch ${epoch}: checkpoint not found" | tee -a "$LOG"
        continue
    fi

    existing=$(ls "${out_dir}/output/"*.png 2>/dev/null | wc -l)
    if [ "$existing" -ge 776 ]; then
        echo "  SKIP epoch ${epoch}: complete (${existing} images)" | tee -a "$LOG"
        if [ ! -f "${out_dir}/metrics.json" ]; then
            python scripts/metrics/run_metrics.py "$out_dir" --save 2>&1 | tee -a "$LOG"
        fi
        continue
    fi

    echo "  Generating epoch ${epoch}..." | tee -a "$LOG"
    python scripts/validation/validate_lora.py \
        --lora_checkpoint "$ckpt" \
        --lora_target controlnet \
        --metadata_csv "$METADATA" \
        --output_dir "$out_dir" \
        --steps 28 --max_samples 776 --seed 42 \
        2>&1 | tee -a "$LOG"

    python scripts/metrics/run_metrics.py "$out_dir" --save 2>&1 | tee -a "$LOG"
done

echo "[$(date +%H:%M)] Step 3 complete: consistency sweep done" | tee -a "$LOG"

# ──────────────────────────────────────────────────────────────────────
# 4. FLUX spatial crossframe re-extraction (~15h)
# ──────────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] Step 4: FLUX spatial crossframe re-extraction..." | tee -a "$LOG"

LORA_CKPT="models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
PROBING_DIR="./probing_results_allblocks"

python scripts/analysis/linear_probing.py \
    --extract \
    --lora_checkpoint "$LORA_CKPT" \
    --output-dir "$PROBING_DIR" \
    --hook-controlnet \
    --spatial-streaming \
    --all-blocks \
    --pca-dim 0 --ridge-lambda 0.1 \
    --metadata_csv "$METADATA" --dataset_base "$DATASET" \
    --max_samples 776 --steps 28 \
    2>&1 | tee -a "$LOG"

python scripts/analysis/linear_probing.py --train --output-dir "$PROBING_DIR" \
    --pca-dim 0 --ridge-lambda 0.1 --max_samples 776 \
    2>&1 | tee -a "$LOG"

python scripts/analysis/update_probing_report.py 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M)] Step 4 complete: spatial crossframe done" | tee -a "$LOG"

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== All experiments complete: $(date) ===" | tee -a "$LOG"
echo "Results:" | tee -a "$LOG"
echo "  1. SD1.5 probing: ${SD15_OUTPUT}/probes/" | tee -a "$LOG"
echo "  2. img2img ablation: ./validation_outputs_img2img_ablation/" | tee -a "$LOG"
echo "  3. Consistency sweep:" | tee -a "$LOG"
for epoch in "${CONSISTENCY_EPOCHS[@]}"; do
    out_dir="./validation_outputs_consistency_epoch${epoch}"
    [ -f "${out_dir}/metrics.json" ] && echo "    epoch-${epoch}: ${out_dir}/metrics.json" | tee -a "$LOG"
done
echo "  4. FLUX spatial crossframe: ${PROBING_DIR}/probes/" | tee -a "$LOG"
