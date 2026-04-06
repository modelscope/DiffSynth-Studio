#!/bin/bash
# ===================================================================
# Linear Probing Full Experiment Pipeline
# ===================================================================
# Runs 4 experiments:
#   1. Full dataset (776 samples), all blocks, with LoRA (main model)
#   2. Full dataset (776 samples), all blocks, NO LoRA (control baseline)
#   3. Train probes on both experiments
#   4. Generate comparison figures
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LORA_CKPT="models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
METADATA="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
DATASET="/home/jw/engsci/thesis/spad/spad_dataset"
MULTISEED_DIR="./validation_outputs_multiseed"
MAX_SAMPLES=776

# Output dirs
OUT_MAIN="./probing_results_allblocks"
OUT_CTRL="./probing_results_control"

echo "=============================================="
echo "  Phase 1: Prepare Targets (776 samples)"
echo "=============================================="
# Only need to do this once — targets are model-independent
if [ ! -f "${OUT_MAIN}/targets.json" ]; then
    python scripts/analysis/linear_probing.py \
        --prepare-targets \
        --output-dir "$OUT_MAIN" \
        --metadata_csv "$METADATA" \
        --dataset_base "$DATASET" \
        --multiseed-dir "$MULTISEED_DIR" \
        --max_samples "$MAX_SAMPLES"
    # Copy targets to control dir too
    mkdir -p "$OUT_CTRL"
    cp "${OUT_MAIN}/targets.json" "${OUT_CTRL}/targets.json"
else
    echo "  Targets already exist, skipping."
    mkdir -p "$OUT_CTRL"
    [ -f "${OUT_CTRL}/targets.json" ] || cp "${OUT_MAIN}/targets.json" "${OUT_CTRL}/targets.json"
fi

echo ""
echo "=============================================="
echo "  Phase 2a: Extract Activations — MAIN MODEL"
echo "  (all 57 blocks × 7 timesteps × 776 samples)"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --extract \
    --all-blocks \
    --output-dir "$OUT_MAIN" \
    --lora_checkpoint "$LORA_CKPT" \
    --metadata_csv "$METADATA" \
    --dataset_base "$DATASET" \
    --max_samples "$MAX_SAMPLES" \
    --steps 28

echo ""
echo "=============================================="
echo "  Phase 2b: Extract Activations — CONTROL"
echo "  (base FLUX + ControlNet, NO LoRA)"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --extract \
    --all-blocks \
    --output-dir "$OUT_CTRL" \
    --metadata_csv "$METADATA" \
    --dataset_base "$DATASET" \
    --max_samples "$MAX_SAMPLES" \
    --steps 28

echo ""
echo "=============================================="
echo "  Phase 3a: Train Probes — MAIN MODEL"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --train \
    --output-dir "$OUT_MAIN" \
    --pca-dim 0 \
    --ridge-lambda 0.1 \
    --max_samples "$MAX_SAMPLES"

echo ""
echo "=============================================="
echo "  Phase 3b: Train Probes — CONTROL"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --train \
    --output-dir "$OUT_CTRL" \
    --pca-dim 0 \
    --ridge-lambda 0.1 \
    --max_samples "$MAX_SAMPLES"

echo ""
echo "=============================================="
echo "  Phase 4: Done!"
echo "=============================================="
echo "Results:"
echo "  Main model:  ${OUT_MAIN}/probes/"
echo "  Control:     ${OUT_CTRL}/probes/"
echo ""
echo "To generate comparison document, run:"
echo "  python scripts/analysis/probing_analysis.py"
