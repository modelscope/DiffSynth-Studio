#!/bin/bash
# ===================================================================
# Comprehensive Linear Probing Experiment Pipeline
# ===================================================================
# Extends the base probing with:
#   1. ControlNet block probing (what does CN itself represent?)
#   2. Streaming spatial probing (per-token, full 776 samples)
#   3. No-ControlNet ablation (pure DiT baseline)
#   4. Main vs control comparisons
#
# Experiment conditions:
#   A. Main:    DiT + ControlNet(LoRA)   — trained model
#   B. Control: DiT + ControlNet(base)   — untrained ControlNet
#   C. No-CN:   DiT only, no ControlNet  — pure DiT baseline
#
# Each condition probes:
#   - DiT blocks (global mean-pooled): 57 blocks × 7 timesteps
#   - DiT blocks (spatial per-token):  10 sparse blocks × 7 timesteps
#   - CN blocks (global mean-pooled):  15 blocks × 7 timesteps (A, B only)
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LORA_CKPT="models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
METADATA="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
DATASET="/home/jw/engsci/thesis/spad/spad_dataset"
MULTISEED_DIR="./validation_outputs_multiseed"
MAX_SAMPLES=776

# Output directories
OUT_MAIN="./probing_results_allblocks"
OUT_CTRL="./probing_results_control"
OUT_NOCN="./probing_results_no_cn"

# Common extraction args
COMMON_ARGS="--metadata_csv $METADATA --dataset_base $DATASET --max_samples $MAX_SAMPLES --steps 28"

# ──────────────────────────────────────────────────────────────────────
# Phase 0: Ensure targets exist
# ──────────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Phase 0: Ensure targets exist"
echo "=============================================="
if [ ! -f "${OUT_MAIN}/targets.json" ]; then
    python scripts/analysis/linear_probing.py \
        --prepare-targets \
        --output-dir "$OUT_MAIN" \
        --multiseed-dir "$MULTISEED_DIR" \
        $COMMON_ARGS
fi
# Copy targets to all output dirs
for d in "$OUT_CTRL" "$OUT_NOCN"; do
    mkdir -p "$d"
    [ -f "$d/targets.json" ] || cp "${OUT_MAIN}/targets.json" "$d/targets.json"
done
echo "  Targets ready in all output dirs."

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Train probes on existing global activations (CPU only)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 1: Train global probes (CPU)"
echo "=============================================="

# Main model probes (if not already done)
if [ ! -f "${OUT_MAIN}/probes/probing_results.json" ]; then
    echo "  Training main model probes..."
    python scripts/analysis/linear_probing.py --train --output-dir "$OUT_MAIN" \
        --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES
fi

# Control probes (if not already done)
if [ ! -f "${OUT_CTRL}/probes/probing_results.json" ]; then
    echo "  Training control probes..."
    python scripts/analysis/linear_probing.py --train --output-dir "$OUT_CTRL" \
        --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES
fi

echo "  Global probes: done."

# ──────────────────────────────────────────────────────────────────────
# Phase 2a: Extract — Main model (LoRA) with CN hooks + spatial streaming
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 2a: Main model — CN probing + spatial streaming"
echo "  (LoRA, sparse DiT blocks, all CN blocks)"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --extract \
    --lora_checkpoint "$LORA_CKPT" \
    --output-dir "$OUT_MAIN" \
    --hook-controlnet \
    --spatial-streaming \
    --pca-dim 0 --ridge-lambda 0.1 \
    $COMMON_ARGS

# ──────────────────────────────────────────────────────────────────────
# Phase 2b: Extract — Control (no LoRA) with CN hooks + spatial streaming
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 2b: Control — CN probing + spatial streaming"
echo "  (No LoRA, sparse DiT blocks, all CN blocks)"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --extract \
    --output-dir "$OUT_CTRL" \
    --hook-controlnet \
    --spatial-streaming \
    --pca-dim 0 --ridge-lambda 0.1 \
    $COMMON_ARGS

# ──────────────────────────────────────────────────────────────────────
# Phase 2c: Extract — No-ControlNet ablation
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 2c: No-ControlNet ablation"
echo "  (Pure DiT, no ControlNet conditioning)"
echo "=============================================="
python scripts/analysis/linear_probing.py \
    --extract \
    --all-blocks \
    --output-dir "$OUT_NOCN" \
    --no-controlnet \
    --spatial-streaming \
    --pca-dim 0 --ridge-lambda 0.1 \
    $COMMON_ARGS

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Train CN probes + merge results
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 3: Train CN probes + merge all results"
echo "=============================================="

# Re-train probes on main (now includes CN activations + spatial streaming results)
python scripts/analysis/linear_probing.py --train --output-dir "$OUT_MAIN" \
    --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES

# Re-train probes on control (now includes CN activations + spatial streaming results)
python scripts/analysis/linear_probing.py --train --output-dir "$OUT_CTRL" \
    --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES

# Train no-CN probes
python scripts/analysis/linear_probing.py --train --output-dir "$OUT_NOCN" \
    --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES

# ──────────────────────────────────────────────────────────────────────
# Phase 4: Summary
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 4: All experiments complete!"
echo "=============================================="
echo "Results:"
echo "  Main model (LoRA):  ${OUT_MAIN}/probes/"
echo "  Control (base):     ${OUT_CTRL}/probes/"
echo "  No-ControlNet:      ${OUT_NOCN}/probes/"
echo ""
echo "To generate comparison analysis, run:"
echo "  python scripts/analysis/probing_analysis.py"
