#!/bin/bash
# ===================================================================
# Segmentation Probing Pipeline
# ===================================================================
# Generates SAM3 masks and runs segmentation probing on all conditions.
# Must be run AFTER the main extraction pipeline (run_probing_comprehensive.sh).
#
# Phase 1: Generate SAM3 masks (GPU, ~1-2s per image × 776 = ~20 min)
# Phase 2: Add targets to all output dirs
# Phase 3: Retrain probes with segmentation targets
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_DIFFSYNTH="/home/jw/miniconda3/envs/diffsynth/bin/python"
PYTHON_SAM3="/home/jw/miniconda3/envs/sam3/bin/python"

METADATA="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
DATASET="/home/jw/engsci/thesis/spad/spad_dataset"
VLM_OBJECTS="/home/jw/engsci/thesis/spad/spad_dataset/RGB/vlm_objects/objects_per_image.jsonl"
MAX_SAMPLES=776

# Output directories
OUT_MAIN="./probing_results_allblocks"
OUT_CTRL="./probing_results_control"
OUT_NOCN="./probing_results_no_cn"

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Generate SAM3 spatial segmentation masks
# ──────────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Phase 1: SAM3 Mask Generation"
echo "  (776 images × 24 objects per-image prompts)"
echo "=============================================="

# Check if SAM3 masks already exist in targets
HAS_SEG=$($PYTHON_DIFFSYNTH -c "
import json
with open('${OUT_MAIN}/targets.json') as f:
    t = json.load(f)
has = any(k.startswith('spatial_seg_') for k in t)
print('yes' if has else 'no')
" 2>/dev/null || echo "no")

if [ "$HAS_SEG" = "no" ]; then
    echo "  Running SAM3 segmentation + adding to all output dirs..."
    $PYTHON_SAM3 generate_seg_targets.py \
        --sam3-masks \
        --metadata_csv "$METADATA" \
        --dataset_base "$DATASET" \
        --vlm_objects "$VLM_OBJECTS" \
        --output-dir "$OUT_MAIN" "$OUT_CTRL" "$OUT_NOCN" \
        --max_samples "$MAX_SAMPLES" \
        --device cuda
else
    echo "  SAM3 masks already in targets.json, skipping."
    # But still add presence targets if missing
    $PYTHON_DIFFSYNTH generate_seg_targets.py \
        --presence-only \
        --metadata_csv "$METADATA" \
        --dataset_base "$DATASET" \
        --vlm_objects "$VLM_OBJECTS" \
        --output-dir "$OUT_MAIN" "$OUT_CTRL" "$OUT_NOCN" \
        --max_samples "$MAX_SAMPLES"
fi

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Retrain all probes (includes new segmentation targets)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Phase 2: Retrain probes with segmentation"
echo "=============================================="

for dir in "$OUT_MAIN" "$OUT_CTRL" "$OUT_NOCN"; do
    if [ -d "${dir}/activations" ]; then
        echo "  Training probes for $dir ..."
        $PYTHON_DIFFSYNTH linear_probing.py --train --output-dir "$dir" \
            --pca-dim 0 --ridge-lambda 0.1 --max_samples $MAX_SAMPLES
    else
        echo "  Skipping $dir (no activations yet)"
    fi
done

echo ""
echo "=============================================="
echo "  Segmentation probing complete!"
echo "=============================================="
echo "Results:"
echo "  Main model:  ${OUT_MAIN}/probes/"
echo "  Control:     ${OUT_CTRL}/probes/"
echo "  No-CN:       ${OUT_NOCN}/probes/"
