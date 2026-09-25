#!/usr/bin/env bash
# ============================================================================
# Google Colab (free / T4 16 GB) one-shot environment setup.
# The Colab runtime already ships a CUDA-enabled torch; we only add the rest.
#
#   !bash scripts/setup_colab_t4.sh
# ============================================================================
set -euo pipefail

REPO_URL="${1:-https://github.com/haotemans/DiffSynth-Studio.git}"
BRANCH="${BRANCH:-v100-t4-16gb}"

echo "[1/3] runtime check"
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}')"

echo "[2/3] dependencies"
pip install -q "bitsandbytes>=0.45.0" "accelerate>=0.34.0" "peft>=0.12.0" \
  "transformers>=4.45.0" sentencepiece protobuf modelscope ftfy pandas einops

echo "[3/3] repo"
if [ ! -d DiffSynth-Studio ]; then
  git clone -b "$BRANCH" "$REPO_URL" || git clone "$REPO_URL"
fi
cd DiffSynth-Studio && git checkout "$BRANCH" 2>/dev/null || true
pip install -q -e . --no-deps

echo "=== done. Run:"
echo "    cd DiffSynth-Studio && bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh"
echo "T4 tip: keep MAX_PIXELS=262144 (512x512); 1024x1024 will OOM on 16 GB."