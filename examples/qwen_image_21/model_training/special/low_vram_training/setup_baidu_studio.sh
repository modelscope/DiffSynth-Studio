#!/usr/bin/env bash
# ============================================================================
# Baidu AI Studio (Feijiang) one-shot environment setup.
# Target: V100 16 GB notebook, PaddlePaddle 3.x image (Python 3.10+).
# Paddle is left untouched; we only add a PyTorch stack next to it.
#
#   bash scripts/setup_baidu_studio.sh [REPO_URL]
# ============================================================================
set -euo pipefail

REPO_URL="${1:-https://github.com/haotemans/DiffSynth-Studio.git}"
BRANCH="${BRANCH:-v100-t4-16gb}"
WORK="${WORK:-/home/aistudio/work}"
PY="$(command -v python3 || command -v python)"

echo "[1/5] python: $($PY --version) at $PY"

# --- uv ---------------------------------------------------------------------
# Managed images (Baidu AI Studio) may put pip console scripts in a directory
# that is not on PATH, so drive uv through a wrapper that falls back to
# `python -m uv` (the uv wheel ships a __main__ entry point).
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1 && ! "$PY" -m uv --version >/dev/null 2>&1; then
  echo "[2/5] installing uv ..."
  "$PY" -m pip install uv -i https://mirrors.aliyun.com/pypi/simple/
fi
if command -v uv >/dev/null 2>&1; then
  UV() { uv "$@"; }
  echo "[2/5] uv: $(uv --version)"
else
  UV() { "$PY" -m uv "$@"; }
  echo "[2/5] uv via python -m: $("$PY" -m uv --version 2>&1)"
fi

# --- torch (Aliyun mirror, CUDA build picked from the driver) ---------------
CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo 11.8)"
if "$PY" -c "import sys; sys.exit(0 if float('$CUDA_VER') >= 12.1 else 1)"; then
  CUDA_TAG="cu121"
else
  CUDA_TAG="cu118"
fi
echo "[3/5] driver CUDA $CUDA_VER -> torch $CUDA_TAG from mirrors.aliyun.com"
UV pip install --python "$PY" \
  "torch==2.5.1+${CUDA_TAG}" "torchvision==0.20.1+${CUDA_TAG}" \
  --find-links "https://mirrors.aliyun.com/pytorch-wheels/${CUDA_TAG}/" \
  --index-url "https://mirrors.aliyun.com/pypi/simple/"

# --- training deps ----------------------------------------------------------
echo "[4/5] training dependencies ..."
UV pip install --python "$PY" \
  "bitsandbytes>=0.45.0" "accelerate>=0.34.0" "peft>=0.12.0" \
  "transformers>=4.45.0" sentencepiece protobuf safetensors \
  modelscope ftfy pandas einops "imageio[ffmpeg]" "numpy<2" \
  --index-url "https://mirrors.aliyun.com/pypi/simple/" \
  --extra-index-url "https://pypi.org/simple"

# --- repo -------------------------------------------------------------------
echo "[5/5] cloning ${REPO_URL} (${BRANCH}) ..."
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d DiffSynth-Studio ]; then
  git clone -b "$BRANCH" "$REPO_URL" || git clone "$REPO_URL"
fi
cd DiffSynth-Studio
git checkout "$BRANCH" 2>/dev/null || true
UV pip install --python "$PY" -e . --no-deps

"$PY" - <<'EOF'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
print(f"VRAM {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
import bitsandbytes, peft, accelerate, transformers
print(f"bitsandbytes {bitsandbytes.__version__} | peft {peft.__version__} | accelerate {accelerate.__version__} | transformers {transformers.__version__}")
EOF
echo "=== done. RESTART the notebook kernel, then run:"
echo "    cd $WORK/DiffSynth-Studio && bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh"