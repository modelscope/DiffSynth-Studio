#!/usr/bin/env bash
# ============================================================================
# Google Colab (free / T4 16 GB) one-shot environment setup.
# The Colab runtime already ships a CUDA-enabled torch; we only add the rest.
#
#   !bash scripts/setup_colab_t4.sh
#
# This does NOT download the model weights (~31 GB). They come from
# modelscope.cn automatically on the first training run, into ./models inside
# the repo (Colab storage is ephemeral, so there is nothing worth persisting).
# Pull them up-front with a progress bar using:
#   !python -c "from modelscope import snapshot_download; \
#       snapshot_download('Qwen/Qwen-Image-2.1', local_dir='DiffSynth-Studio/models/Qwen/Qwen-Image-2.1')"
# ============================================================================
set -euo pipefail

REPO_URL="${1:-https://github.com/haotemans/DiffSynth-Studio.git}"
BRANCH="${BRANCH:-v100-t4-16gb}"
export GIT_TERMINAL_PROMPT=0

echo "[1/4] runtime check"
python -c "import torch; print(f'torch {torch.__version__} cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)}')"
echo "    VRAM $(python -c 'import torch;print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))') GB"
df -h /content | tail -1 | awk '{print "    disk: "$4" free on "$6"  (weights need ~31 GB)"}'

echo "[2/4] dependencies"
pip install -q "bitsandbytes>=0.45.0" "accelerate>=0.34.0" "peft>=0.12.0" \
  "transformers>=4.57.0" sentencepiece protobuf modelscope ftfy pandas einops

echo "[3/4] repo"
if [ ! -d DiffSynth-Studio ]; then
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" || git clone "$REPO_URL"
fi
cd DiffSynth-Studio
git checkout "$BRANCH" 2>/dev/null || true
pip install -q -e . --no-deps

echo "[4/4] probes (sm_75 T4)"
python - <<'EOF'
import importlib.util, torch, transformers
# The Qwen-Image-2.1 text encoder is Qwen3-VL; older transformers do not have it.
assert importlib.util.find_spec("transformers.models.qwen3_vl.modeling_qwen3_vl") is not None, \
    f"transformers {transformers.__version__} has no qwen3_vl; need >=4.57.0"
print(f"transformers {transformers.__version__}: Qwen3-VL ok")
import bitsandbytes, bitsandbytes.nn as bnb
lin = bnb.LinearNF4(64, 32, bias=False).to("cuda")
out = lin(torch.randn(2, 64, device="cuda", dtype=torch.float16))
print(f"NF4 forward ok: {tuple(out.shape)} {out.dtype}")
opt = bitsandbytes.optim.AdamW8bit([torch.zeros(1, requires_grad=True, device="cuda")], lr=1e-4)
print(f"AdamW8bit ok: {type(opt).__name__}")
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline
print(f"QwenImage21Pipeline import ok")
EOF

echo "=== done. Run:"
echo "    cd DiffSynth-Studio && bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh"
echo "T4 tips: keep MAX_PIXELS=262144 (512x512); 1024x1024 will OOM on 16 GB."
echo "         Colab shows ~13 GB of host RAM, so stage 1 auto-streams the text encoder"
echo "         (TE_DISK_OFFLOAD=1) and sets DIFFSYNTH_QUANT_STREAM=1."