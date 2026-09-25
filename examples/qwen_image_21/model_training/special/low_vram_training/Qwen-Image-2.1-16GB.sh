#!/usr/bin/env bash
# ============================================================================
# Qwen-Image-2.1 LoRA training for 16 GB GPUs (Baidu AI Studio V100 / Colab T4)
#
# Pipeline:
#   Stage 1 (sft:data_process): text encoder + VAE run once, results cached to disk.
#   Stage 2 (sft:train):        only the INT4-quantized DiT (+ FP16 LoRA) is loaded.
#
# Usage:
#   bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
#
# Environment knobs (all optional):
#   DATA_DIR      dataset root        (default ./data/diffsynth_example_dataset)
#   CACHE_DIR     stage-1 cache root  (default ./cache/qwen_image_21_16gb)
#   OUT_DIR       lora output root    (default ./models/train/Qwen-Image-2.1_lora_16gb)
#   MAX_PIXELS    training resolution (default 262144 = 512x512; 1048576 = 1024x1024)
#   LORA_RANK     LoRA rank           (default 32)
#   EPOCHS        stage-2 epochs      (default 5)
#   GRAD_ACCUM    gradient accum.     (default 4)
#   SKIP_STAGE1   set to 1 to reuse an existing cache
# ============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/diffsynth_example_dataset}"
CACHE_DIR="${CACHE_DIR:-./cache/qwen_image_21_16gb}"
OUT_DIR="${OUT_DIR:-./models/train/Qwen-Image-2.1_lora_16gb}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
LORA_RANK="${LORA_RANK:-32}"
EPOCHS="${EPOCHS:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

# V100 / T4: no BF16 tensor cores -> force FP16; Triton flex-attn is unstable on sm_70.
export DIFFSYNTH_DISABLE_FLEX_ATTN=1
FP16_FLAG="--force_fp16"

TRAIN_PY="examples/qwen_image_21/model_training/train.py"
DS="${DATA_DIR}/qwen_image_21/Qwen-Image-2.1"

# Modelscope is the default download source of DiffSynth-Studio; keep it.
if [ ! -f "${DS}/metadata.csv" ]; then
  echo "[setup] downloading example dataset ..."
  modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "qwen_image_21/Qwen-Image-2.1/*" --local_dir "${DATA_DIR}"
fi

COMMON_MODEL_ARGS="--model_id_with_origin_paths Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors"

if [ "${SKIP_STAGE1}" != "1" ]; then
  echo "[stage 1/2] caching text-encoder + VAE outputs to disk ..."
  # The DiT is never used in this stage, so it stays on disk (offload_models).
  accelerate launch "${TRAIN_PY}" \
    --dataset_base_path "${DS}" \
    --dataset_metadata_path "${DS}/metadata.csv" \
    --max_pixels "${MAX_PIXELS}" \
    --dataset_repeat 1 \
    ${COMMON_MODEL_ARGS} \
    --offload_models "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors" \
    --quant_options "Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors:bitsandbytes_nf4" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${CACHE_DIR}" \
    --lora_base_model "dit" \
    --lora_target_modules "" \
    --lora_rank "${LORA_RANK}" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    ${FP16_FLAG} \
    --task "sft:data_process"
else
  echo "[stage 1/2] skipped (SKIP_STAGE1=1), reusing ${CACHE_DIR}"
fi

echo "[stage 2/2] training LoRA on the INT4 DiT (text encoder / VAE stay on disk) ..."
accelerate launch "${TRAIN_PY}" \
  --dataset_base_path "${CACHE_DIR}" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${EPOCHS}" \
  ${COMMON_MODEL_ARGS} \
  --offload_models "Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --quant_options "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors:bitsandbytes_nf4" \
  --customized_optimizer "bitsandbytes.optim.AdamW8bit" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --find_unused_parameters \
  --enable_csv_log \
  ${FP16_FLAG} \
  --task "sft:train"

echo "[done] LoRA checkpoints: ${OUT_DIR}"
