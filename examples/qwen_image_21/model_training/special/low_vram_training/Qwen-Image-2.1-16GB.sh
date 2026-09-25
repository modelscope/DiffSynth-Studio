#!/usr/bin/env bash
# ============================================================================
# Qwen-Image-2.1 LoRA training for 16 GB GPUs (Baidu AI Studio V100 / Colab T4)
#
# Pipeline:
#   Stage 1 (sft:data_process): NF4 text encoder + VAE run once; their outputs
#                               (prompt embeds + latents) are cached to disk.
#   Stage 2 (sft:train):        only the INT4 DiT (+ FP16 LoRA) is loaded.
#
# Usage:
#   bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
#
# Environment knobs (all optional):
#   DATA_DIR        dataset root        (default ./data/diffsynth_example_dataset)
#   DS_SUBDIR       sub-path under DATA_DIR holding metadata.csv (default
#                   qwen_image_21/Qwen-Image-2.1 for the example dataset, "." for
#                   a folder you prepared yourself)
#   CACHE_DIR       stage-1 cache root  (default ./cache/qwen_image_21_16gb)
#   OUT_DIR         lora output root    (default ./models/train/Qwen-Image-2.1_lora_16gb)
#   MAX_PIXELS      training resolution (default 262144 = 512x512; 1048576 = 1024x1024)
#   LORA_RANK       LoRA rank           (default 32)
#   EPOCHS          stage-2 epochs      (default 5)
#   GRAD_ACCUM      gradient accum.     (default 4)
#   GC_BLOCKS       checkpoint first N of 32 blocks (-1 = all)
#   NUM_WORKERS     dataloader workers  (default 4)
#   SKIP_STAGE1     1 = never run stage 1 (reuse the cache)
#   FORCE_STAGE1    1 = rebuild the cache even when one exists
#   TE_DISK_OFFLOAD auto|0|1 (default auto) = in stage 1, stream the FP16 text
#                   encoder straight from the safetensors file instead of
#                   NF4-quantizing it. `auto` picks streaming when host RAM is
#                   under 24 GB (online NF4 keeps an FP16 copy in RAM).
#   NO_NF4          1 = bitsandbytes is unusable on this GPU: stage 1 streams the
#                   text encoder from disk and stage 2 trains with
#                   --enable_model_cpu_offload (no quantization at all). Slow but
#                   runs on anything with enough host RAM.
#   FREE_TEXT_ENCODER 1 = delete the 16 GB text encoder once stage 1 is cached
#   DRY_RUN         1 = print the commands instead of running them
#
# Weights (~31 GB total) are NOT bundled and NOT downloaded by the setup script.
# Each stage pulls only what it needs from modelscope.cn (CN-reachable, no GitHub):
#   stage 1 -> text_encoder/ (16.3 GB) + vae/ (1.3 GB) + processor/ (~15 MB)
#   stage 2 -> transformer/  (13.3 GB)
# into ${DIFFSYNTH_MODEL_BASE_PATH:-./models}. Fetch them up front with progress:
#   PRELOAD_MODELS=1 bash .../setup_baidu_studio.sh
# ============================================================================
set -euo pipefail

# On managed images (Baidu AI Studio) the system site-packages is read-only, so
# setup_baidu_studio.sh builds a dedicated uv venv plus a small env file with
# persistent cache locations. Pick both up when they exist.
VENV="${VENV:-/home/aistudio/work/venv-diffsynth}"
BAIDU_ENV="${BAIDU_ENV:-/home/aistudio/work/diffsynth_env.sh}"
if [ -f "${BAIDU_ENV}" ]; then
  . "${BAIDU_ENV}"
fi
if [ -f "${VENV}/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  . "${VENV}/bin/activate"
  echo "[env] using venv ${VENV} ($(python -V 2>&1))"
fi
USER_BASE="$(python3 -m site --user-base 2>/dev/null || echo "${HOME}/.local")"
export PATH="${HOME}/.local/bin:${USER_BASE}/bin:${PATH}"

# Fail fast with a useful message instead of an import traceback deep inside accelerate.
if ! python -c 'import torch, accelerate' >/dev/null 2>&1; then
  echo "[env] ERROR: torch/accelerate not importable with '$(command -v python || echo python)'." >&2
  echo "[env]       run the setup script first:" >&2
  echo "[env]       bash examples/qwen_image_21/model_training/special/low_vram_training/setup_baidu_studio.sh" >&2
  echo "[env]       (Colab: setup_colab_t4.sh)" >&2
  exit 1
fi
echo "[env] python $(python -V 2>&1) @ $(command -v python)"

DRY_RUN="${DRY_RUN:-0}"
RUN=()
if [ "${DRY_RUN}" = "1" ]; then RUN=(echo "[dry-run]"); fi

MODEL_DIR="${DIFFSYNTH_MODEL_BASE_PATH:-./models}/Qwen/Qwen-Image-2.1"
TE_DIR="${MODEL_DIR}/text_encoder"
report_dir() {  # $1 = label, $2 = dir, $3 = note
  if [ -d "$2" ]; then
    printf '[env]   %-13s %-8s %s\n' "$1" "$(du -sh "$2" 2>/dev/null | cut -f1 || echo '?')" "cached"
  else
    printf '[env]   %-13s %-8s %s\n' "$1" "-" "missing -> modelscope.cn $3 on first use"
  fi
}
echo "[env] weights: ${MODEL_DIR}"
report_dir "transformer" "${MODEL_DIR}/transformer" "downloads 13.3 GB (stage 2)"
report_dir "text_encoder" "$TE_DIR" "downloads 16.3 GB (stage 1)"
report_dir "vae" "${MODEL_DIR}/vae" "downloads 1.3 GB (stage 1)"
echo "[env] disk   : $(df -h . | tail -1 | awk '{print $4" free on "$6}')"

DATA_DIR="${DATA_DIR:-./data/diffsynth_example_dataset}"
DS_SUBDIR="${DS_SUBDIR:-qwen_image_21/Qwen-Image-2.1}"
CACHE_DIR="${CACHE_DIR:-./cache/qwen_image_21_16gb}"
OUT_DIR="${OUT_DIR:-./models/train/Qwen-Image-2.1_lora_16gb}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
LORA_RANK="${LORA_RANK:-32}"
EPOCHS="${EPOCHS:-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GC_BLOCKS="${GC_BLOCKS:--1}"   # selective gradient checkpointing: only first N of 32 blocks; -1 = all
NUM_WORKERS="${NUM_WORKERS:-4}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
FORCE_STAGE1="${FORCE_STAGE1:-0}"
NO_NF4="${NO_NF4:-0}"

RAM_GB="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
LOW_RAM=0
if [ "${RAM_GB:-0}" -gt 0 ] && [ "${RAM_GB}" -lt 24 ]; then LOW_RAM=1; fi
echo "[env] host RAM: ${RAM_GB:-?} GB (low-RAM mode: ${LOW_RAM})"

# Online NF4 quantization materializes the FP16 checkpoint in host memory before
# packing it (16.3 GB for the text encoder, 13.3 GB for the DiT). On a small
# runtime, stream it block by block instead: same weights, ~2 GB peak.
if [ "${LOW_RAM}" = "1" ] && [ -z "${DIFFSYNTH_QUANT_STREAM:-}" ]; then
  export DIFFSYNTH_QUANT_STREAM=1
  echo "[env] DIFFSYNTH_QUANT_STREAM=1 (quantize block by block straight from disk)"
fi

if [ "${NO_NF4}" = "1" ]; then
  TE_DISK_OFFLOAD=1
  echo "[env] NO_NF4=1 -> no bitsandbytes: stage 1 streams the text encoder from disk,"
  echo "[env]            stage 2 trains with --enable_model_cpu_offload."
elif [ "${TE_DISK_OFFLOAD:-auto}" = "auto" ]; then
  TE_DISK_OFFLOAD="${LOW_RAM}"
  echo "[env] TE_DISK_OFFLOAD=${TE_DISK_OFFLOAD} (auto)"
fi

# V100 / T4: no BF16 tensor cores -> force FP16; Triton flex-attn is unstable on sm_70.
export DIFFSYNTH_DISABLE_FLEX_ATTN=1
FP16_FLAG="--force_fp16"

TRAIN_PY="examples/qwen_image_21/model_training/train.py"
DS="${DATA_DIR}/${DS_SUBDIR}"

# Each stage lists only the models it actually runs, so nothing extra is
# downloaded or loaded: stage 1 never touches the DiT, stage 2 never touches the
# text encoder / VAE (it reads their cached outputs instead).
STAGE1_MODELS="Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors"
STAGE1_EXTRA=()
STAGE2_EXTRA=()
if [ "${TE_DISK_OFFLOAD:-0}" = "1" ]; then
  # Stream the FP16 text encoder from the safetensors file, layer by layer.
  STAGE1_EXTRA+=(--offload_models "Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors")
else
  STAGE1_EXTRA+=(--quant_options "Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors:bitsandbytes_nf4")
fi
STAGE2_MODELS="Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors"
if [ "${NO_NF4}" = "1" ]; then
  STAGE2_EXTRA+=(--enable_model_cpu_offload --initialize_model_on_cpu)
else
  STAGE2_EXTRA+=(--quant_options "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors:bitsandbytes_nf4" \
                 --customized_optimizer "bitsandbytes.optim.AdamW8bit")
fi

# Modelscope is the default download source of DiffSynth-Studio; keep it.
if [ ! -f "${DS}/metadata.csv" ]; then
  echo "[setup] downloading example dataset ..."
  ${RUN[@]+"${RUN[@]}"} modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "qwen_image_21/Qwen-Image-2.1/*" --local_dir "${DATA_DIR}"
fi

cache_ready() { [ -d "$1" ] && [ -n "$(find "$1" -name '*.pth' -print -quit 2>/dev/null)" ]; }

if [ "${SKIP_STAGE1}" = "1" ]; then
  echo "[stage 1/2] skipped (SKIP_STAGE1=1)"
  if [ "${DRY_RUN}" != "1" ] && ! cache_ready "${CACHE_DIR}"; then
    echo "[stage 1/2] ERROR: no *.pth cache under ${CACHE_DIR}; unset SKIP_STAGE1 and re-run." >&2
    exit 1
  fi
elif [ "${DRY_RUN}" != "1" ] && cache_ready "${CACHE_DIR}" && [ "${FORCE_STAGE1}" != "1" ]; then
  echo "[stage 1/2] cache present at ${CACHE_DIR} ($(find "${CACHE_DIR}" -name '*.pth' | wc -l) items) -> skipped"
  echo "[stage 1/2] set FORCE_STAGE1=1 to rebuild it (needed after changing MAX_PIXELS or the dataset)."
else
  echo "[stage 1/2] caching text-encoder + VAE outputs to disk ..."
  ${RUN[@]+"${RUN[@]}"} accelerate launch "${TRAIN_PY}" \
    --dataset_base_path "${DS}" \
    --dataset_metadata_path "${DS}/metadata.csv" \
    --max_pixels "${MAX_PIXELS}" \
    --dataset_repeat 1 \
    --dataset_num_workers "${NUM_WORKERS}" \
    --model_id_with_origin_paths "${STAGE1_MODELS}" \
    ${STAGE1_EXTRA[@]+"${STAGE1_EXTRA[@]}"} \
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
  if [ "${DRY_RUN}" != "1" ] && ! cache_ready "${CACHE_DIR}"; then
    echo "[stage 1/2] ERROR: stage 1 finished but produced no cache under ${CACHE_DIR}." >&2
    exit 1
  fi
fi

# Stage 2 never reads the text encoder, so its 16 GB can be handed back to the
# disk (AI Studio quotas are tight). Deletion is opt-in.
if [ -d "${TE_DIR}" ]; then
  if [ "${FREE_TEXT_ENCODER:-0}" = "1" ]; then
    echo "[tips] FREE_TEXT_ENCODER=1 -> removing ${TE_DIR} ($(du -sh "${TE_DIR}" 2>/dev/null | cut -f1 || echo 16G))"
    ${RUN[@]+"${RUN[@]}"} rm -rf "${TE_DIR}"
  elif [ "${KEEP_TEXT_ENCODER:-0}" != "1" ]; then
    echo "[tips] the text encoder at ${TE_DIR} is not needed by stage 2."
    echo "[tips]   free the disk:  rm -rf \"${TE_DIR}\"   (or set FREE_TEXT_ENCODER=1)"
    echo "[tips]   re-download:    PRELOAD_MODELS=1 PRELOAD_STAGE=stage1 bash setup_baidu_studio.sh"
    echo "[tips]   silence this:   KEEP_TEXT_ENCODER=1"
  fi
fi

echo "[stage 2/2] training LoRA on the INT4 DiT ..."
${RUN[@]+"${RUN[@]}"} accelerate launch "${TRAIN_PY}" \
  --dataset_base_path "${CACHE_DIR}" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${EPOCHS}" \
  --dataset_num_workers "${NUM_WORKERS}" \
  --model_id_with_origin_paths "${STAGE2_MODELS}" \
  ${STAGE2_EXTRA[@]+"${STAGE2_EXTRA[@]}"} \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing \
  --gradient_checkpointing_blocks "${GC_BLOCKS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --find_unused_parameters \
  --enable_csv_log \
  ${FP16_FLAG} \
  --task "sft:train"

echo "[done] LoRA checkpoints: ${OUT_DIR}"
echo "[done] hot-load at inference with pipe.load_lora(pipe.dit, '<epoch-N>.safetensors'); never merge."