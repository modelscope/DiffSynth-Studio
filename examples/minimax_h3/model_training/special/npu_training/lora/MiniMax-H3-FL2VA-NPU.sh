#!/bin/bash
# MiniMax-H3 FL2VA LoRA training on Ascend NPU.
# Task: FL2VA (first/last frame conditioned).
# Two stages: 1) data process (cache text/vae latents), 2) LoRA train on DiT.
#
# Training mode (env):
#   ENABLE_FSDP=0 (default) : single card (accelerate --num_processes 1).
#   ENABLE_FSDP=1           : FSDP2 multi-card LoRA (--enable_fsdp, mutually exclusive with
#                             --enable_model_cpu_offload); process count = number of cards in
#                             ASCEND_RT_VISIBLE_DEVICES; free cards are auto-picked by default for BOTH single- and multi-card (AUTO_PICK_CARDS=0 disables); explicitly setting ASCEND_RT_VISIBLE_DEVICES skips auto-pick.
# Examples:
#   bash MiniMax-H3-FL2VA-NPU.sh
#   ENABLE_FSDP=1 ASCEND_RT_VISIBLE_DEVICES=12,13,14,15 bash MiniMax-H3-FL2VA-NPU.sh
#   ENABLE_FSDP=1 AUTO_PICK_CARDS=1 CARD_COUNT=4 bash MiniMax-H3-FL2VA-NPU.sh
set -e

# ---------- HCCL port range auto-pick ----------
# On shared hosts a fixed 62000-62100 may already be bound by other jobs (HCCL fails with
# EI0020 port already bound). Unless HCCL_NPU_SOCKET_PORT_RANGE is set explicitly, scan
# /proc/net/{tcp,tcp6,udp,udp6} for a contiguous free block and use that.
pick_hccl_port_range() {
python3 - "$1" "$2" "$3" <<'PY'
import sys
start, end, need = map(int, sys.argv[1:4])
used = set()
for f in ("/proc/net/tcp", "/proc/net/tcp6", "/proc/net/udp", "/proc/net/udp6"):
    try:
        with open(f) as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if len(parts) > 1 and ":" in parts[1]:
                    used.add(int(parts[1].split(":")[1], 16))
    except OSError:
        pass
run = 0
for p in range(start, end + 1):
    run = 0 if p in used else run + 1
    if run == need:
        print(f"{p - need + 1}-{p}")
        sys.exit(0)
sys.exit(1)
PY
}

# ---------- training mode & card selection ----------
ENABLE_FSDP=${ENABLE_FSDP:-0}
# ---------- card selection: auto-pick free cards (unless user sets cards or disables it) ----------
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && [ "${AUTO_PICK_CARDS:-1}" = "1" ]; then
    if [ "$ENABLE_FSDP" = "1" ]; then CARD_COUNT=${CARD_COUNT:-4}; else CARD_COUNT=${CARD_COUNT:-1}; fi
    FREE_MEM_THRESHOLD_MB=${FREE_MEM_THRESHOLD_MB:-4096}
    FREE_CARDS=()
    while read -r line; do
        id=$(awk '{print $3}' <<< "$line")
        used=$(grep -oE '[0-9]+ */ *[0-9]+' <<< "$line" | tail -n1 | cut -d'/' -f1 | tr -d ' ')
        [ -n "$id" ] && [ -n "$used" ] && [ "$used" -lt "$FREE_MEM_THRESHOLD_MB" ] && FREE_CARDS+=("$id")
    done < <(npu-smi info | grep -E '^\| *[0-9]+ +[0-9]+ +\| 0000')
    if [ "${#FREE_CARDS[@]}" -lt "$CARD_COUNT" ]; then
        echo "ERROR: free cards (HBM used<${FREE_MEM_THRESHOLD_MB}MB) = [${FREE_CARDS[*]}], need $CARD_COUNT" >&2
        exit 1
    fi
    export ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${FREE_CARDS[*]:0:$CARD_COUNT}")
    echo "auto-picked free cards (HBM used<${FREE_MEM_THRESHOLD_MB}MB): $ASCEND_RT_VISIBLE_DEVICES"
fi
if [ "$ENABLE_FSDP" = "1" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
    TRAIN_NPROC=$(awk -F',' '{print NF}' <<< "$ASCEND_RT_VISIBLE_DEVICES")
    if [ -z "${HCCL_NPU_SOCKET_PORT_RANGE:-}" ]; then
        HCCL_RANGE=$(pick_hccl_port_range 62000 65535 101) || { echo "ERROR: no free HCCL port range in 62000-65535" >&2; exit 1; }
        export HCCL_NPU_SOCKET_PORT_RANGE=$HCCL_RANGE
        echo "HCCL_NPU_SOCKET_PORT_RANGE auto-picked (free block): $HCCL_RANGE"
    else
        echo "HCCL_NPU_SOCKET_PORT_RANGE (user-set): $HCCL_NPU_SOCKET_PORT_RANGE"
    fi
    TRAIN_OFFLOAD_FLAGS="--enable_fsdp"
    TRAIN_UNUSED_FLAGS=""
    echo "ENABLE_FSDP=1: FSDP LoRA on ${TRAIN_NPROC} card(s): ${ASCEND_RT_VISIBLE_DEVICES}"
else
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
    TRAIN_NPROC=1
    TRAIN_OFFLOAD_FLAGS="--enable_model_cpu_offload"
    TRAIN_UNUSED_FLAGS="--find_unused_parameters"
    echo "ENABLE_FSDP=0: single-card LoRA (${ASCEND_RT_VISIBLE_DEVICES})"
fi

# IMPORTANT: on many-core hosts (e.g. 640 cores) torch defaults OMP threads to #cores;
# per-op fork-join overhead makes tiny CPU ops (optimizer on offloaded states,
# cache copies) ~180x slower. 1-16 threads is optimal.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch
export TOKENIZERS_PARALLELISM=false

WEIGHT_DIR=${WEIGHT_DIR:-/mnt/weight/MiniMax-H3/FL2VA}
DATA_DIR=${DATA_DIR:-data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA}
NUM_FRAMES=124
HEIGHT=480
WIDTH=832
LORA_TARGET_MODULES="attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
# Train-stage knobs (defaults = production full run).
# Quick smoke test: OUTPUT_BASE=./models/train/validate_run TRAIN_REPEAT=10 TRAIN_EPOCHS=1 bash MiniMax-H3-FL2VA-NPU.sh
TRAIN_REPEAT=${TRAIN_REPEAT:-100}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}
OUTPUT_BASE=${OUTPUT_BASE:-./models/train}
CACHE_BASE=${CACHE_BASE:-$OUTPUT_BASE}
SKIP_CACHE=${SKIP_CACHE:-0}
SILENT_ON_MISSING_AUDIO=${SILENT_ON_MISSING_AUDIO:-0}
SILENT_FLAG=""; [ "$SILENT_ON_MISSING_AUDIO" = "1" ] && SILENT_FLAG="--silent_on_missing_audio"

# `--model_paths` requires concrete files (no directories).
# Each JSON element is ONE model (a list of its shard files).
MODEL_PATHS_STAGE1=$(python -c "
import glob, json
files = [
    sorted(glob.glob('$WEIGHT_DIR/text_encoder/model-*.safetensors')),
    sorted(glob.glob('$WEIGHT_DIR/video_vae/source/model.safetensors')),
    sorted(glob.glob('$WEIGHT_DIR/audio_vae/model.safetensors')),
]
print(json.dumps(files))
")
MODEL_PATHS_STAGE2=$(python -c "
import glob, json
print(json.dumps([sorted(glob.glob('$WEIGHT_DIR/transformer/model-*.safetensors'))]))
")

# FL2VA - stage 1 (data process): first/last-frame conditioning (input_image/end_image).
if [ "$SKIP_CACHE" != "1" ]; then
accelerate launch --num_processes 1 examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$DATA_DIR" \
  --dataset_metadata_path "$DATA_DIR/metadata.csv" \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --dataset_repeat 1 \
  --model_paths "$MODEL_PATHS_STAGE1" \
  --processor_path "$WEIGHT_DIR/processor" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$CACHE_BASE/MiniMax-H3-FL2VA-npu-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  $SILENT_FLAG \
  --task "sft:data_process"
fi

# FL2VA - stage 2 (LoRA training)
accelerate launch --num_processes $TRAIN_NPROC examples/minimax_h3/model_training/train.py \
  --dataset_base_path $CACHE_BASE/MiniMax-H3-FL2VA-npu-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --dataset_repeat $TRAIN_REPEAT \
  --model_paths "$MODEL_PATHS_STAGE2" \
  --processor_path "$WEIGHT_DIR/processor" \
  --learning_rate 1e-4 \
  --num_epochs $TRAIN_EPOCHS \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_BASE/MiniMax-H3-FL2VA-npu-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  $TRAIN_OFFLOAD_FLAGS \
  --enable_csv_log \
  $TRAIN_UNUSED_FLAGS \
  --task "sft:train"
