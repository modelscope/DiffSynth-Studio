#!/bin/bash
# Phase 4a-i: Fine-tune RAW_empty checkpoint on OD_03 data

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

DATASET_BASE_PATH="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_PATH="./models/train/FLUX-SPAD-LoRA-SceneAware-OD03-FT"
LOG_DIR="./logs/flux_spad_lora_scene_aware_od03_ft"

# Fine-tune from the RAW_empty best checkpoint
RAW_CHECKPOINT="./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"

NUM_EPOCHS=20
LEARNING_RATE=5e-5
DATASET_REPEAT=1
MAX_PIXELS=262144
LOG_FREQ=300
IMAGE_LOG_FREQ=1000

FP8_MODELS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"
MODEL_ID_WITH_ORIGIN_PATHS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"

# Use bits_RAW_OD_03 metadata
TRAIN_CSV="${DATASET_BASE_PATH}/bits_RAW_OD_03/metadata_train.csv"
VAL_CSV="${DATASET_BASE_PATH}/bits_RAW_OD_03/metadata_val.csv"

echo "=========================================="
echo "FLUX LoRA -- OD_03 Fine-tuned from RAW_empty"
echo "=========================================="
echo "Train CSV: ${TRAIN_CSV}"
echo "Val CSV:   ${VAL_CSV}"
echo "Resume:    ${RAW_CHECKPOINT}"
echo "Output:    ${OUTPUT_PATH}"
echo "=========================================="

accelerate launch --config_file accelerate_config.yaml train_lora.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${TRAIN_CSV}" \
  --val_metadata_path "${VAL_CSV}" \
  --data_file_keys "image,controlnet_image" \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "controlnet" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "controlnet_image" \
  --use_gradient_checkpointing \
  --fp8_models "${FP8_MODELS}" \
  --log_dir "${LOG_DIR}" \
  --log_freq ${LOG_FREQ} \
  --image_log_freq ${IMAGE_LOG_FREQ} \
  --lora_checkpoint "${RAW_CHECKPOINT}"

echo "Training complete! Checkpoints: ${OUTPUT_PATH}"
