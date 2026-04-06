#!/bin/bash
# FLUX ControlNet LoRA Training for SPAD→RGB
# Based on examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Upscaler.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

# Paths
DATASET_BASE_PATH="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_PATH="./models/train/FLUX-SPAD-ControlNet-LoRA"
LOG_DIR="./logs/flux_spad_controlnet_lora"


# Training config
NUM_EPOCHS=40
LEARNING_RATE=1e-4
LORA_RANK=32
DATASET_REPEAT=1  # Each epoch = 1 pass through dataset
MAX_PIXELS=262144  # ~512x512
LOG_FREQ=300       # Log scalars every 300 steps
IMAGE_LOG_FREQ=1000  # Log images every 1000 steps

# VRAM optimization: Offload T5 encoder to CPU/disk (the largest model)
FP8_MODELS="black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:flux1-dev.safetensors"
OFFLOAD_MODELS="black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors"

# Resume from checkpoint (uncomment to continue training)
# LORA_CHECKPOINT="./models/train/FLUX-SPAD-ControlNet-LoRA/epoch-9.safetensors"

echo "=========================================="
echo "FLUX ControlNet LoRA - SPAD to RGB"
echo "=========================================="
echo "Dataset: ${DATASET_BASE_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Logs: ${LOG_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="
echo ""

# Launch training with accelerate (using config file for bf16 mixed precision)
accelerate launch --config_file accelerate_config.yaml train_lora.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_BASE_PATH}/metadata_train.csv" \
  --val_metadata_path "${DATASET_BASE_PATH}/metadata_val.csv" \
  --data_file_keys "image,controlnet_image" \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "controlnet_image" \
  --align_to_opensource_format \
  --use_gradient_checkpointing \
  --fp8_models "${FP8_MODELS}" \
  --log_dir "${LOG_DIR}" \
  --log_freq ${LOG_FREQ} \
  --image_log_freq ${IMAGE_LOG_FREQ}
  # --lora_checkpoint "${LORA_CHECKPOINT}"  # Uncomment to resume

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints: ${OUTPUT_PATH}"
echo "TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo "=========================================="
