#!/bin/bash
# FLUX ControlNet LoRA training for SPAD→RGB (ControlNet frozen in FP8, LoRA in bf16)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

# Paths
DATASET_BASE_PATH="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_PATH="./models/train/FLUX-SPAD-LoRA-On-ControlNet_TEST"
LOG_DIR="./logs/flux_spad_lora_on_controlnet_TEST"

# Training config
NUM_EPOCHS=40
LEARNING_RATE=1e-4
DATASET_REPEAT=1
MAX_PIXELS=262144   # 512x512
LOG_FREQ=300
IMAGE_LOG_FREQ=1000

# VRAM: keep DiT/text encoders + ControlNet in FP8 (all frozen)
FP8_MODELS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"

# Model list (includes ControlNet Union-alpha)
MODEL_ID_WITH_ORIGIN_PATHS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"

LORA_CHECKPOINT="/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/models/train/FLUX-SPAD-LoRA-On-ControlNet/epoch-32.safetensors"


echo "=========================================="
echo "FLUX LoRA on ControlNet (SPAD→RGB)"
echo "=========================================="
echo "Train CSV: ${DATASET_BASE_PATH}/metadata_train.csv"
echo "Val CSV:   ${DATASET_BASE_PATH}/metadata_val.csv"
echo "Output:    ${OUTPUT_PATH}"
echo "Logs:      ${LOG_DIR}"
echo "=========================================="
echo ""

accelerate launch --config_file accelerate_config.yaml train_lora.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_BASE_PATH}/metadata_train.csv" \
  --val_metadata_path "${DATASET_BASE_PATH}/metadata_val.csv" \
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
  --lora_checkpoint "${LORA_CHECKPOINT}"
  # ControlNet is frozen FP8; LoRA adapters are trained in bf16.

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints: ${OUTPUT_PATH}"
echo "TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo "=========================================="
