#!/bin/bash
# Phase 3c: Train FLUX ControlNet LoRA with Per-Frame Consistency Loss
# Fine-tunes from the best RAW checkpoint (epoch-15) using paired binary
# frames from 7 different temporal realizations of each scene.
#
# Loss: L = L_flow_match(F1) + λ · ||v_θ(z_t, t, F1) - v_θ(z_t, t, F2)||²

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

DATASET_BASE_PATH="/home/jw/engsci/thesis/spad/spad_dataset"
BASE_CHECKPOINT="./models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
OUTPUT_PATH="./models/train/FLUX-SPAD-LoRA-Consistency"
LOG_DIR="./logs/flux_spad_lora_consistency"

NUM_EPOCHS=30
LEARNING_RATE=5e-5
CONSISTENCY_WEIGHT=0.1
MAX_PIXELS=262144   # 512x512
LOG_FREQ=300
IMAGE_LOG_FREQ=1000

FP8_MODELS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"

MODEL_ID_WITH_ORIGIN_PATHS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,InstantX/FLUX.1-dev-Controlnet-Union-alpha:diffusion_pytorch_model.safetensors"

echo "=========================================="
echo "FLUX LoRA + Consistency Loss Training"
echo "=========================================="
echo "Base checkpoint: ${BASE_CHECKPOINT}"
echo "Consistency λ:   ${CONSISTENCY_WEIGHT}"
echo "Learning rate:   ${LEARNING_RATE}"
echo "Epochs:          ${NUM_EPOCHS}"
echo "Output:          ${OUTPUT_PATH}"
echo "Logs:            ${LOG_DIR}"
echo "=========================================="
echo ""

accelerate launch --config_file accelerate_config.yaml train_consistency.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_BASE_PATH}/metadata_train.csv" \
  --val_metadata_path "${DATASET_BASE_PATH}/metadata_val.csv" \
  --data_file_keys "image,controlnet_image" \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat 1 \
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
  --lora_checkpoint "${BASE_CHECKPOINT}" \
  --consistency_weight ${CONSISTENCY_WEIGHT}

echo ""
echo "=========================================="
echo "Consistency training complete!"
echo "Checkpoints: ${OUTPUT_PATH}"
echo "TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo "=========================================="
