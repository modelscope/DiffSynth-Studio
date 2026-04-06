#!/bin/bash
# Ablation: FLUX img2img + LoRA-on-DiT, NO ControlNet
#
# The SPAD conditioning enters only at inference via VAE-encoded input_image
# + partial denoising (denoising_strength < 1.0). During training, the DiT
# LoRA learns the target image distribution via standard flow matching SFT.
# No ControlNet is loaded or trained.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

DATASET_BASE_PATH="/home/jw/engsci/thesis/spad/spad_dataset"
OUTPUT_PATH="./models/train/FLUX-SPAD-LoRA-Img2Img-Ablation"
LOG_DIR="./logs/flux_spad_lora_img2img_ablation"

NUM_EPOCHS=20
LEARNING_RATE=1e-4
DATASET_REPEAT=1
MAX_PIXELS=262144   # 512x512
LOG_FREQ=300
IMAGE_LOG_FREQ=1000

# No ControlNet in FP8 list or model list
FP8_MODELS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors"

MODEL_ID_WITH_ORIGIN_PATHS="black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors"

echo "=========================================="
echo "FLUX LoRA on DiT -- img2img Ablation (NO ControlNet)"
echo "=========================================="
echo "Train CSV: ${DATASET_BASE_PATH}/metadata_train.csv"
echo "Val CSV:   ${DATASET_BASE_PATH}/metadata_val.csv"
echo "Output:    ${OUTPUT_PATH}"
echo "Logs:      ${LOG_DIR}"
echo "LoRA target: DiT (NOT ControlNet)"
echo "=========================================="
echo ""

accelerate launch --config_file accelerate_config.yaml train_lora.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_BASE_PATH}/metadata_train.csv" \
  --val_metadata_path "${DATASET_BASE_PATH}/metadata_val.csv" \
  --data_file_keys "image" \
  --max_pixels ${MAX_PIXELS} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --fp8_models "${FP8_MODELS}" \
  --log_dir "${LOG_DIR}" \
  --log_freq ${LOG_FREQ} \
  --image_log_freq ${IMAGE_LOG_FREQ}

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints: ${OUTPUT_PATH}"
echo "TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo ""
echo "Next: sweep denoising_strength with:"
echo "  bash run_img2img_ablation.sh"
echo "=========================================="
