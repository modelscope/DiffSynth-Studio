#!/bin/bash
# Run validation on the validation set

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

LORA_CHECKPOINT="/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/models/train/FLUX-SPAD-LoRA-On-ControlNet/epoch-34.safetensors"
METADATA_CSV="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv"
OUTPUT_DIR="./validation_results_lora_on_controlnet_seed_67"
MAX_SAMPLES=50  # Validate on first 50 samples (or remove for all)
SEED=67
# CONTROLNET_FP8="--controlnet_fp8"  # Enable only if you want FP8 controlnet (can reduce quality)
CONTROLNET_FP8=""

echo "=========================================="
echo "FLUX LoRA Validation"
echo "=========================================="
echo "Checkpoint: ${LORA_CHECKPOINT}"
echo "Validation set: ${METADATA_CSV}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python validate_lora.py \
  --lora_checkpoint "${LORA_CHECKPOINT}" \
  --lora_target controlnet \
  --metadata_csv "${METADATA_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --steps 50 \
  --cfg_scale 1.0 \
  --embedded_guidance 3.5 \
  --denoising_strength 1.0 \
  --height 512 \
  --width 512 \
  --controlnet_scale 1.0 \
  --processor_id gray \
  ${CONTROLNET_FP8} \
  --max_samples ${MAX_SAMPLES} \
  --seed ${SEED}

echo ""
echo "✅ Done! Check ${OUTPUT_DIR}"
