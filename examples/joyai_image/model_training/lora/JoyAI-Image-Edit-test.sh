#!/bin/bash
# Minimal LoRA training smoke test for JoyAI-Image-Edit
# Runs only 1 epoch with minimal data to verify training pipeline works

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFFSYNTH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$DIFFSYNTH_ROOT"

export CONDA_DEFAULT_ENV=joyai-image-diffsynth
export PATH="/root/miniconda3/envs/joyai-image-diffsynth/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$DIFFSYNTH_ROOT:$PYTHONPATH"

accelerate launch --config_file examples/joyai_image/model_training/lora/accelerate_config_single_gpu.yaml \
  examples/joyai_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv \
  --data_file_keys "image,input_image" \
  --extra_inputs "input_image" \
  --max_pixels 262144 \
  --height 256 \
  --width 256 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth,jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/JoyAI-Image-Edit_lora_test" \
  --trainable_models "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 4 \
  --use_gradient_checkpointing \
  --initialize_model_on_cpu \
  --find_unused_parameters
