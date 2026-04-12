#!/bin/bash
# LoRA training for JoyAI-Image-Edit — full validation run (5 epochs, 1024x1024)

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
  --max_pixels 1048576 \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth,jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/JoyAI-Image-Edit_lora" \
  --trainable_models "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --initialize_model_on_cpu \
  --find_unused_parameters
