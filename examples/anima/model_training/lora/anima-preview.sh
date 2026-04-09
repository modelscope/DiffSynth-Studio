modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "anima/anima-preview/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/anima/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/anima/anima-preview \
  --dataset_metadata_path data/diffsynth_example_dataset/anima/anima-preview/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "circlestone-labs/Anima:split_files/diffusion_models/anima-preview.safetensors,circlestone-labs/Anima:split_files/text_encoders/qwen_3_06b_base.safetensors,circlestone-labs/Anima:split_files/vae/qwen_image_vae.safetensors" \
  --tokenizer_path "Qwen/Qwen3-0.6B:./" \
  --tokenizer_t5xxl_path "stabilityai/stable-diffusion-3.5-large:tokenizer_3/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/anima-preview_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing
