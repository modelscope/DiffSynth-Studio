modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "stable_diffusion_xl/stable-diffusion-xl-base-1.0/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/stable_diffusion_xl/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/stable_diffusion_xl/stable-diffusion-xl-base-1.0 \
  --dataset_metadata_path data/diffsynth_example_dataset/stable_diffusion_xl/stable-diffusion-xl-base-1.0/metadata.csv \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "stabilityai/stable-diffusion-xl-base-1.0:text_encoder/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:text_encoder_2/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:unet/diffusion_pytorch_model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.unet." \
  --output_path "./models/train/stable-diffusion-xl-base-1.0_lora" \
  --lora_base_model "unet" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing
