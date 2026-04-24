modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "stable_diffusion/stable-diffusion-v1-5/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/stable_diffusion/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/stable_diffusion/stable-diffusion-v1-5 \
  --dataset_metadata_path data/diffsynth_example_dataset/stable_diffusion/stable-diffusion-v1-5/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "AI-ModelScope/stable-diffusion-v1-5:text_encoder/model.safetensors,AI-ModelScope/stable-diffusion-v1-5:unet/diffusion_pytorch_model.safetensors,AI-ModelScope/stable-diffusion-v1-5:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --trainable_models "unet" \
  --remove_prefix_in_ckpt "pipe.unet." \
  --output_path "./models/train/stable-diffusion-v1-5_full" \
  --use_gradient_checkpointing
