set -e

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "z_image/Z-Image/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1: cache deterministic preprocessing outputs.
accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/Z-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/Z-Image/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths 'Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/Z-Image_split_cache \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,to_out.0,w1,w2,w3 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --offload_models 'Tongyi-MAI/Z-Image:transformer/*.safetensors' \
  --task sft:data_process

# Stage 2: train LoRA from the cached dataset.
accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path ./models/train/Z-Image_split_cache \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths 'Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/Z-Image_split \
  --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,to_out.0,w1,w2,w3 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --offload_models 'Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors' \
  --task sft:train
