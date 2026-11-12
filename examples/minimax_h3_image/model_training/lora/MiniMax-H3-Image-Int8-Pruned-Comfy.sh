modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1 of the split training: cache the deterministic preprocessing outputs (prompt embeddings and VAE
# latents) with the transformer offloaded to disk, so the caching step never holds the 66GB DiT on the
# accelerator.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors,Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --offload_models "Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"

# Stage 2 of the split training: train the LoRA on the cached dataset with only the DiT loaded; the cached
# inputs already carry the text embeddings and VAE latents.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-split-cache" \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-split-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"

# Image editing LoRA: the same two stages on an editing dataset, whose `metadata.csv` carries an extra
# `edit_image` column (`image,prompt,edit_image`). `edit_image` is listed in both `--data_file_keys` and
# `--extra_inputs`, so the references are cached next to the target latents.
# Uncomment the blocks below and drop the two stages above to train on edits instead of text-to-image.
#
# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset
#
# accelerate launch examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image-Edit \
#   --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image-Edit/metadata.csv \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors,Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-Edit-split-cache" \
#   --lora_base_model "dit" \
#   --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing \
#   --task "sft:data_process" \
#   --offload_models "Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"
#
# accelerate launch examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-Edit-split-cache" \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "Comfy-Org/MiniMax-H3-Image:diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Int8-Pruned-Comfy-Edit-split-lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing \
#   --find_unused_parameters \
#   --task "sft:train"
