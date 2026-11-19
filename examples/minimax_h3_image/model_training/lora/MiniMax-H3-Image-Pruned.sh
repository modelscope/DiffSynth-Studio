# Recommended: train MiniMax-H3-Image on images larger than 2048x2048 resolution (--min_pixels 4194304).
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# Optional: fuse the DeCFG training adapter into the DiT while training, for a better optimization landscape on this CFG-distilled base. Training only -- do not load it at inference.
# modelscope download --model DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter --include model.safetensors --local_dir ./models/DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter
#   --preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter/model.safetensors" \
#   --preset_lora_model "dit"

# Stage 1 of the split training: cache the deterministic preprocessing outputs (prompt embeddings and VAE
# latents) with the transformer offloaded to disk, so the caching step never holds the 66GB DiT on the
# accelerator.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 16777216 \
  --min_pixels 4194304 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-Pruned-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --offload_models "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"

# Stage 2 of the split training: train the LoRA on the cached dataset with the text encoder and the VAE
# offloaded to disk, so only the DiT and its LoRA adapters occupy the accelerator.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Image-Pruned-split-cache" \
  --data_file_keys "image" \
  --max_pixels 16777216 \
  --min_pixels 4194304 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-Pruned-split-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train" \
  --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"

# Image editing LoRA: the same two stages on an editing dataset, whose `metadata.json` carries an extra
# `edit_image` column (`image,prompt,edit_image`). `edit_image` is listed in both `--data_file_keys` and
# `--extra_inputs`, so the references are cached next to the target latents.
# Uncomment the blocks below and drop the two stages above to train on edits instead of text-to-image.
#
# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset
#
# accelerate launch examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 16777216 \
#   --min_pixels 4194304 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Pruned-Edit-split-cache" \
#   --lora_base_model "dit" \
#   --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
#   --lora_rank 32 \
# #   --task "sft:data_process" \
#   --offload_models "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"
#
# accelerate launch examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path "./models/train/MiniMax-H3-Image-Pruned-Edit-split-cache" \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 16777216 \
#   --min_pixels 4194304 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/bf16_table_pruned.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Pruned-Edit-split-lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
#   --lora_rank 32 \
# #   --task "sft:train" \
#   --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
