modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1 of the split training: cache the deterministic preprocessing outputs with the transformer offloaded to disk.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-full-split-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --offload_models "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"

# Stage 2 of the split training: full fine-tuning on the cached dataset with ZeRO-3, text encoder and VAE offloaded.
accelerate launch --config_file examples/minimax_h3_image/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Image-full-split-cache \
  --data_file_keys "image" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-full-split" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:train" \
  --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"

# Image editing full fine-tuning: the same two stages on an editing dataset, whose `metadata.csv` carries an
# extra `edit_image` column (`image,prompt,edit_image`). `edit_image` is listed in both `--data_file_keys` and
# `--extra_inputs`, so the references are cached next to the target latents.
# Uncomment the blocks below and drop the two stages above to train on edits instead.
#
# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset
#
# accelerate launch examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image-Edit \
#   --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image-Edit/metadata.csv \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --height 1024 \
#   --width 1024 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Edit-full-split-cache" \
#   --trainable_models "dit" \
#   --use_gradient_checkpointing \
#   --task "sft:data_process" \
#   --offload_models "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"
#
# accelerate launch --config_file examples/minimax_h3_image/model_training/full/accelerate_config_zero3.yaml \
#   examples/minimax_h3_image/model_training/train.py \
#   --dataset_base_path ./models/train/MiniMax-H3-Image-Edit-full-split-cache \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --height 1024 \
#   --width 1024 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-5 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/MiniMax-H3-Image-Edit-full-split" \
#   --trainable_models "dit" \
#   --use_gradient_checkpointing \
#   --task "sft:train" \
#   --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
