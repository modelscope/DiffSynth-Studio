set -e

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-FL2VA/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1: cache deterministic preprocessing outputs.
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image,end_image \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths 'MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/MiniMax-H3-FL2VA-split-cache \
  --lora_base_model dit \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task sft:data_process \
  --offload_models 'MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors'

# Stage 2: train LoRA from the cached dataset.
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-FL2VA-split-cache \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image,end_image \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths 'MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/MiniMax-H3-FL2VA-split \
  --lora_base_model dit \
  --lora_target_modules attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task sft:train \
  --offload_models 'MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors'
