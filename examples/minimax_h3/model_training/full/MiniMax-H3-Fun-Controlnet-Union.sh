modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset

# Only the control branch is trained: the 50 base layers stay frozen, and `before_proj` / `after_proj`
# start at zero, so an untrained branch reproduces the base model exactly.

# Control - stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union/metadata.json \
  --data_file_keys "video,control_video,input_audio" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:FL2VA/processor/" \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-full-split-cache" \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# # Control - stage 2 (train)
accelerate launch --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Fun-Controlnet-Union-full-split-cache \
  --data_file_keys "video,control_video,input_audio" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:FL2VA/processor/" \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-full" \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"

# `--enable_inpaint` additionally feeds a random inpaint mask through the control branch, so one
# checkpoint serves both control and inpainting. It requires a control_in_dim that covers the mask
# channels, which the released MiniMax-H3-Fun-Controlnet-Union checkpoint has (49 channels).
# Inpaint - stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union/metadata.json \
  --data_file_keys "video,control_video,input_audio" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:FL2VA/processor/" \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-Inpaint-full-split-cache" \
  --trainable_models "controlnet" \
  --enable_inpaint \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# Inpaint - stage 2 (train)
accelerate launch --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Fun-Controlnet-Union-Inpaint-full-split-cache \
  --data_file_keys "video,control_video,input_audio" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 39 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:FL2VA/processor/" \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-Inpaint-full" \
  --trainable_models "controlnet" \
  --enable_inpaint \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
