modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset

# Diffutoon-2 pairs a line-art ControlNet with a color ControlNet. The two ControlNets are trained
# separately: every run below loads exactly one ControlNet plus the sample control video, reusing the
# standard MiniMax-H3 training path. All runs share the MiniMax-H3-Fun-Controlnet-Union training data.

# ===== Line-Art ControlNet =====
# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union/metadata.json \
  --data_file_keys "video,input_audio,control_video" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Diffutoon-2-Line-Art-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# stage 2 (train), the frozen line-art ControlNet runs inside the diffusion model
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/Diffutoon-2-Line-Art-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,DiffSynth-Studio/Diffutoon-2:line-art_module.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Diffutoon-2-Line-Art_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"

# ===== Color ControlNet =====
# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Fun-Controlnet-Union/metadata.json \
  --data_file_keys "video,input_audio,control_video" \
  --extra_inputs "input_audio,control_video" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Diffutoon-2-Color-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# stage 2 (train), the frozen color ControlNet runs inside the diffusion model
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/Diffutoon-2-Color-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,DiffSynth-Studio/Diffutoon-2:color_module.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Diffutoon-2-Color_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
