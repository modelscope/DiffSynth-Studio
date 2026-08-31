modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-FL2VA/*" --local_dir ./data/diffsynth_example_dataset

# T2VA - stage 1: encode text/video/audio and cache pipeline inputs (text encoder loaded here only)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors,Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-T2VA-int8-convrot-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# T2VA - stage 2: train LoRA on cached inputs (only the DiT is loaded, no text encoder)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-T2VA-int8-convrot-cache" \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-T2VA-int8-convrot" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
# Optional: fuse the DeCFG training adapter into the DiT while training, for a better optimization landscape on this CFG-distilled base. This DiT uses the ComfyUI qkv layout, so take the model_for_comfy_dit variant. Training only -- do not load it at inference.
# modelscope download --model DiffSynth-Studio/MiniMax-H3-TrainingAdapter --local_dir ./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter
#   --preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter/model_for_comfy_dit.safetensors" \
#   --preset_lora_model "dit"

# input_image / end_image take the first and last frame of the training video
# FL2VA - stage 1
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors,Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-int8-convrot-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# FL2VA - stage 2
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-FL2VA-int8-convrot-cache" \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-int8-convrot" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
