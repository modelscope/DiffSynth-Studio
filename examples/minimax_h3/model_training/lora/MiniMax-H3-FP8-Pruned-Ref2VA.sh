modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Pruned-Ref2VA/*" --local_dir ./data/diffsynth_example_dataset

# Ref2VA - stage 1: encode reference/text/video/audio and cache (text encoder + processor loaded here only)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Pruned-Ref2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Pruned-Ref2VA/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-pruned-fp8-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# Ref2VA - stage 2: train LoRA on cached inputs (only the DiT is loaded)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Ref2VA-pruned-fp8-cache" \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Comfy-Org/MiniMax-H3:diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-pruned-fp8" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
