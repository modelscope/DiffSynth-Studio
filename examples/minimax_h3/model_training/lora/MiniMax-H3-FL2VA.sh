modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-FL2VA/*" --local_dir ./data/diffsynth_example_dataset

# T2VA - stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-T2VA-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# T2VA - stage 2 (train)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-T2VA-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-T2VA-split" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"

# input_image / end_image take the first and last frame of the training video
# FL2VA - stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# FL2VA - stage 2 (train)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-FL2VA-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-split" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
