modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-FL2VA/*" --local_dir ./data/diffsynth_example_dataset

# T2VA
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-text-encoder-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:video_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:audio_vae_nf4.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-T2VA-nf4" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters

# input_image / end_image take the first and last frame of the training video
# FL2VA
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-FL2VA/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,end_image" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-text-encoder-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:video_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:audio_vae_nf4.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-FL2VA-nf4" \
  --lora_base_model "dit" \
  --lora_target_modules "qkv_proj,out_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters
