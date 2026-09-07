modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset

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
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union-split-cache" \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# stage 2 (train)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Fun-Controlnet-Union-split-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors,PAI/MiniMax-H3-Fun-Controlnet-Union:MiniMax-H3-Fun-Controlnet-Union.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/MiniMax-H3-Fun-Controlnet-Union_full" \
  --trainable_models "controlnet" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
