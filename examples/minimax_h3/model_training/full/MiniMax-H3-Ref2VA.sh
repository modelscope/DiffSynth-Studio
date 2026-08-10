modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Ref2VA/*" --local_dir ./data/diffsynth_example_dataset

# stage 1 (data process)
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-full-split-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# stage 2 (train)
accelerate launch --config_file examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Ref2VA-full-split-cache \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
