export DIFFSYNTH_MODEL_BASE_PATH="/root/models"
# stage 1 (data process)
# accelerate launch examples/minimax_h3/model_training/train.py \
#   --dataset_base_path /mnt/nas3/sunyuzework/Diffutoon-2/data \
#   --dataset_metadata_path /mnt/nas3/sunyuzework/Diffutoon-2/data/stress_test-single.jsonl \
#   --data_file_keys "video,input_audio,vace_video" \
#   --extra_inputs "input_audio,vace_video" \
#   --max_pixels 1044480 \
#   --num_frames 39 \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:FL2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:FL2VA/audio_vae/model.safetensors" \
#   --learning_rate 1e-5 \
#   --num_epochs 1 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "./models/train/MiniMax-H3-VACE-full-split-cache" \
#   --trainable_models "vace" \
#   --use_gradient_checkpointing \
#   --task "sft:data_process"

# stage 2 (train)
accelerate launch --config_file /mnt/nas3/sunyuzework/myown/DiffSynth-Studio/examples/minimax_h3/model_training/full/accelerate_config_zero2.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-VACE-full-split-cache \
  --data_file_keys "video,input_audio,vace_video" \
  --extra_inputs "input_audio,vace_video" \
  --max_pixels 1044480 \
  --num_frames 39 \
  --dataset_repeat 1000 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/transformer/model*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 20 \
  --save_steps 50 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/MiniMax-H3-VACE-full" \
  --trainable_models "vace" \
  --vace_layers "0,8,16,24,32,40" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train" \
  --enable_tensorboard_log
