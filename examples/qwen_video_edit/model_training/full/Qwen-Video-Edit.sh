modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_video_edit/Qwen-Video-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/qwen_video_edit/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit/metadata.json \
  --data_file_keys "video,input_video" \
  --height 640 \
  --width 384 \
  --num_frames 45 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "yunpeng1998/Qwen-Video-Edit:360P/step-30000.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Video-Edit_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --zero_cond_t \
  --find_unused_parameters
