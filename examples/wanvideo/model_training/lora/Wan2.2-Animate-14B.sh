# 1*80G GPU cannot train Wan2.2-Animate-14B LoRA
# We tested on 8*80G GPUs
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_animate.csv \
  --data_file_keys "video,animate_pose_video,animate_face_video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-14B:Wan2.1_VAE.pth,Wan-AI/Wan2.2-Animate-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-14B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,animate_pose_video,animate_face_video" \
  --use_gradient_checkpointing_offload