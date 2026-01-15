accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /mnt/bucket/dawy/video_generation/two_stage_dataset \
  --dataset_metadata_path /mnt/bucket/dawy/video_generation/two_stage_dataset/metadata_vanilla_stage1.csv \
  --data_file_keys "video,vace_video,vace_reference_image,vace_video_mask" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "video,vace_video,vace_reference_image,vace_video_mask" \
  --use_gradient_checkpointing_offload \
  --val_dataset_base_path /mnt/bucket/dawy/video_generation/two_stage_dataset \
  --val_dataset_metadata_path /mnt/bucket/dawy/video_generation/two_stage_dataset/metadata_vanilla_stage1_val.csv \
  --val_data_file_keys "video,vace_video,vace_reference_image,vace_video_mask" \
  --val_batch_size 2 \
  --eval_every_n_epochs 1 \
  --eval_max_batches 50


