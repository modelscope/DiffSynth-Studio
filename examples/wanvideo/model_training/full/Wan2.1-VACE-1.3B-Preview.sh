accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "iic/VACE-Wan2.1-1.3B-Preview:diffusion_pytorch_model*.safetensors,iic/VACE-Wan2.1-1.3B-Preview:models_t5_umt5-xxl-enc-bf16.pth,iic/VACE-Wan2.1-1.3B-Preview:Wan2.1_VAE.pth" \
  --learning_rate 5e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B-Preview_full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload
# The learning rate is kept consistent with the settings in the original paper