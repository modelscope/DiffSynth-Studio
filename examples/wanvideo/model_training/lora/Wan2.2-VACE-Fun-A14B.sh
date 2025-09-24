accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.2-VACE-Fun-A14B_high_noise_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.2-VACE-Fun-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-VACE-Fun-A14B:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-VACE-Fun-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.2-VACE-Fun-A14B_low_noise_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
# boundary corresponds to timesteps [0, 900]