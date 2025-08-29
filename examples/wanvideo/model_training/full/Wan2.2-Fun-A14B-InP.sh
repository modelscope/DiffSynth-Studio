accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-InP:high_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-A14B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-InP:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Fun-A14B-InP_high_niose_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,end_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.2-Fun-A14B-InP:low_noise_model/diffusion_pytorch_model*.safetensors,PAI/Wan2.2-Fun-A14B-InP:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.2-Fun-A14B-InP:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Fun-A14B-InP_low_noise_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,end_image" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
# boundary corresponds to timesteps [0, 900]