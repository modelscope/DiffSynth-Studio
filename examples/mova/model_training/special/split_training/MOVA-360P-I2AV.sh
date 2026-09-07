modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "mova/MOVA-360P-I2AV/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1: cache deterministic preprocessing outputs.
accelerate launch examples/mova/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/mova/MOVA-360P-I2AV \
  --dataset_metadata_path data/diffsynth_example_dataset/mova/MOVA-360P-I2AV/metadata.csv \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image \
  --height 352 \
  --width 640 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths 'openmoss/MOVA-360p:video_dit/diffusion_pytorch_model-*.safetensors,openmoss/MOVA-360p:audio_dit/diffusion_pytorch_model.safetensors,openmoss/MOVA-360p:dual_tower_bridge/diffusion_pytorch_model.safetensors,openmoss/MOVA-720p:audio_vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.1_VAE.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:models_t5_umt5-xxl-enc-bf16.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt pipe.video_dit. \
  --output_path ./models/train/MOVA-360p-I2AV_high_noise_split_cache \
  --lora_base_model video_dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --use_gradient_checkpointing \
  --offload_models 'openmoss/MOVA-360p:video_dit/diffusion_pytorch_model-*.safetensors,openmoss/MOVA-360p:audio_dit/diffusion_pytorch_model.safetensors,openmoss/MOVA-360p:dual_tower_bridge/diffusion_pytorch_model.safetensors' \
  --task sft:data_process

# Stage 2: train LoRA from the cached dataset.
accelerate launch examples/mova/model_training/train.py \
  --dataset_base_path ./models/train/MOVA-360p-I2AV_high_noise_split_cache \
  --data_file_keys video,input_audio \
  --extra_inputs input_audio,input_image \
  --height 352 \
  --width 640 \
  --num_frames 121 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths 'openmoss/MOVA-360p:video_dit/diffusion_pytorch_model-*.safetensors,openmoss/MOVA-360p:audio_dit/diffusion_pytorch_model.safetensors,openmoss/MOVA-360p:dual_tower_bridge/diffusion_pytorch_model.safetensors,openmoss/MOVA-720p:audio_vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.1_VAE.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:models_t5_umt5-xxl-enc-bf16.safetensors' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt pipe.video_dit. \
  --output_path ./models/train/MOVA-360p-I2AV_high_noise_split \
  --lora_base_model video_dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --use_gradient_checkpointing \
  --offload_models openmoss/MOVA-720p:audio_vae/diffusion_pytorch_model.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:Wan2.1_VAE.safetensors,DiffSynth-Studio/Wan-Series-Converted-Safetensors:models_t5_umt5-xxl-enc-bf16.safetensors \
  --task sft:train
