modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-I2V-A14B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_high_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
# boundary corresponds to timesteps [900, 1000]

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_low_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
# boundary corresponds to timesteps [0, 900)
