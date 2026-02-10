accelerate launch --num_machines 1 --num_processes 2 \
  --config_file mine/train/configs/accelerate-wan2_1-i2v-14b-480p.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --initialize_model_on_cpu \
  --find_unused_parameters \
  --task sft \
  --output_path "./train/train/Wan2.1-I2V-14B-480P_full" \
  --save_steps 1000