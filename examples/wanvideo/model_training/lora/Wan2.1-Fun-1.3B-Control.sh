export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /eva_data1/lynn/data/data_10k \
  --dataset_metadata_path /eva_data1/lynn/data/data_10k/metadata.csv \
  --data_file_keys "video,control_video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-1.3B-Control:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-1.3B-Control:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-1.3B-Control:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-1.3B-Control:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_frames 1 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-1.3B-Control_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video" \
  --gradient_accumulation_steps 4