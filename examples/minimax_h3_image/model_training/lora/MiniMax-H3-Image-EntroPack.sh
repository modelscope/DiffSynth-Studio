modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,DiffSynth-Studio/EntroPackPreQuants:models/MiniMax-H3/text_encoder_4bpp.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-EntroPack_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --dataset_num_workers 8 \
  --find_unused_parameters
