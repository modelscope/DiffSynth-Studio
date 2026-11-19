# Recommended: train MiniMax-H3-Image on images larger than 2048x2048 resolution (--min_pixels 4194304).
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# Optional: fuse the DeCFG training adapter into the DiT while training, for a better optimization landscape on this CFG-distilled base. Training only -- do not load it at inference.
# modelscope download --model DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter --include model.safetensors --local_dir ./models/DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter
#   --preset_lora_path "./models/DiffSynth-Studio/MiniMax-H3-Image-TrainingAdapter/model.safetensors" \
#   --preset_lora_model "dit"

accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 16777216 \
  --min_pixels 4194304 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,DiffSynth-Studio/EntroPackPreQuants:models/MiniMax-H3/text_encoder_4bpp.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/vae_int8.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-EntroPack_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --dataset_num_workers 8 \
  --find_unused_parameters
