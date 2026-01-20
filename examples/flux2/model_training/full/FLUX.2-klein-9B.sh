# This script is tested on 8*A100
accelerate launch --config_file examples/flux2/model_training/full/accelerate_config.yaml examples/flux2/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-klein-9B_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing

# Edit
# accelerate launch --config_file examples/flux2/model_training/full/accelerate_config.yaml examples/flux2/model_training/train.py \
#   --dataset_base_path data/example_image_dataset \
#   --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
#   --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/FLUX.2-klein-9B_full" \
#   --trainable_models "dit" \
#   --use_gradient_checkpointing
