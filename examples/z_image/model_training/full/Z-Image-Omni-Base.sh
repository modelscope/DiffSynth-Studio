# This example is tested on 8*A100
# Text to image training
accelerate launch --config_file examples/z_image/model_training/full/accelerate_config.yaml examples/z_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 400 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Omni-Base:transformer/*.safetensors,Tongyi-MAI/Z-Image-Omni-Base:siglip/model.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Omni-Base_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --dataset_num_workers 8

# Image(s) to image training
# accelerate launch --config_file examples/z_image/model_training/full/accelerate_config.yaml examples/z_image/model_training/train.py \
#   --dataset_base_path data/example_image_dataset \
#   --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 400 \
#   --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Omni-Base:transformer/*.safetensors,Tongyi-MAI/Z-Image-Omni-Base:siglip/model.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Z-Image-Omni-Base_full_edit" \
#   --trainable_models "dit" \
#   --use_gradient_checkpointing \
#   --find_unused_parameters \
#   --dataset_num_workers 8
