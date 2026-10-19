modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux3/flux-3-image-klein-base/*" --local_dir ./data/diffsynth_example_dataset

# The training command below downloads the access-request-only model `black-forest-labs/flux-3-image-klein-base`.
# 1. Request access at https://modelscope.cn/models/black-forest-labs/flux-3-image-klein-base
# 2. Log in via CLI: `modelscope login --token ms-xxx`. Get your token at https://modelscope.cn/my/settings/token

accelerate launch examples/flux3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux3/flux-3-image-klein-base \
  --dataset_metadata_path data/diffsynth_example_dataset/flux3/flux-3-image-klein-base/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/flux-3-image-klein-base:model.safetensors,black-forest-labs/flux-3-image-klein-base:text_encoder/*.safetensors,black-forest-labs/flux-3-image-klein-base:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "black-forest-labs/flux-3-image-klein-base:text_encoder/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/flux-3-image-klein-base_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters

# Edit

# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset

# accelerate launch examples/flux3/model_training/train.py \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "black-forest-labs/flux-3-image-klein-base:model.safetensors,black-forest-labs/flux-3-image-klein-base:text_encoder/*.safetensors,black-forest-labs/flux-3-image-klein-base:vae/diffusion_pytorch_model.safetensors" \
#   --processor_path "black-forest-labs/flux-3-image-klein-base:text_encoder/" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/flux-3-image-klein-base-Multi-Reference_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing \
#   --find_unused_parameters
