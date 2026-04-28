# This script is tested on 8*910B(NPU)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux2/FLUX.2-klein-9B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/flux2/model_training/full/accelerate_config.yaml examples/flux2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-9B \
  --dataset_metadata_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-9B/metadata.csv \
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

# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset

# accelerate launch --config_file examples/flux2/model_training/full/accelerate_config.yaml examples/flux2/model_training/train.py \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
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
