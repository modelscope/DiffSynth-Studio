# In this training script, masking is enabled. 
# To disable it, update the arguments to `--data_file_keys "image,edit_image" --extra_inputs "edit_image"`
# and remove the `mask` field from the dataset metadata.

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "fibo/Fibo-2.0-dev-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/fibo/model_training/full/accelerate_config.yaml examples/fibo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/metadata.json \
  --data_file_keys "image,edit_image,mask" \
  --extra_inputs "edit_image,mask" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "briaai/Fibo-2.0-dev:transformer/*.safetensors,briaai/Fibo-2.0-dev:text_encoder/model.safetensors,briaai/Fibo-2.0-dev:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "briaai/Fibo-2.0-dev:processor/" \
  --font_path "briaai/Fibo-2.0-dev:fonts/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Fibo-2.0-dev-Edit_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing
