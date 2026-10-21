# In this training script, masking is enabled. 
# To disable it, update the arguments to `--data_file_keys "image,edit_image" --extra_inputs "edit_image"`
# and remove the `mask` field from the dataset metadata.

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "fibo/Fibo-2.0-dev-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/fibo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/metadata.json \
  --data_file_keys "image,edit_image,mask" \
  --extra_inputs "edit_image,mask" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "briaai/Fibo-2.0-dev:transformer/*.safetensors,briaai/Fibo-2.0-dev:text_encoder/model.safetensors,briaai/Fibo-2.0-dev:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "briaai/Fibo-2.0-dev:processor/" \
  --font_path "briaai/Fibo-2.0-dev:fonts/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Fibo-2.0-dev-Edit_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing
