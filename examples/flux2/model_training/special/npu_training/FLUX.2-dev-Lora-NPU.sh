export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-dev:text_encoder/*.safetensors,black-forest-labs/FLUX.2-dev:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-dev-LoRA-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_qkv_mlp_proj,to_out.0,to_add_out,linear_in,linear_out,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out,single_transformer_blocks.24.attn.to_out,single_transformer_blocks.25.attn.to_out,single_transformer_blocks.26.attn.to_out,single_transformer_blocks.27.attn.to_out,single_transformer_blocks.28.attn.to_out,single_transformer_blocks.29.attn.to_out,single_transformer_blocks.30.attn.to_out,single_transformer_blocks.31.attn.to_out,single_transformer_blocks.32.attn.to_out,single_transformer_blocks.33.attn.to_out,single_transformer_blocks.34.attn.to_out,single_transformer_blocks.35.attn.to_out,single_transformer_blocks.36.attn.to_out,single_transformer_blocks.37.attn.to_out,single_transformer_blocks.38.attn.to_out,single_transformer_blocks.39.attn.to_out,single_transformer_blocks.40.attn.to_out,single_transformer_blocks.41.attn.to_out,single_transformer_blocks.42.attn.to_out,single_transformer_blocks.43.attn.to_out,single_transformer_blocks.44.attn.to_out,single_transformer_blocks.45.attn.to_out,single_transformer_blocks.46.attn.to_out,single_transformer_blocks.47.attn.to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --task "sft:data_process"

accelerate launch --config_file examples/flux2/model_training/full/accelerate_config_zero3.yaml examples/flux2/model_training/train.py \
  --dataset_base_path "./models/train/FLUX.2-dev-LoRA-splited-cache" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-dev:transformer/*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-dev-LoRA-splited" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_qkv_mlp_proj,to_out.0,to_add_out,linear_in,linear_out,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out,single_transformer_blocks.24.attn.to_out,single_transformer_blocks.25.attn.to_out,single_transformer_blocks.26.attn.to_out,single_transformer_blocks.27.attn.to_out,single_transformer_blocks.28.attn.to_out,single_transformer_blocks.29.attn.to_out,single_transformer_blocks.30.attn.to_out,single_transformer_blocks.31.attn.to_out,single_transformer_blocks.32.attn.to_out,single_transformer_blocks.33.attn.to_out,single_transformer_blocks.34.attn.to_out,single_transformer_blocks.35.attn.to_out,single_transformer_blocks.36.attn.to_out,single_transformer_blocks.37.attn.to_out,single_transformer_blocks.38.attn.to_out,single_transformer_blocks.39.attn.to_out,single_transformer_blocks.40.attn.to_out,single_transformer_blocks.41.attn.to_out,single_transformer_blocks.42.attn.to_out,single_transformer_blocks.43.attn.to_out,single_transformer_blocks.44.attn.to_out,single_transformer_blocks.45.attn.to_out,single_transformer_blocks.46.attn.to_out,single_transformer_blocks.47.attn.to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --initialize_model_on_cpu \
  --task "sft:train"
