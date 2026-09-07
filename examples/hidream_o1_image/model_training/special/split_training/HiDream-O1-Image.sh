modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hidream_o1_image/HiDream-O1-Image/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1: cache deterministic preprocessing outputs.
accelerate launch examples/hidream_o1_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image/metadata.csv \
  --max_pixels 4194304 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths 'HiDream-ai/HiDream-O1-Image:model-*.safetensors' \
  --processor_config HiDream-ai/HiDream-O1-Image:./ \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_rank 32 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/HiDream-O1-Image_split_cache \
  --lora_base_model dit \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2 \
  --use_gradient_checkpointing \
  --noise_scale 8.0 \
  --offload_models 'HiDream-ai/HiDream-O1-Image:model-*.safetensors' \
  --task sft:data_process

# Stage 2: train LoRA from the cached dataset.
accelerate launch examples/hidream_o1_image/model_training/train.py \
  --dataset_base_path ./models/train/HiDream-O1-Image_split_cache \
  --max_pixels 4194304 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths 'HiDream-ai/HiDream-O1-Image:model-*.safetensors' \
  --processor_config HiDream-ai/HiDream-O1-Image:./ \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_rank 32 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path ./models/train/HiDream-O1-Image_split \
  --lora_base_model dit \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2 \
  --use_gradient_checkpointing \
  --noise_scale 8.0 \
  --task sft:train
