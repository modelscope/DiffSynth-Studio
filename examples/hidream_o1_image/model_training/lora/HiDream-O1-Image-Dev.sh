modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hidream_o1_image/HiDream-O1-Image-Dev/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/hidream_o1_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image-Dev \
  --dataset_metadata_path data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image-Dev/metadata.csv \
  --max_pixels 4194304 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "HiDream-ai/HiDream-O1-Image-Dev:model-*.safetensors" \
  --processor_config "HiDream-ai/HiDream-O1-Image-Dev:./" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/HiDream-O1-Image-Dev_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --use_gradient_checkpointing \
  --noise_scale 7.5


# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset

# accelerate launch examples/hidream_o1_image/model_training/train.py \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 4194304 \
#   --dataset_repeat 50 \
#   --model_id_with_origin_paths "HiDream-ai/HiDream-O1-Image-Dev:model-*.safetensors" \
#   --processor_config "HiDream-ai/HiDream-O1-Image-Dev:./" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --lora_rank 32 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/HiDream-O1-Image-Dev_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#   --use_gradient_checkpointing \
#   --noise_scale 7.5
