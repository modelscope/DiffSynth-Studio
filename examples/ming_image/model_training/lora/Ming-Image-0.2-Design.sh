modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ming_image/Ming-Image-0.2-Design/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/ming_image/model_training/train.py \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --output_path "./models/train/Ming-Image-0.2-Design_lora" \
  --dataset_base_path data/diffsynth_example_dataset/ming_image/Ming-Image-0.2-Design \
  --dataset_metadata_path data/diffsynth_example_dataset/ming_image/Ming-Image-0.2-Design/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "inclusionAI/Ming-Image-0.2-Design:transformer/diffusion_pytorch_model*.safetensors,inclusionAI/Ming-Image-0.2-Design:vae/diffusion_pytorch_model.safetensors,inclusionAI/Ming-Image-0.2-Design:mllm/model*.safetensors,inclusionAI/Ming-Image-0.2-Design:connector/model*.safetensors,inclusionAI/Ming-Image-0.2-Design:mlp/model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --data_file_keys "image"

# Edit

# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset

# accelerate launch examples/ming_image/model_training/train.py \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --lora_base_model "dit" \
#   --lora_target_modules "" \
#   --lora_rank 32 \
#   --output_path "./models/train/Ming-Image-0.2-Design_lora" \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.csv \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "inclusionAI/Ming-Image-0.2-Design:transformer/diffusion_pytorch_model*.safetensors,inclusionAI/Ming-Image-0.2-Design:vae/diffusion_pytorch_model.safetensors,inclusionAI/Ming-Image-0.2-Design:mllm/model*.safetensors,inclusionAI/Ming-Image-0.2-Design:connector/model*.safetensors,inclusionAI/Ming-Image-0.2-Design:mlp/model.safetensors" \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --use_gradient_checkpointing \
#   --find_unused_parameters
