modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image21/Qwen-Image2.1/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/qwen_image21/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image21/Qwen-Image2.1 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image21/Qwen-Image2.1/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image2.1:vae/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image2.1_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters

# Edit

# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2511/*" --local_dir ./data/diffsynth_example_dataset

# accelerate launch examples/qwen_image21/model_training/train.py \
#   --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511 \
#   --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2511/metadata.json \
#   --data_file_keys "image,edit_image" \
#   --extra_inputs "edit_image" \
#   --max_pixels 1048576 \
#   --dataset_repeat 100 \
#   --model_id_with_origin_paths "Qwen/Qwen-Image2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image2.1:vae/diffusion_pytorch_model.safetensors" \
#   --learning_rate 1e-4 \
#   --num_epochs 5 \
#   --remove_prefix_in_ckpt "pipe.dit." \
#   --output_path "./models/train/Qwen-Image2.1_lora" \
#   --lora_base_model "dit" \
#   --lora_target_modules "to_q,to_k,to_v,to_out.0" \
#   --lora_rank 32 \
#   --use_gradient_checkpointing \
#   --find_unused_parameters
