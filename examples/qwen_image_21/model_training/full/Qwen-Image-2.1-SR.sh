# Qwen-Image-2.1-SR is trained in two parts. Part 1 trains the Template model, the
# continuous detail control module. Part 2 trains the SR LoRA on the DiT.

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image_21/Qwen-Image-2.1-SR/*" --local_dir ./data/diffsynth_example_dataset

# Part 1: train the Template model
accelerate launch --config_file examples/qwen_image_21/model_training/full/accelerate_config.yaml \
  examples/qwen_image_21/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-SR \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-SR/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image,template_inputs" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/Qwen-Image-2.1-SR:Template-DetailScale/" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --trainable_models "template_model" \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "./models/train/Qwen-Image-2.1-SR_template" \
  --use_gradient_checkpointing \
  --find_unused_parameters

# Part 2: train the SR LoRA
accelerate launch --config_file examples/qwen_image_21/model_training/full/accelerate_config.yaml \
  examples/qwen_image_21/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-SR \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-SR/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image,template_inputs" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/Qwen-Image-2.1-SR:Template-DetailScale/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-2.1-SR_lora" \
  --use_gradient_checkpointing \
  --find_unused_parameters
