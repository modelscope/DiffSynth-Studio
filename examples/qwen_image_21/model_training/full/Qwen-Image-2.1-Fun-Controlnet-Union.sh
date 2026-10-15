modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image_21/Qwen-Image-2.1-Fun-Controlnet-Union/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/qwen_image_21/model_training/full/accelerate_config_zero3.yaml \
  examples/qwen_image_21/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-Fun-Controlnet-Union \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1-Fun-Controlnet-Union/metadata.csv \
  --data_file_keys "image,controlnet_image" \
  --max_pixels 1048576 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors,PAI/Qwen-Image-2.1-Fun-Controlnet-Union:Qwen-Image-2.1-Fun-Controlnet-Union.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.controlnet." \
  --output_path "./models/train/Qwen-Image-2.1-Fun-Controlnet-Union_full" \
  --trainable_models "controlnet" \
  --extra_inputs "controlnet_image" \
  --use_gradient_checkpointing \
  --find_unused_parameters
