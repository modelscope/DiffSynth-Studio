# Example Dataset: https://modelscope.cn/datasets/DiffSynth-Studio/example_image_dataset/tree/master/layer

accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset/layer \
  --dataset_metadata_path data/example_image_dataset/layer/metadata_layered.json \
  --data_file_keys "image,layer_input_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Layered:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image-Layered:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Layered_full" \
  --trainable_models "dit" \
  --extra_inputs "layer_num,layer_input_image" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
