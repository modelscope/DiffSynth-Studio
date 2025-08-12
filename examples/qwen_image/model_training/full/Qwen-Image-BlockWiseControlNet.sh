accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path data/t2i_dataset_annotations/blip3o/blip3o_control_images_train_for_diffsynth.jsonl \
  --data_file_keys "image,controlnet_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_paths '[
    [
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
    ],
    [
        "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
    ],
    "models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors",
    "models/DiffSynth-Studio/BlockWiseControlnet/model_init.safetensors"
]' \
  --learning_rate 1e-3 \
  --num_epochs 1000000 \
  --remove_prefix_in_ckpt "pipe.blockwise_controlnet." \
  --output_path "./models/train/Qwen-Image-BlockWiseControlNet_full_lr1e-3_wd1e-6" \
  --trainable_models "blockwise_controlnet" \
  --extra_inputs "controlnet_image" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --save_steps 2000
