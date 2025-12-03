# Z-Image-Turbo is a distilled model.
# After training, it loses its distillation-based acceleration capability,
# leading to degraded generation quality at fewer inference steps.
# This issue can be mitigated by using a pre-trained LoRA model to assist the training process.
# https://www.modelscope.cn/models/ostris/zimage_turbo_training_adapter

accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Turbo_lora_differential" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --preset_lora_path "models/ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors" \
  --preset_lora_model "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8
