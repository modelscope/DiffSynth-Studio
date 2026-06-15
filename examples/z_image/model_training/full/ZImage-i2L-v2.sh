modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "z_image/ZImage-i2L-v2/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/ZImage-i2L-v2 \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/ZImage-i2L-v2/metadata.jsonl \
  --max_pixels 1048576 \
  --dataset_repeat 400 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --output_path "./models/train/ZImage-i2L-v2_full" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --dataset_num_workers 8 \
  --enable_lora_hot_loading \
  --template_model_id_or_path "DiffSynth-Studio/ZImage-i2L-v2:" \
  --extra_inputs "template_inputs" \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --trainable_models "template_model" # Use `template_model.emb2lora` to freeze the image encoder.
