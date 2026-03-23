export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "z_image/Z-Image-Turbo/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/z_image/model_training/full/accelerate_config.yaml examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 400 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Turbo_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --enable_npu_patch
