modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Distill-Full/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Distill-Full \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Distill-Full/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "DiffSynth-Studio/Qwen-Image-Distill-Full:diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Distill-Full_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
