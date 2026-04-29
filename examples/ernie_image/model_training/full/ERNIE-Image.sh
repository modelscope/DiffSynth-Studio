# Dataset: data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I/

accelerate launch --config_file examples/ernie_image/model_training/full/accelerate_config_zero3.yaml \
  examples/ernie_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I \
  --dataset_metadata_path data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "PaddlePaddle/ERNIE-Image:transformer/diffusion_pytorch_model*.safetensors,PaddlePaddle/ERNIE-Image:text_encoder/model.safetensors,PaddlePaddle/ERNIE-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Ernie-Image-T2I_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
