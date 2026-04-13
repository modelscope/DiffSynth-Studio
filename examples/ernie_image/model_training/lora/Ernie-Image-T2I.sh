# Dataset: data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ernie_image/Ernie-Image-T2I/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/ernie_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I \
  --dataset_metadata_path data/diffsynth_example_dataset/ernie_image/Ernie-Image-T2I/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "PaddlePaddle/ERNIE-Image:transformer/diffusion_pytorch_model*.safetensors,PaddlePaddle/ERNIE-Image:text_encoder/model.safetensors,PaddlePaddle/ERNIE-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Ernie-Image-T2I_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
