# This script was tested using zero3 and on 8*910B(NPU)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image-Edit-2509/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero3.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2509 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit-2509/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2509_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --initialize_model_on_cpu
