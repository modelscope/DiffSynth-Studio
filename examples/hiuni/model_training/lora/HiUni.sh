modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hiuni/HiUni/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/hiuni/model_training/train.py \
  --dataset_base_path "data/diffsynth_example_dataset/hiuni/HiUni" \
  --dataset_metadata_path "data/diffsynth_example_dataset/hiuni/HiUni/metadata.csv" \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "ernie-research/HiUni:model-*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/HiUni_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
