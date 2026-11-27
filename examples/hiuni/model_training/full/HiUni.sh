modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hiuni/HiUni/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/hiuni/model_training/full/accelerate_config_zero2offload.yaml examples/hiuni/model_training/train.py \
  --dataset_base_path "data/diffsynth_example_dataset/hiuni/HiUni" \
  --dataset_metadata_path "data/diffsynth_example_dataset/hiuni/HiUni/metadata.csv" \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "ernie-research/HiUni:model-*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/HiUni_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
