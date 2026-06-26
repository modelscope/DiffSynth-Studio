# Please run `accelerate config` to configure GPU, DeepSpeed, etc.
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "boogu_image/Boogu-Image-0.1-Base/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/boogu_image/model_training/train.py \
  --dataset_base_path "./data/diffsynth_example_dataset/boogu_image/Boogu-Image-0.1-Base" \
  --dataset_metadata_path "./data/diffsynth_example_dataset/boogu_image/Boogu-Image-0.1-Base/metadata.csv" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Boogu/Boogu-Image-0.1-Base:transformer/*.safetensors,Boogu/Boogu-Image-0.1-Base:mllm/*.safetensors,Boogu/Boogu-Image-0.1-Base:vae/*.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Boogu-Image-0.1-Base_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --data_file_keys "image"
