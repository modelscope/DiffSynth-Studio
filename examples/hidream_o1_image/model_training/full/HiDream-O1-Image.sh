# Dataset: data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hidream_o1_image/HiDream-O1-Image/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file examples/hidream_o1_image/model_training/full/accelerate_config_zero3.yaml \
    examples/hidream_o1_image/model_training/train.py \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --trainable_models "dit" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image/metadata.csv" \
    --model_id_with_origin_paths "HiDream-ai/HiDream-O1-Image:model-*.safetensors" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 4194304 \
    --dataset_repeat 50 \
    --output_path "./models/train/HiDream-O1-Image_full" \
    --data_file_keys "image" \
    --noise_scale 8.0
