# Dataset: data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "joyai_image/JoyAI-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFFSYNTH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$DIFFSYNTH_ROOT"

export CONDA_DEFAULT_ENV=joyai-image-diffsynth
export PATH="/root/miniconda3/envs/joyai-image-diffsynth/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$DIFFSYNTH_ROOT:$PYTHONPATH"

# ==================== 第一阶段：前处理计算 ====================
# 加载 VAE + TextEncoder，缓存编码结果到硬盘
accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
    --learning_rate 1e-5 --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/JoyAI-Image-Edit-full-cache" \
    --use_gradient_checkpointing --find_unused_parameters \
    --data_file_keys "image,edit_images" \
    --extra_inputs "edit_images" \
    --task "sft:data_process"

# ==================== 第二阶段：DiT 训练 ====================
# 从缓存读取，仅训练 DiT
# NOTE: Full training of 16B DiT requires DeepSpeed ZeRO-3 with multiple GPUs.
# This script uses single GPU config. If OOM occurs, install DeepSpeed and use
# accelerate_config_zero3.yaml instead.
accelerate launch --config_file examples/joyai_image/model_training/full/accelerate_config_single_gpu.yaml \
    examples/joyai_image/model_training/train.py \
    --dataset_base_path "./models/train/JoyAI-Image-Edit-full-cache" \
    --max_pixels 1048576 --dataset_repeat 50 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth" \
    --learning_rate 1e-5 --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/JoyAI-Image-Edit-full" \
    --trainable_models "dit" \
    --use_gradient_checkpointing --find_unused_parameters \
    --data_file_keys "image,edit_images" \
    --extra_inputs "edit_images" \
    --task "sft:train"
