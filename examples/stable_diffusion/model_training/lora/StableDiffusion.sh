# Dataset: data/diffsynth_example_dataset/stable_diffusion/StableDiffusion/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "stable_diffusion/StableDiffusion/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
accelerate launch examples/stable_diffusion/model_training/train.py \
    --learning_rate 1e-4 --num_epochs 5 \
    --lora_rank 32 \
    --use_gradient_checkpointing --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/stable_diffusion/StableDiffusion" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/stable_diffusion/StableDiffusion/metadata.csv" \
    --model_id_with_origin_paths "AI-ModelScope/stable-diffusion-v1-5:text_encoder/model.safetensors,AI-ModelScope/stable-diffusion-v1-5:unet/diffusion_pytorch_model.safetensors,AI-ModelScope/stable-diffusion-v1-5:vae/diffusion_pytorch_model.safetensors" \
    --lora_base_model "unet" \
    --remove_prefix_in_ckpt "pipe.unet." \
    --max_pixels 262144 \
    --height 512 --width 512 \
    --dataset_repeat 50 \
    --output_path "./models/train/StableDiffusion_lora" \
    --lora_target_modules "to_q,to_k,to_v,to_out.0" \
    --data_file_keys "image"
