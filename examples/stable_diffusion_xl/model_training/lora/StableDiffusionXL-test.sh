# Dataset: data/diffsynth_example_dataset/stable_diffusion_xl/StableDiffusionXL/
# Debug test: num_epochs=1, dataset_repeat=1 for quick validation

# ===== 固定参数（无需修改） =====
accelerate launch examples/stable_diffusion_xl/model_training/train.py \
    --learning_rate 1e-4 --num_epochs 1 \
    --lora_rank 32 \
    --use_gradient_checkpointing --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/stable_diffusion_xl/StableDiffusionXL" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/stable_diffusion_xl/StableDiffusionXL/metadata.csv" \
    --model_id_with_origin_paths "AI-ModelScope/stable-diffusion-xl-base-1.0:text_encoder/model.safetensors,AI-ModelScope/stable-diffusion-xl-base-1.0:text_encoder_2/model.safetensors,AI-ModelScope/stable-diffusion-xl-base-1.0:unet/diffusion_pytorch_model.safetensors,AI-ModelScope/stable-diffusion-xl-base-1.0:vae/diffusion_pytorch_model.safetensors" \
    --tokenizer_path "AI-ModelScope/stable-diffusion-xl-base-1.0:tokenizer/" \
    --tokenizer_2_path "AI-ModelScope/stable-diffusion-xl-base-1.0:tokenizer_2/" \
    --lora_base_model "unet" \
    --remove_prefix_in_ckpt "pipe.unet." \
    --max_pixels 1048576 \
    --height 1024 --width 1024 \
    --dataset_repeat 1 \
    --output_path "./models/train/StableDiffusionXL_lora_debug" \
    --lora_target_modules "to_q,to_k,to_v,to_out.0" \
    --data_file_keys "image"
