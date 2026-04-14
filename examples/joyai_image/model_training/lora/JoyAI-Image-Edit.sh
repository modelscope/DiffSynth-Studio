# Dataset: data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "joyai_image/JoyAI-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/JoyAI-Image-Edit-split-cache" \
    --lora_base_model "dit" \
    --lora_target_modules "img_attn_qkv,txt_attn_qkv,img_attn_proj,txt_attn_proj" \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --data_file_keys "image,edit_images" \
    --extra_inputs "edit_images" \
    --task "sft:data_process"

accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./models/train/JoyAI-Image-Edit-split-cache" \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/JoyAI-Image-Edit-lora" \
    --lora_base_model "dit" \
    --lora_target_modules "img_attn_qkv,txt_attn_qkv,img_attn_proj,txt_attn_proj" \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --data_file_keys "image,edit_images" \
    --extra_inputs "edit_images" \
    --task "sft:train"
