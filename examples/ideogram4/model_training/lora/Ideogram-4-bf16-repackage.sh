# Dataset: data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage/
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ideogram4/Ideogram-4-bf16-repackage/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/ideogram4/model_training/train.py \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage/metadata.json" \
    --model_id_with_origin_paths "DiffSynth-Studio/ideogram-4-bf16-repackage:transformer/diffusion_pytorch_model.safetensors,DiffSynth-Studio/ideogram-4-bf16-repackage:text_encoder/model.safetensors,DiffSynth-Studio/ideogram-4-bf16-repackage:vae/diffusion_pytorch_model.safetensors" \
    --lora_base_model "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 1048576 \
    --dataset_repeat 100 \
    --output_path "./models/train/Ideogram-4-bf16-repackage_lora" \
    --lora_target_modules "attention.qkv,attention.o,feed_forward.w1,feed_forward.w2,feed_forward.w3,adaln_modulation" \
    --data_file_keys "image"
