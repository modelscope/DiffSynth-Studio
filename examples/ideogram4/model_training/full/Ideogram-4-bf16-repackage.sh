# Dataset: data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage/
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ideogram4/Ideogram-4-bf16-repackage/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/ideogram4/model_training/full/accelerate_config.yaml examples/ideogram4/model_training/train.py \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/ideogram4/Ideogram-4-bf16-repackage/metadata.json" \
    --model_id_with_origin_paths "DiffSynth-Studio/ideogram-4-bf16-repackage:transformer/diffusion_pytorch_model.safetensors,DiffSynth-Studio/ideogram-4-bf16-repackage:text_encoder/model.safetensors,DiffSynth-Studio/ideogram-4-bf16-repackage:vae/diffusion_pytorch_model.safetensors" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --trainable_models "dit" \
    --max_pixels 1048576 \
    --dataset_repeat 100 \
    --output_path "./models/train/Ideogram-4-bf16-repackage_full" \
    --data_file_keys "image"
