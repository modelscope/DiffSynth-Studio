# Dataset: data/diffsynth_example_dataset/ace_step/Ace-Step1.5/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ace_step/Ace-Step1.5/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/ace_step/model_training/train.py \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --trainable_models "dit" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/ace_step/Ace-Step1.5" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/ace_step/Ace-Step1.5/metadata.json" \
    --model_id_with_origin_paths "ACE-Step/Ace-Step1.5:acestep-v15-turbo/model.safetensors,ACE-Step/Ace-Step1.5:Qwen3-Embedding-0.6B/model.safetensors,ACE-Step/Ace-Step1.5:vae/diffusion_pytorch_model.safetensors" \
    --tokenizer_path "ACE-Step/Ace-Step1.5:Qwen3-Embedding-0.6B/" \
    --silence_latent_path "ACE-Step/Ace-Step1.5:acestep-v15-turbo/silence_latent.pt" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --dataset_repeat 50 \
    --output_path "./models/train/Ace-Step1.5_full" \
    --data_file_keys "audio"
