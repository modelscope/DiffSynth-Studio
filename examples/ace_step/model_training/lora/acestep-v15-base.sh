# Dataset: data/diffsynth_example_dataset/ace_step/acestep-v15-base/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ace_step/acestep-v15-base/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/ace_step/model_training/train.py \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --dataset_base_path "./data/diffsynth_example_dataset/ace_step/acestep-v15-base" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/ace_step/acestep-v15-base/metadata.json" \
    --model_id_with_origin_paths "ACE-Step/acestep-v15-base:model.safetensors,ACE-Step/Ace-Step1.5:Qwen3-Embedding-0.6B/model.safetensors,ACE-Step/Ace-Step1.5:vae/diffusion_pytorch_model.safetensors" \
    --tokenizer_path "ACE-Step/Ace-Step1.5:Qwen3-Embedding-0.6B/" \
    --silence_latent_path "ACE-Step/Ace-Step1.5:acestep-v15-turbo/silence_latent.pt" \
    --lora_base_model "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --dataset_repeat 50 \
    --output_path "./models/train/acestep-v15-base_lora" \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --data_file_keys "audio"
