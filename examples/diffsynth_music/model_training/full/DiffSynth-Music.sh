modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "diffsynth_music/DiffSynth-Music/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/diffsynth_music/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/diffsynth_music/DiffSynth-Music" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/diffsynth_music/DiffSynth-Music/metadata.json" \
    --data_file_keys "audio" \
    --extra_inputs "template_inputs" \
    --dataset_repeat 200 \
    --model_id_with_origin_paths "DiffSynth-Studio/DiffSynth-Music:transformer/model.safetensors,DiffSynth-Studio/DiffSynth-Music:conditioner/model.safetensors,DiffSynth-Studio/DiffSynth-Music:text_encoder/model.safetensors,DiffSynth-Studio/DiffSynth-Music:vae/model.safetensors" \
    --template_model_id_or_path "DiffSynth-Studio/DiffSynth-Music:template_control/" \
    --tokenizer_path "DiffSynth-Studio/DiffSynth-Music:text_encoder/" \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.template_model." \
    --output_path "./models/train/DiffSynth-Music_full" \
    --trainable_models "template_model" \
    --use_gradient_checkpointing \
    --find_unused_parameters
