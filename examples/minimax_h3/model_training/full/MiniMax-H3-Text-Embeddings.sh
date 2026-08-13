modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Text-Embeddings/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Text-Embeddings \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Text-Embeddings/metadata.json \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image,template_inputs" \
  --height 832 \
  --width 480 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:video_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:audio_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-pruned-nf4.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/MiniMax-H3-Text-Embeddings:models/art_is_explosion/" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "./models/train/MiniMax-H3-Text-Embeddings-full" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing
