# 8*H200 required
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/WanToDance-14B-global/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/WanToDance-14B-global \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/WanToDance-14B-global/metadata.json \
  --data_file_keys "video,wantodance_reference_image,wantodance_keyframes,wantodance_music_path" \
  --height 1280 \
  --width 720 \
  --num_frames 149 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/WanToDance-14B:global_model.safetensors,Wan-AI/WanToDance-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/WanToDance-14B:Wan2.1_VAE.pth,Wan-AI/WanToDance-14B:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/WanToDance-14B-global_full" \
  --trainable_models "dit" \
  --extra_inputs "wantodance_music_path,wantodance_reference_image,wantodance_fps,wantodance_keyframes,wantodance_keyframes_mask,framewise_decoding" \
  --use_gradient_checkpointing_offload \
  --framewise_decoding
