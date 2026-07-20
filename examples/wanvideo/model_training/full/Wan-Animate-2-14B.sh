modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan-Animate-2-14B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan-Animate-2-14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan-Animate-2-14B/metadata.json \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 41 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan-Animate-2-14B:videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan-Animate-2-14B:videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan-Animate-2-14B_full_splited_cache" \
  --trainable_models "dit" \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path models/train/Wan-Animate-2-14B_full_splited_cache \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 41 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan-Animate-2-14B:wan_animate_2/wan_animate_2_bf16.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan-Animate-2-14B_full" \
  --trainable_models "dit" \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:train"
