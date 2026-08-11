modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-Animate-2-14B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B/metadata.json \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-2-14B:videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-2-14B:videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-2-14B_full_splited_cache" \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path models/train/Wan2.2-Animate-2-14B_full_splited_cache \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-2-14B:wan_animate_2/wan_animate_2_bf16.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-2-14B_full" \
  --trainable_models "dit.blocks.0.block.self_attn.q,dit.blocks.0.block.self_attn.k,dit.blocks.0.block.self_attn.v,dit.blocks.1.block.self_attn.q,dit.blocks.1.block.self_attn.k,dit.blocks.1.block.self_attn.v,dit.blocks.2.block.self_attn.q,dit.blocks.2.block.self_attn.k,dit.blocks.2.block.self_attn.v,dit.blocks.3.block.self_attn.q,dit.blocks.3.block.self_attn.k,dit.blocks.3.block.self_attn.v,dit.blocks.4.block.self_attn.q,dit.blocks.4.block.self_attn.k,dit.blocks.4.block.self_attn.v,dit.blocks.5.block.self_attn.q,dit.blocks.5.block.self_attn.k,dit.blocks.5.block.self_attn.v,dit.blocks.6.block.self_attn.q,dit.blocks.6.block.self_attn.k,dit.blocks.6.block.self_attn.v,dit.blocks.7.block.self_attn.q,dit.blocks.7.block.self_attn.k,dit.blocks.7.block.self_attn.v,dit.blocks.8.block.self_attn.q,dit.blocks.8.block.self_attn.k,dit.blocks.8.block.self_attn.v,dit.blocks.9.block.self_attn.q,dit.blocks.9.block.self_attn.k,dit.blocks.9.block.self_attn.v,dit.blocks.10.block.self_attn.q,dit.blocks.10.block.self_attn.k,dit.blocks.10.block.self_attn.v,dit.blocks.11.block.self_attn.q,dit.blocks.11.block.self_attn.k,dit.blocks.11.block.self_attn.v,dit.blocks.12.block.self_attn.q,dit.blocks.12.block.self_attn.k,dit.blocks.12.block.self_attn.v,dit.blocks.13.block.self_attn.q,dit.blocks.13.block.self_attn.k,dit.blocks.13.block.self_attn.v,dit.blocks.14.block.self_attn.q,dit.blocks.14.block.self_attn.k,dit.blocks.14.block.self_attn.v,dit.blocks.15.block.self_attn.q,dit.blocks.15.block.self_attn.k,dit.blocks.15.block.self_attn.v,dit.blocks.16.block.self_attn.q,dit.blocks.16.block.self_attn.k,dit.blocks.16.block.self_attn.v,dit.blocks.17.block.self_attn.q,dit.blocks.17.block.self_attn.k,dit.blocks.17.block.self_attn.v,dit.blocks.18.block.self_attn.q,dit.blocks.18.block.self_attn.k,dit.blocks.18.block.self_attn.v,dit.blocks.19.block.self_attn.q,dit.blocks.19.block.self_attn.k,dit.blocks.19.block.self_attn.v,dit.blocks.20.block.self_attn.q,dit.blocks.20.block.self_attn.k,dit.blocks.20.block.self_attn.v,dit.blocks.21.block.self_attn.q,dit.blocks.21.block.self_attn.k,dit.blocks.21.block.self_attn.v,dit.blocks.22.block.self_attn.q,dit.blocks.22.block.self_attn.k,dit.blocks.22.block.self_attn.v,dit.blocks.23.block.self_attn.q,dit.blocks.23.block.self_attn.k,dit.blocks.23.block.self_attn.v,dit.blocks.24.block.self_attn.q,dit.blocks.24.block.self_attn.k,dit.blocks.24.block.self_attn.v,dit.blocks.25.block.self_attn.q,dit.blocks.25.block.self_attn.k,dit.blocks.25.block.self_attn.v,dit.blocks.26.block.self_attn.q,dit.blocks.26.block.self_attn.k,dit.blocks.26.block.self_attn.v,dit.blocks.27.block.self_attn.q,dit.blocks.27.block.self_attn.k,dit.blocks.27.block.self_attn.v,dit.blocks.28.block.self_attn.q,dit.blocks.28.block.self_attn.k,dit.blocks.28.block.self_attn.v,dit.blocks.29.block.self_attn.q,dit.blocks.29.block.self_attn.k,dit.blocks.29.block.self_attn.v,dit.blocks.30.block.self_attn.q,dit.blocks.30.block.self_attn.k,dit.blocks.30.block.self_attn.v,dit.blocks.31.block.self_attn.q,dit.blocks.31.block.self_attn.k,dit.blocks.31.block.self_attn.v,dit.blocks.32.block.self_attn.q,dit.blocks.32.block.self_attn.k,dit.blocks.32.block.self_attn.v,dit.blocks.33.block.self_attn.q,dit.blocks.33.block.self_attn.k,dit.blocks.33.block.self_attn.v,dit.blocks.34.block.self_attn.q,dit.blocks.34.block.self_attn.k,dit.blocks.34.block.self_attn.v,dit.blocks.35.block.self_attn.q,dit.blocks.35.block.self_attn.k,dit.blocks.35.block.self_attn.v,dit.blocks.36.block.self_attn.q,dit.blocks.36.block.self_attn.k,dit.blocks.36.block.self_attn.v,dit.blocks.37.block.self_attn.q,dit.blocks.37.block.self_attn.k,dit.blocks.37.block.self_attn.v,dit.blocks.38.block.self_attn.q,dit.blocks.38.block.self_attn.k,dit.blocks.38.block.self_attn.v,dit.blocks.39.block.self_attn.q,dit.blocks.39.block.self_attn.k,dit.blocks.39.block.self_attn.v" \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:train"
