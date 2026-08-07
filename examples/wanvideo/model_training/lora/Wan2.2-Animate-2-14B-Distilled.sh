modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-Animate-2-14B-Distilled/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B-Distilled \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-Animate-2-14B-Distilled/metadata.json \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-2-14B:videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-Animate-2-14B:videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-2-14B-Distilled_lora_splited_cache" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v" \
  --lora_rank 32 \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path models/train/Wan2.2-Animate-2-14B-Distilled_lora_splited_cache \
  --data_file_keys "video,animate2_reference_image,animate2_reference_video" \
  --height 640 \
  --width 352 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-2-14B:wan_animate_2/wan_animate_2_bf16_distillation.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-2-14B-Distilled_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v" \
  --lora_rank 32 \
  --extra_inputs "animate2_prompt_ref,animate2_reference_image,animate2_reference_video" \
  --use_gradient_checkpointing \
  --task "sft:train"
