modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_video_edit/Qwen-Video-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/qwen_video_edit/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit/metadata.json \
  --data_file_keys "video,input_video" \
  --height 640 \
  --width 384 \
  --num_frames 45 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "yunpeng1998/Qwen-Video-Edit:360P/step-30000.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Video-Edit_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --zero_cond_t \
  --dataset_num_workers 8 \
  --find_unused_parameters
