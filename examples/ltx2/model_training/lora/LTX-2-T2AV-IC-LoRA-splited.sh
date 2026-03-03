# Splited Training
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/example_video_dataset/ltx2 \
  --dataset_metadata_path data/example_video_dataset/ltx2_t2av_iclora.json \
  --data_file_keys "video,input_audio,in_context_videos" \
  --extra_inputs "input_audio,in_context_videos,in_context_downsample_factor,frame_rate" \
  --height 512 \
  --width 768 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2-T2AV-IC-LoRA-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path ./models/train/LTX2-T2AV-IC-LoRA-splited-cache \
  --data_file_keys "video,input_audio,in_context_videos" \
  --extra_inputs "input_audio,in_context_videos,in_context_downsample_factor,frame_rate" \
  --height 512 \
  --width 768 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2-Repackage:transformer.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2-T2AV-IC-LoRA" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train"
