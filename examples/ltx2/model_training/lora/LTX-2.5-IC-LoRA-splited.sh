# LTX-2.5 IC-LoRA split training (DiffSynth idiom, donor: ltx-trainer v2v_ic_lora).
#
# Metadata keys per sample: video, reference_video, caption (prompt also accepted).
#   video:          target video the model learns to generate (full resolution).
#   reference_video: conditioning video at native resolution; the downscale ratio
#                   against the target is inferred from data (exact uniform integer)
#                   and exported to the adapter as reference_downscale_factor.
#   caption:        text prompt.
# Example metadata.json entry:
#   {"video": "target.mp4", "reference_video": "ref_half.mp4",
#    "caption": "[VISUAL]:a cat boxing. [SOUNDS]:boxing sounds", "frame_rate": 24}
#
# Notes:
# - reference_video keeps native resolution through the dataset; the pipeline
#   resizes it to height/factor x width/factor (check_in_context_video).
# - first_frame_conditioning_p 0.2 composes I2V samples into IC-LoRA training
#   (donor v2v_ic_lora.yaml); the draw freezes per sample in the split cache.
# - LoRA targets are video self-attention plus feed-forward per card recipes
#   (peft matches ".attn1.to_q" suffixes, so audio_* modules are excluded).
# - To train with audio as well, add input_audio to data_file_keys/extra_inputs
#   and ship audio files in metadata (donor V2V strategy is video-only).
# - On a GPU shared with a desktop, --abort_vram_gb caps this process below the
#   card-wide sysfs line and kills it if the line is crossed. Model CPU offload
#   keeps the frozen bulk in host RAM and streams one layer at a time.

# Splited Training
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/ic_lora_2_5 \
  --dataset_metadata_path data/ic_lora_2_5/metadata.json \
  --data_file_keys "video,reference_video" \
  --extra_inputs "reference_video,input_image,frame_rate" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --tokenizer_path "./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer" \
  --model_id_with_origin_paths "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors,DiffSynth-Studio/LTX-2.5-Repackage:text_encoder_post_modules.safetensors" \
  --first_frame_conditioning_p 0.2 \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-IC-LoRA-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0,ff.net.0.proj,ff.net.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --abort_vram_gb 29.5 \
  --task "sft:data_process"

accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path ./models/train/LTX2.5-IC-LoRA-splited-cache \
  --data_file_keys "video,reference_video" \
  --extra_inputs "reference_video,input_image,frame_rate" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 100 \
  --tokenizer_path "./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer" \
  --model_id_with_origin_paths "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors,DiffSynth-Studio/LTX-2.5-Repackage:text_encoder_post_modules.safetensors" \
  --fp8_models "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors,DiffSynth-Studio/LTX-2.5-Repackage:text_encoder_post_modules.safetensors" \
  --first_frame_conditioning_p 0.2 \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-IC-LoRA" \
  --lora_base_model "dit" \
  --lora_target_modules "attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0,ff.net.0.proj,ff.net.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --abort_vram_gb 29.5 \
  --task "sft:train"
