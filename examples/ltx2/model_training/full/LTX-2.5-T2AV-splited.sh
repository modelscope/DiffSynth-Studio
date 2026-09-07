modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ltx2/LTX-2.3-T2AV-splited/*" --local_dir ./data/diffsynth_example_dataset

# Splited Training
accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited \
  --dataset_metadata_path data/diffsynth_example_dataset/ltx2/LTX-2.3-T2AV-splited/metadata.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-T2AV-full-splited-cache" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch --config_file examples/ltx2/model_training/full/accelerate_config_zero2offload.yaml examples/ltx2/model_training/train.py \
  --dataset_base_path ./models/train/LTX2.5-T2AV-full-splited-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio" \
  --height 512 \
  --width 768 \
  --num_frames 121 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --fp8_models "Lightricks/LTX-2.5:text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-video-vae-bf16.safetensors,Lightricks/LTX-2.5:vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/LTX2.5-T2AV-full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --task "sft:train"
