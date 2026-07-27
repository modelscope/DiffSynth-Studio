# Download the example video-SFT dataset (a small text-to-video set with a `video` +
# `prompt` metadata.csv), the same DiffSynth-Studio example dataset the other training
# scripts use. NOTE: its prompts are plain prose; for best quality rewrite them into
# structured captions first with model_training/rewrite_captions.py (see the README).
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset

# Attention-only LoRA SFT.
# `--lora_target_modules "to_q,to_k,to_v,to_out"` patches LoRA on the joint
# text+video self-attention only, leaving the MoE / FFN experts and the router frozen.
# `--num_frames 169` = 7 s at the model's native 24 fps (169 = 4k+1, required by the
# VAE's 4x temporal compression). The loader samples the first 169 frames of each clip.
accelerate launch examples/lingbot_video/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 169 \
  --dataset_repeat 200 \
  --model_id_with_origin_paths "Robbyant/lingbot-video-dense-1.3b:transformer/diffusion_pytorch_model.safetensors,Robbyant/lingbot-video-dense-1.3b:text_encoder/model*.safetensors,Robbyant/lingbot-video-dense-1.3b:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "Robbyant/lingbot-video-dense-1.3b:processor/" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/lingbot-video-dense-1.3b_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing
