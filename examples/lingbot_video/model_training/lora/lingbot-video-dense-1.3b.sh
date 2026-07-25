# Download the LingBot-Video Dense-1.3B weights (DiT + text encoder + VAE + processor).
# This fetches the whole repo once into ./models so the paths below resolve locally.
modelscope download --model Robbyant/lingbot-video-dense-1.3b --local_dir ./models/Robbyant/lingbot-video-dense-1.3b

# Attention-only LoRA SFT.
# `--lora_target_modules "to_q,to_k,to_v,to_out"` patches LoRA on the joint
# text+video self-attention only, leaving the MoE / FFN experts and the router frozen.
accelerate launch examples/lingbot_video/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_paths '[
    [
      "./models/Robbyant/lingbot-video-dense-1.3b/text_encoder/model-00001-of-00002.safetensors",
      "./models/Robbyant/lingbot-video-dense-1.3b/text_encoder/model-00002-of-00002.safetensors"
    ],
    "./models/Robbyant/lingbot-video-dense-1.3b/transformer/diffusion_pytorch_model.safetensors",
    "./models/Robbyant/lingbot-video-dense-1.3b/vae/diffusion_pytorch_model.safetensors"
  ]' \
  --processor_path "./models/Robbyant/lingbot-video-dense-1.3b/processor" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/lingbot-video-dense-1.3b_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing
