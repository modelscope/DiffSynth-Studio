# Image-to-video (TI2V) LoRA SFT.
#
# Same DiT / dataset / attention-only LoRA scope as the t2v script -- the only delta is
# `--first_frame_as_condition`, which conditions each clip on its OWN first frame: the
# frame is VAE-encoded to a clean latent pinned into the first temporal slot (and fed to
# the Qwen3-VL text encoder), and excluded from the flow-matching loss, so the LoRA learns
# to animate frames 2..N from frame 1. Dense-1.3B reuses the same T2V weights -- no
# separate i2v checkpoint. Reuses the shared example dataset (plain `video` + `prompt`);
# no condition column is required. If your dataset instead ships a distinct condition
# frame, drop this flag and pass it via `--extra_inputs input_image` (adding that column
# to --data_file_keys).
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "lingbot_video/lingbot-video-dense-1.3b/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/lingbot_video/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b \
  --dataset_metadata_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b/metadata.json \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --first_frame_as_condition \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Robbyant/lingbot-video-dense-1.3b:transformer/diffusion_pytorch_model.safetensors,Robbyant/lingbot-video-dense-1.3b:text_encoder/model*.safetensors,Robbyant/lingbot-video-dense-1.3b:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "Robbyant/lingbot-video-dense-1.3b:processor/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/lingbot-video-dense-1.3b_ti2v_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing
