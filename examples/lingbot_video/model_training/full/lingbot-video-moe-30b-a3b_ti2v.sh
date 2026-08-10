modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "lingbot_video/lingbot-video-moe-30b-a3b_ti2v/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/lingbot_video/model_training/full/accelerate_config_moe.yaml examples/lingbot_video/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-moe-30b-a3b_ti2v \
  --dataset_metadata_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-moe-30b-a3b_ti2v/metadata.json \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --first_frame_as_condition \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Robbyant/lingbot-video-moe-30b-a3b:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen3-VL-4B-Instruct:*.safetensors,Robbyant/lingbot-video-moe-30b-a3b:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "Qwen/Qwen3-VL-4B-Instruct:" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/lingbot-video-moe-30b-a3b_ti2v_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload
