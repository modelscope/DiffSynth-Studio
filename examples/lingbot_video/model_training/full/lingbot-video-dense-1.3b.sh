modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "lingbot_video/lingbot-video-dense-1.3b/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/lingbot_video/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b \
  --dataset_metadata_path data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b/metadata.json \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Robbyant/lingbot-video-dense-1.3b:transformer/diffusion_pytorch_model.safetensors,Robbyant/lingbot-video-dense-1.3b:text_encoder/model*.safetensors,Robbyant/lingbot-video-dense-1.3b:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "Robbyant/lingbot-video-dense-1.3b:processor/" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/lingbot-video-dense-1.3b_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing
