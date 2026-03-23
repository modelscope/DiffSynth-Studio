modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.1-Fun-V1.1-14B-Control-Camera/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-Fun-V1.1-14B-Control-Camera \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-Fun-V1.1-14B-Control-Camera/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:diffusion_pytorch_model*.safetensors,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_t5_umt5-xxl-enc-bf16.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:Wan2.1_VAE.pth,PAI/Wan2.1-Fun-V1.1-14B-Control-Camera:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control-Camera_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image,camera_control_direction,camera_control_speed"
