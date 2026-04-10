modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "joyai_image/JoyAI-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/joyai_image/model_training/full/accelerate_config_zero2offload.yaml examples/joyai_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv \
  --data_file_keys "image,input_image" \
  --extra_inputs "input_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth,jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/JoyAI-Image-Edit_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
