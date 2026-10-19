modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "moss_image/MOSS-Image-Base/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/moss_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/moss_image/MOSS-Image-Base \
  --dataset_metadata_path data/diffsynth_example_dataset/moss_image/MOSS-Image-Base/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "openmoss/MOSS-Image-Base:transformer/diffusion_pytorch_model*.safetensors,openmoss/MOSS-Image-Base:text_encoder/model*.safetensors,openmoss/MOSS-Image-Base:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MOSS-Image-Base_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
