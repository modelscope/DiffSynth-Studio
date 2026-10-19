# Base: replace MOSS-Image-SFT with MOSS-Image-Base in the dataset and model paths.
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "moss_image/MOSS-Image-SFT/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/moss_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/moss_image/MOSS-Image-SFT \
  --dataset_metadata_path data/diffsynth_example_dataset/moss_image/MOSS-Image-SFT/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "DiffSynth-Studio/MOSS-Image-EntroPack:models/MOSS-Image-SFT/dit_6bit.safetensors,DiffSynth-Studio/MOSS-Image-EntroPack:models/text_encoder_6bit.safetensors,openmoss/MOSS-Image-SFT:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MOSS-Image-entropack_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
