modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux/FLUX.1-Fill-dev/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux/FLUX.1-Fill-dev \
  --dataset_metadata_path data/diffsynth_example_dataset/flux/FLUX.1-Fill-dev/metadata.csv \
  --data_file_keys "image,flux_fill_image,flux_fill_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 200 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Fill-dev:flux1-fill-dev.safetensors,black-forest-labs/FLUX.1-Fill-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-Fill-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-Fill-dev:ae.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-Fill-dev_full" \
  --trainable_models "dit" \
  --extra_inputs "flux_fill_image,flux_fill_mask" \
  --use_gradient_checkpointing
