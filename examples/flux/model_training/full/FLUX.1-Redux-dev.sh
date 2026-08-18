modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux/FLUX.1-Redux-dev/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux/FLUX.1-Redux-dev \
  --dataset_metadata_path data/diffsynth_example_dataset/flux/FLUX.1-Redux-dev/metadata.csv \
  --data_file_keys "image,flux_redux_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,black-forest-labs/FLUX.1-Redux-dev:image_encoder/model.safetensors,black-forest-labs/FLUX.1-Redux-dev:image_embedder/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.1-Redux-dev_full" \
  --trainable_models "dit" \
  --extra_inputs "flux_redux_image" \
  --use_gradient_checkpointing
