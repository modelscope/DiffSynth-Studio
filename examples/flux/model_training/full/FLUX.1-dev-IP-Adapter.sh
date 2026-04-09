modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux/FLUX.1-dev-IP-Adapter/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux/FLUX.1-dev-IP-Adapter \
  --dataset_metadata_path data/diffsynth_example_dataset/flux/FLUX.1-dev-IP-Adapter/metadata.csv \
  --data_file_keys "image,ipadapter_images" \
  --max_pixels 1048576 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-dev:ae.safetensors,InstantX/FLUX.1-dev-IP-Adapter:ip-adapter.bin,google/siglip-so400m-patch14-384:model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.ipadapter." \
  --output_path "./models/train/FLUX.1-dev-IP-Adapter_full" \
  --trainable_models "ipadapter" \
  --extra_inputs "ipadapter_images" \
  --use_gradient_checkpointing
