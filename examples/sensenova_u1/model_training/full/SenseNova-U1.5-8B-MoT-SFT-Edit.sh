modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "sensenova_u1/SenseNova-U1.5-8B-MoT-SFT-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch --config_file examples/sensenova_u1/model_training/full/accelerate_config_zero2.yaml examples/sensenova_u1/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-SFT-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-SFT-Edit/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "SenseNova/SenseNova-U1.5-8B-MoT-SFT:model*.safetensors" \
  --tokenizer_config "SenseNova/SenseNova-U1.5-8B-MoT-SFT:./" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/SenseNova-U1.5-8B-MoT-SFT-Edit_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
