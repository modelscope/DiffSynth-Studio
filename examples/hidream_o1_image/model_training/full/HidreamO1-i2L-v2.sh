modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hidream_o1_image/HidreamO1-i2L-v2/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/hidream_o1_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/hidream_o1_image/HidreamO1-i2L-v2 \
  --dataset_metadata_path data/diffsynth_example_dataset/hidream_o1_image/HidreamO1-i2L-v2/metadata.jsonl \
  --max_pixels 4194304 \
  --dataset_repeat 400 \
  --model_id_with_origin_paths "HiDream-ai/HiDream-O1-Image:model-*.safetensors" \
  --processor_config "HiDream-ai/HiDream-O1-Image:./" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --output_path "./models/train/HidreamO1-i2L-v2_full" \
  --use_gradient_checkpointing \
  --noise_scale 8.0 \
  --enable_lora_hot_loading \
  --template_model_id_or_path "DiffSynth-Studio/HidreamO1-i2L-v2:" \
  --extra_inputs "template_inputs" \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --trainable_models "template_model" # Use `template_model.emb2lora` to freeze the image encoder.
