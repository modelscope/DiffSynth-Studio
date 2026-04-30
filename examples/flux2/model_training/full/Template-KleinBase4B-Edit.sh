modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux2/Template-KleinBase4B-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux2/Template-KleinBase4B-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/flux2/Template-KleinBase4B-Edit/metadata.jsonl \
  --extra_inputs "template_inputs" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --template_model_id_or_path "DiffSynth-Studio/Template-KleinBase4B-Edit:" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.template_model." \
  --output_path "./models/train/Template-KleinBase4B-Edit_full" \
  --trainable_models "template_model" \
  --use_gradient_checkpointing \
  --find_unused_parameters
