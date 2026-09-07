modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux/Insert-Anything/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux/Insert-Anything \
  --dataset_metadata_path data/diffsynth_example_dataset/flux/Insert-Anything/metadata.csv \
  --data_file_keys "image,insert_anything_source_image,insert_anything_source_mask,insert_anything_ref_image,insert_anything_ref_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 200 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Fill-dev:flux1-fill-dev.safetensors,black-forest-labs/FLUX.1-Fill-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-Fill-dev:text_encoder_2/*.safetensors,black-forest-labs/FLUX.1-Fill-dev:ae.safetensors,black-forest-labs/FLUX.1-Redux-dev:image_encoder/model.safetensors,black-forest-labs/FLUX.1-Redux-dev:image_embedder/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Insert-Anything_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --extra_inputs "insert_anything_source_image,insert_anything_source_mask,insert_anything_ref_image,insert_anything_ref_mask" \
  --align_to_opensource_format \
  --use_gradient_checkpointing
