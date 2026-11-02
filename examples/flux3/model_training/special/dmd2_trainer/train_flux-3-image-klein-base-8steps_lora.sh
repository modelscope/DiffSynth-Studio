# Run this script in the base path of DiffSynth-Studio:
# PYTHONPATH="./" bash examples/flux3/model_training/special/dmd2_trainer/train_flux-3-image-klein-base-8steps_lora.sh

# Download datasets
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux3/flux-3-image-klein-base/*" --local_dir ./data/diffsynth_example_dataset

# Calculate the sigmas and timesteps, and fill them into the script below.
python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "FLUX.3" --num_inference_steps 8 --extra_kwargs '{"alpha": 6.93}'

# Start training
accelerate launch examples/flux3/model_training/special/dmd2_trainer/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux3/flux-3-image-klein-base \
  --dataset_metadata_path data/diffsynth_example_dataset/flux3/flux-3-image-klein-base/metadata.csv \
  --data_file_keys image \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/flux-3-image-klein-base:model.safetensors,black-forest-labs/flux-3-image-klein:text_encoder/*.safetensors,black-forest-labs/flux-3-image-klein:vae/diffusion_pytorch_model.safetensors" \
  --processor_path "black-forest-labs/flux-3-image-klein:text_encoder/" \
  --learning_rate 5e-6 \
  --num_epochs 5 \
  --output_path "./models/train/flux-3-image-klein-base_dmd2_lora_8steps" \
  --lora_base_model dit \
  --lora_target_modules "q_proj,k_proj,v_proj,attn_out,mlp_in,mlp_out" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --dmd2_teacher_cfg 3.5 \
  --dmd2_student_sigmas "1.0, 0.9798020720481873, 0.954107403755188, 0.920318603515625, 0.873896598815918, 0.8061264157295227, 0.6978851556777954, 0.4974874258041382" \
  --dmd2_student_timesteps "1000.0, 979.8020629882812, 954.107421875, 920.318603515625, 873.8966064453125, 806.1264038085938, 697.8851318359375, 497.4874267578125" \
  --feature_module_of_dit_for_gan single_blocks.13
