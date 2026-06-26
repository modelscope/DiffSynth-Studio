modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux2/FLUX.2-klein-base-4B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/flux2/model_training/special/dmd2/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-base-4B \
  --dataset_metadata_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-base-4B/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-5 \
  --dmd2_fake_score_learning_rate 1e-5 \
  --dmd2_discriminator_learning_rate 1e-5 \
  --num_epochs 10 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/FLUX.2-klein-base-4B_dmd2" \
  --trainable_models "dit" \
  --task dmd2 \
  --dmd2_student_sample_steps 4 \
  --dmd2_student_sample_type sde \
  --dmd2_student_update_freq 5 \
  --dmd2_gan_loss_weight 0.03 \
  --dmd2_feature_indices 12 \
  --embedded_guidance 4 \
  --dmd2_teacher_cfg_scale 4 \
  --use_gradient_checkpointing