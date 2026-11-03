modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "flux2/FLUX.2-klein-base-4B/*" --local_dir ./data/diffsynth_example_dataset

python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "FLUX.2" --num_inference_steps 8 --extra_kwargs '{"dynamic_shift_len": 4096}'

accelerate launch examples/flux2/model_training/special/dmd2/train.py \
  --dataset_base_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-base-4B \
  --dataset_metadata_path data/diffsynth_example_dataset/flux2/FLUX.2-klein-base-4B/metadata.csv \
  --data_file_keys image \
  --height 1024 --width 1024 \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-4B:tokenizer/" \
  --trainable_models dit \
  --learning_rate 5e-6 \
  --num_epochs 5 \
  --output_path "workspace/$(date -u +%Y%m%d_%H%M%S)_flux2_klein_base4b_dmd2_full_8steps" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --dmd2_teacher_cfg 4 \
  --dmd2_student_sigmas "1.0, 0.9854254722595215, 0.9666411280632019, 0.9415147304534912, 0.9061826467514038, 0.8528416752815247, 0.7630144953727722, 0.5798068046569824" \
  --dmd2_student_timesteps "1000.0, 985.4254760742188, 966.64111328125, 941.5147094726562, 906.1826171875, 852.8416748046875, 763.0144653320312, 579.8068237304688" \
  --dmd2_feature_module_of_dit_for_gan single_transformer_blocks.7
