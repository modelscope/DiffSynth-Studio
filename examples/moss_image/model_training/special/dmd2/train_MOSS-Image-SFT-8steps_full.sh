modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "moss_image/MOSS-Image-SFT/*" --local_dir ./data/diffsynth_example_dataset

python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "MOSS-Image" --num_inference_steps 8 --extra_kwargs '{"shift": 5.0}'

accelerate launch examples/moss_image/model_training/special/dmd2/train.py \
  --dataset_base_path data/diffsynth_example_dataset/moss_image/MOSS-Image-SFT \
  --dataset_metadata_path data/diffsynth_example_dataset/moss_image/MOSS-Image-SFT/metadata.csv \
  --data_file_keys image \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "openmoss/MOSS-Image-SFT:transformer/diffusion_pytorch_model*.safetensors,openmoss/MOSS-Image-SFT:text_encoder/model*.safetensors,openmoss/MOSS-Image-SFT:vae/diffusion_pytorch_model.safetensors" \
  --trainable_models dit \
  --learning_rate 5e-6 \
  --num_epochs 5 \
  --output_path "workspace/$(date -u +%Y%m%d_%H%M%S)_moss_dmd2_full_8steps" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --dmd2_teacher_cfg 3 \
  --dmd2_student_sigmas "1.0, 0.9679999947547913, 0.9259999990463257, 0.8709999918937683, 0.7910000085830688, 0.6710000038146973, 0.46300002932548523, 0.024000000208616257" \
  --dmd2_student_timesteps "1000.0, 968.0, 926.0, 871.0, 791.0, 671.0, 463.0000305175781, 24.0" \
  --dmd2_feature_module_of_dit_for_gan layers.21
