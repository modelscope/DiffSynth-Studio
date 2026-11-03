# Download datasets
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image_21/Qwen-Image-2.1/*" --local_dir ./data/diffsynth_example_dataset

# Calculate the sigmas and timesteps, and fill them into the script below.
python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "Qwen-Image" --num_inference_steps 3 --extra_kwargs '{"dynamic_shift_len": 4096}'

# Start training
accelerate launch examples/qwen_image_21/model_training/special/dmd2/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1/metadata.csv \
  --data_file_keys image \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1-Turbo:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --learning_rate 5e-6 \
  --num_epochs 5 \
  --output_path "./models/train/Qwen-Image-2.1-Turbo_dmd2_lora_3steps" \
  --lora_base_model dit \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,img_mlp.proj,img_mlp.out,img_mlp.gate_layer,time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,modulation.1,norm_out.linear" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --dmd2_teacher_cfg 1 \
  --dmd2_student_sigmas "1.0, 0.6080472469329834, 0.020000040531158447" \
  --dmd2_student_timesteps "1000.0, 608.0472412109375, 20.00004005432129" \
  --dmd2_feature_module_of_dit_for_gan transformer_blocks.16
