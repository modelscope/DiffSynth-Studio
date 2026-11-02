# Run this script in the base path of DiffSynth-Studio:
# PYTHONPATH="./" bash examples/qwen_image_21/model_training/special/dmd2_trainer/train_Qwen-Image-2.1-8steps_lora.sh

# Download datasets
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image_21/Qwen-Image-2.1/*" --local_dir ./data/diffsynth_example_dataset

# Calculate the sigmas and timesteps, and fill them into the script below.
python examples/dev_tools/compute_dmd2_sigmas.py --scheduler_name "Qwen-Image" --num_inference_steps 8 --extra_kwargs '{"dynamic_shift_len": 4096}'

# Start training
accelerate launch \
  examples/qwen_image_21/model_training/special/dmd2_trainer/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1/metadata.csv \
  --dataset_repeat 50 \
  --dataset_num_workers 8 --data_file_keys image --num_epochs 5 \
  --max_pixels 1048576 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --lora_base_model dit \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,img_mlp.proj,img_mlp.out,img_mlp.gate_layer,time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,modulation.1,norm_out.linear" \
  --lora_rank 128 \
  --dmd2_student_sigmas "1.0, 0.9160239696502686, 0.8200457692146301, 0.7092946171760559, 0.5800749063491821, 0.42734700441360474, 0.24405395984649658, 0.019999980926513672" \
  --dmd2_student_timesteps "1000.0, 916.0239868164062, 820.0457763671875, 709.2946166992188, 580.0748901367188, 427.3470153808594, 244.053955078125, 19.999980926513672" \
  --dmd2_teacher_cfg 1 \
  --feature_module_of_dit_for_gan transformer_blocks.16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 \
  --save_steps 100 \
  --use_gradient_checkpointing --find_unused_parameters \
  --output_path "./models/train/Qwen-Image-2.1_dmd2_lora_8steps"
