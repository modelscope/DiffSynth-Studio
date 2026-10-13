# Run this script in the base path of DiffSynth-Studio:
# PYTHONPATH="./" bash examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-Turbo-3steps_lora.sh

# Download datasets
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image_21/Qwen-Image-2.1/*" --local_dir ./data/diffsynth_example_dataset

# Start training
accelerate launch examples/qwen_image_21/model_training/special/opd/train_opd_lora.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1 \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image_21/Qwen-Image-2.1/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 40 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2.1-Turbo:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors,Qwen/Qwen-Image-2.1:vae/diffusion_pytorch_model*.safetensors" \
  --learning_rate 2e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-2.1-Turbo_opd_lora_3steps" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,img_mlp.proj,img_mlp.out,img_mlp.gate_layer,time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,modulation.1,norm_out.linear" \
  --lora_rank 128 \
  --save_steps 100 \
  --use_gradient_checkpointing \
  --opd_teacher_sigmas "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568" \
  --opd_teacher_timesteps "1000.0, 978.453, 954.18, 926.626, 895.08, 845.148, 704.534, 414.568" \
  --opd_teacher_cfg 2 \
  --opd_student_sigmas "1.0, 0.845148, 0.414568" \
  --opd_student_timesteps "1000.0, 845.148, 414.568" \
  --opd_student_cfg 1
