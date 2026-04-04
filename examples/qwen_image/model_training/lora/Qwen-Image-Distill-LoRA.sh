accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_distill_qwen_image.csv \
  --data_file_keys "image" \
  --extra_inputs "seed,rand_device,num_inference_steps,cfg_scale" \
  --height 1328 \
  --width 1328 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Distill-LoRA_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task direct_distill

# This is an experimental training feature designed to directly distill the model, enabling generation results with fewer steps to approximate those achieved with more steps.
# The model (https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA) is trained using this script.
# The sample dataset is provided solely to demonstrate the dataset format. For actual usage, please construct a larger dataset using the base model.
