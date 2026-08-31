modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/sensenova_u1/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-Edit \
  --dataset_metadata_path data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/metadata.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "SenseNova/SenseNova-U1.5-8B-MoT-SFT:model*.safetensors" \
  --tokenizer_config "SenseNova/SenseNova-U1.5-8B-MoT-SFT:./" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_rank 32 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/SenseNova-U1.5-8B-MoT-SFT-Edit_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj" \
  --use_gradient_checkpointing \
  --find_unused_parameters
