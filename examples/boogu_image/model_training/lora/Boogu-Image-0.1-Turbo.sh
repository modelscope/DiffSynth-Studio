modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "boogu_image/Boogu-Image-0.1-Turbo/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/boogu_image/model_training/train.py \
  --dataset_base_path "./data/diffsynth_example_dataset/boogu_image/Boogu-Image-0.1-Turbo" \
  --dataset_metadata_path "./data/diffsynth_example_dataset/boogu_image/Boogu-Image-0.1-Turbo/metadata.csv" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Boogu/Boogu-Image-0.1-Turbo:transformer/*.safetensors,Boogu/Boogu-Image-0.1-Turbo:mllm/*.safetensors,Boogu/Boogu-Image-0.1-Turbo:vae/*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Boogu-Image-0.1-Turbo_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,img_to_q,img_to_k,img_to_v,img_out,instruct_to_q,instruct_to_k,instruct_to_v,instruct_out" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --data_file_keys "image"
