modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "krea2/Krea-2-Raw/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/krea2/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/krea2/Krea-2-Raw \
  --dataset_metadata_path data/diffsynth_example_dataset/krea2/Krea-2-Raw/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "krea/Krea-2-Raw:raw.safetensors,Qwen/Qwen3-VL-4B-Instruct:*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "Qwen/Qwen3-VL-4B-Instruct:" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Krea-2-Raw_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "wq,wk,wv,gate,wo,gate,up,down,first,tmlp.0,tmlp.2,projector,txtmlp.1,txtmlp.3,last.linear,tproj.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --align_to_opensource_format
