modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --echo_memory_model_id_with_origin_path "Echo-Team/Echo-Memory:context_k1/epoch-0.safetensors" \
  --echo_memory_download_source "huggingface" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --output_path "./models/train/Echo-Memory_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters
