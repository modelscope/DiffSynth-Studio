# 训练测试：验证「对已预量化(NF4)权重传 --quant_options 会被忽略并 warning」
#
# 期望现象：
#   - 加载 transformer(minimax-h3-ref2va-nf4.safetensors) 时打印：
#       Warning: `...` is already a pre-quantized checkpoint; ignoring the dynamic
#       quantization option (method=`bitsandbytes_nf4`) and loading it as pre-quantized.
#   - 该模型按预量化(load_prequantized)加载，不会再跑在线量化(不打印 "N nn.Linear layers quantized")。
#   - LoRA 训练在预量化基座上正常进行。
#
# 用小 dataset_repeat / 1 epoch 做快速冒烟测试。

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Ref2VA/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-text-encoder-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-ref2va-nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:video_vae_nf4.safetensors,DiffSynth-Studio/MiniMax-H3-NF4:audio_vae_nf4.safetensors" \
  --quant_options "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-ref2va-nf4.safetensors:bitsandbytes_nf4" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-NF4-Ref2VA-quant-skip-test" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters
