export DATASET_BASE_PATH=/work/lynn9106/00_Reprojection
export METADATA_PATH=$DATASET_BASE_PATH/metadata.csv

# VACE context embedder input channels
#  - 112: vace_video + mask + depth
export DIFFSYNTH_WAN_VACE_IN_DIM=112

accelerate launch --config_file examples/wanvideo/model_training/default_config_4GPU.yaml  examples/wanvideo/model_training/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $METADATA_PATH \
  --data_file_keys "video,vace_video,vace_video_mask,depth_video" \
  --extra_inputs "vace_video,vace_video_mask,depth_video" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --tokenizer_path "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-14B_lora_depth_multikeyframes_4GPU" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing_offload \
  --save_steps 500 \
  --use_wandb \
  --wandb_project_name "wan2.1-VACE-14B" \
  --wandb_run_name "style_da3_depth_multikeyframes_lora32_1e-4_4GPU"
