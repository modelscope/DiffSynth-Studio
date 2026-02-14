#export PYTHONPATH="/pfs/mahaocong/DiffSynth-Studio:$PYTHONPATH"
accelerate launch  --num_processes 2 --config_file examples/wanvideo/model_training/full/accelerate_config_zero2_usp.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path TestVidData/video_1sampe \
  --dataset_metadata_path TestVidData/video_1sampe/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --sp_size 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image"
