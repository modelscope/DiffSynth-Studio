# Dataset: data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/
# Download: modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "joyai_image/JoyAI-Image-Edit/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/joyai_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/metadata.csv" \
    --max_pixels 1048576 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:JoyAI-Image-Und/model*.safetensors,jd-opensource/JoyAI-Image-Edit:vae/Wan2.1_VAE.pth" \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/JoyAI-Image-Edit-full-cache" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --task "sft:data_process"

accelerate launch --config_file examples/joyai_image/model_training/full/accelerate_config_zero3.yaml \
    examples/joyai_image/model_training/train.py \
    --dataset_base_path "./models/train/JoyAI-Image-Edit-full-cache" \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --model_id_with_origin_paths "jd-opensource/JoyAI-Image-Edit:transformer/transformer.pth" \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/JoyAI-Image-Edit-full" \
    --trainable_models "dit" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --task "sft:train"
