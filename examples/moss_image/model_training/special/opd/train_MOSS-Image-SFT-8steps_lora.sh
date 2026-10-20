# Run this script in the base path of DiffSynth-Studio:
# PYTHONPATH="./" bash examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_lora.sh

# Download datasets
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "moss_image/MOSS-Image-SFT/*" --local_dir ./data/diffsynth_example_dataset

# Calculate the sigmas and timesteps, and fill them into the script below.
python examples/moss_image/model_training/special/opd/compute_sigmas.py --scheduler_name "MOSS-Image" --num_inference_steps 50 --extra_kwargs '{"shift": 5.0}'
python examples/moss_image/model_training/special/opd/compute_sigmas.py --scheduler_name "MOSS-Image" --num_inference_steps 8 --extra_kwargs '{"shift": 5.0}'

# Start training
accelerate launch examples/moss_image/model_training/special/opd/train_opd_lora.py \
  --dataset_base_path data/diffsynth_example_dataset/moss_image/MOSS-Image-Base \
  --dataset_metadata_path data/diffsynth_example_dataset/moss_image/MOSS-Image-Base/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "openmoss/MOSS-Image-SFT:transformer/diffusion_pytorch_model*.safetensors,openmoss/MOSS-Image-SFT:text_encoder/model*.safetensors,openmoss/MOSS-Image-SFT:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MOSS-Image-SFT_opd_lora_8steps" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 64 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --opd_teacher_sigmas "1.0, 0.9959999918937683, 0.9919999837875366, 0.9869999885559082, 0.9829999804496765, 0.9779999852180481, 0.9729999899864197, 0.9679999947547913, 0.9629999995231628, 0.9570000171661377, 0.9520000219345093, 0.9459999799728394, 0.9390000104904175, 0.9329999685287476, 0.9259999990463257, 0.918999969959259, 0.9120000004768372, 0.9049999713897705, 0.8970000147819519, 0.8880000114440918, 0.8799999952316284, 0.8709999918937683, 0.8610000014305115, 0.8510000109672546, 0.8399999737739563, 0.8289999961853027, 0.8169999718666077, 0.8050000071525574, 0.7910000085830688, 0.7770000100135803, 0.7619999647140503, 0.7459999918937683, 0.7289999723434448, 0.7109999656677246, 0.6919999718666077, 0.6710000038146973, 0.6480000019073486, 0.6229999661445618, 0.597000002861023, 0.5679999589920044, 0.5359999537467957, 0.5009999871253967, 0.46300002932548523, 0.42100003361701965, 0.37300002574920654, 0.32100000977516174, 0.26100000739097595, 0.19300000369548798, 0.11500000208616257, 0.024000000208616257" \
  --opd_teacher_timesteps "1000.0, 996.0, 992.0, 987.0, 983.0, 978.0, 973.0, 968.0, 963.0, 957.0, 952.0, 946.0, 939.0, 932.9999389648438, 926.0, 919.0, 912.0, 905.0, 897.0, 888.0, 880.0, 871.0, 861.0, 851.0, 840.0, 829.0, 817.0, 805.0, 791.0, 777.0, 761.9999389648438, 746.0, 729.0, 710.9999389648438, 692.0, 671.0, 648.0, 622.9999389648438, 597.0, 567.9999389648438, 535.9999389648438, 501.0, 463.0000305175781, 421.0000305175781, 373.0000305175781, 321.0, 261.0, 193.0, 115.0, 24.0" \
  --opd_teacher_cfg 3 \
  --opd_student_sigmas "1.0, 0.9679999947547913, 0.9259999990463257, 0.8709999918937683, 0.7910000085830688, 0.6710000038146973, 0.46300002932548523, 0.024000000208616257" \
  --opd_student_timesteps "1000.0, 968.0, 926.0, 871.0, 791.0, 671.0, 463.0000305175781, 24.0" \
  --opd_student_cfg 1
