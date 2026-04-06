#!/bin/bash
# Generate SPAD dataset metadata CSV with train/val split

cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

python scripts/data_prep/prepare_dataset.py \
  --spad_control_dir /home/jw/engsci/thesis/spad/spad_dataset/bits \
  --rgb_ground_truth_dir /home/jw/engsci/thesis/spad/spad_dataset/RGB \
  --output_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata.csv \
  --test_size 0.2 \
  --random_state 42 \
  --prompt ""

echo ""
echo "✅ Dataset prepared!"
echo "   - metadata_train.csv (~80%)"
echo "   - metadata_val.csv (~20%)"
