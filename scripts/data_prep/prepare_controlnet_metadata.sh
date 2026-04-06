#!/bin/bash
# Update CSV to use controlnet_image column instead of input_image

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

python - << 'EOF'
import pandas as pd
from pathlib import Path

dataset_path = Path("/home/jw/engsci/thesis/spad/spad_dataset")

# Read current metadata
df_train = pd.read_csv(dataset_path / "metadata_train.csv")
df_val = pd.read_csv(dataset_path / "metadata_val.csv")

# Rename column: input_image -> controlnet_image
df_train = df_train.rename(columns={"input_image": "controlnet_image"})
df_val = df_val.rename(columns={"input_image": "controlnet_image"})

# Keep only needed columns
df_train = df_train[["image", "prompt", "controlnet_image"]]
df_val = df_val[["image", "prompt", "controlnet_image"]]

# Save
df_train.to_csv(dataset_path / "metadata_train.csv", index=False)
df_val.to_csv(dataset_path / "metadata_val.csv", index=False)

print(f"✓ Updated metadata files:")
print(f"  Train: {len(df_train)} samples")
print(f"  Val: {len(df_val)} samples")
print(f"  Columns: {list(df_train.columns)}")
EOF





