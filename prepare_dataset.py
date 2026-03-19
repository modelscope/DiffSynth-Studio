"""
Generate metadata.csv for FLUX ControlNet training from existing SPAD dataset.

Simple approach: Just create CSV with paths to existing PNG files.
The 16-bit grayscale → RGB conversion will happen on-the-fly during training.
"""

import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def match_spad_to_rgb(spad_dir, rgb_dir):
    """
    Match SPAD control images to RGB ground truth by parsing filenames.
    
    Example filenames:
        SPAD: 0724-dgp-001_RAW_empty_frames0-9_p.png
        RGB:  0724-dgp-001_frames0-19999_linear16.png
    
    Match on: "0724-dgp-001"
    """
    # Get all RGB files
    rgb_files = sorted(Path(rgb_dir).glob("*.png"))
    rgb_dict = {}
    for rgb_file in rgb_files:
        # Extract scene ID: "0724-dgp-001"
        parts = rgb_file.stem.split("_frames")
        if len(parts) >= 1:
            scene_id = parts[0]
            rgb_dict[scene_id] = rgb_file
    
    # Get all SPAD files
    spad_files = sorted(Path(spad_dir).glob("*.png"))
    matched_pairs = []
    
    for spad_file in spad_files:
        # Extract scene ID from SPAD filename
        parts = spad_file.stem.split("_RAW")
        if len(parts) >= 1:
            scene_id = parts[0]
            if scene_id in rgb_dict:
                matched_pairs.append((spad_file, rgb_dict[scene_id]))
    
    return matched_pairs


def generate_metadata_csv(
    spad_control_dir,
    rgb_ground_truth_dir,
    output_csv,
    prompt="",
    test_size=0.2,
    random_state=42,
    use_relative_paths=True,
):
    """
    Generate metadata.csv with paths to existing PNG files.
    Uses sklearn train_test_split to match SD1.5 ControlNet setup EXACTLY.
    
    Args:
        spad_control_dir: Directory with SPAD control images (16-bit grayscale PNG)
        rgb_ground_truth_dir: Directory with RGB ground truth (8-bit RGB PNG)
        output_csv: Output CSV file path
        prompt: Default text prompt for all samples
        test_size: Fraction of data for validation (default: 0.2, matching SD1.5)
        random_state: Random seed for reproducible splits (default: 42, matching SD1.5)
        use_relative_paths: If True, use paths relative to CSV location
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Match SPAD to RGB
    print(f"Matching SPAD controls to RGB ground truth...")
    matched_pairs = match_spad_to_rgb(spad_control_dir, rgb_ground_truth_dir)
    print(f"Found {len(matched_pairs)} matched pairs")
    
    if len(matched_pairs) == 0:
        print("ERROR: No matched pairs found!")
        print("Check that filenames follow the pattern:")
        print("  SPAD: SCENE-ID_RAW_empty_frames*_p.png")
        print("  RGB:  SCENE-ID_frames*_linear16.png")
        return
    
    # Prepare paths
    metadata = []
    base_path = output_csv.parent if use_relative_paths else None
    
    for spad_path, rgb_path in matched_pairs:
        if use_relative_paths:
            # Make paths relative to CSV location
            try:
                spad_rel = os.path.relpath(spad_path, base_path)
                rgb_rel = os.path.relpath(rgb_path, base_path)
            except ValueError:
                # If on different drives (Windows), use absolute paths
                spad_rel = str(spad_path.absolute())
                rgb_rel = str(rgb_path.absolute())
        else:
            spad_rel = str(spad_path.absolute())
            rgb_rel = str(rgb_path.absolute())
        
        metadata.append({
            'image': rgb_rel,
            'prompt': prompt,
            # FLUX ControlNet expects `controlnet_image`
            'controlnet_image': spad_rel,
        })
    
    # Use sklearn train_test_split - EXACTLY like SD1.5 ControlNet setup
    metadata_train, metadata_val = train_test_split(
        metadata,
        test_size=test_size,
        random_state=random_state
    )
    print(f"\n[Split] Using sklearn.train_test_split (test_size={test_size}, random_state={random_state})")
    print(f"[Split] Train: {len(metadata_train)} samples")
    print(f"[Split] Val:   {len(metadata_val)} samples")
    
    # Write CSVs
    fieldnames = ['image', 'prompt', 'controlnet_image']
    
    # Training CSV
    train_csv = output_csv.parent / f"{output_csv.stem}_train.csv"
    with open(train_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_train)
    
    # Validation CSV
    val_csv = output_csv.parent / f"{output_csv.stem}_val.csv"
    with open(val_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_val)
    
    # Combined CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n✅ Metadata CSV generated!")
    print(f"   Training samples: {len(metadata_train)}")
    print(f"   Validation samples: {len(metadata_val)}")
    print(f"   Total samples: {len(metadata)}")
    print(f"\n   Output files:")
    print(f"     - {output_csv} (all data)")
    print(f"     - {train_csv} (training)")
    print(f"     - {val_csv} (validation)")
    print(f"\nNext steps:")
    print(f"1. The 16-bit grayscale SPAD images will be converted to RGB on-the-fly during training")
    print(f"2. Point your training script (e.g., train_lora.sh) to:")
    print(f"   --dataset_base_path \"{output_csv.parent.absolute()}\"")
    print(f"   --dataset_metadata_path \"{train_csv.name}\"")
    print(f"3. Use metadata_val.csv for validation scripts")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata.csv for FLUX ControlNet training (simple version - no file duplication)"
    )
    parser.add_argument(
        "--spad_control_dir",
        type=str,
        required=True,
        help="Directory with SPAD control images (e.g., bits_10frames/)"
    )
    parser.add_argument(
        "--rgb_ground_truth_dir",
        type=str,
        required=True,
        help="Directory with RGB ground truth (e.g., RGB/)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV file path (e.g., dataset/metadata.csv)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Default text prompt for all samples (empty string recommended)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2, matching SD1.5)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42, matching SD1.5)"
    )
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Use absolute paths instead of relative paths"
    )
    
    args = parser.parse_args()
    
    generate_metadata_csv(
        spad_control_dir=args.spad_control_dir,
        rgb_ground_truth_dir=args.rgb_ground_truth_dir,
        output_csv=args.output_csv,
        prompt=args.prompt,
        test_size=args.test_size,
        random_state=args.random_state,
        use_relative_paths=not args.absolute_paths,
    )


if __name__ == "__main__":
    main()
