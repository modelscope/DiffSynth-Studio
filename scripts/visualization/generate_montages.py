#!/usr/bin/env python3
"""
Generate side-by-side comparison montages for all validation images.

Creates a horizontal strip per image: [SPAD Input | Baseline | DPS | Consistency | Ground Truth]
Also generates a multi-seed variance overlay montage.

Usage:
    python generate_montages.py                          # all 776 images
    python generate_montages.py --max-images 50          # first 50 only
    python generate_montages.py --pick-diverse 12        # auto-pick 12 diverse scenes
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Configuration: methods to compare
# ──────────────────────────────────────────────────────────────

METHODS = {
    "Baseline": {
        "output_dir": "validation_outputs_scene_aware/seed_42",
        "output_pattern": "output/output_{idx:04d}.png",
    },
    "DPS η=1.0": {
        "output_dir": "validation_outputs_physics_ablation/dps_eta1.0",
        "output_pattern": "output/output_{idx:04d}.png",
    },
    "Consistency": {
        "output_dir": "validation_outputs_consistency/epoch-0",
        "output_pattern": "output/output_{idx:04d}.png",
    },
    "Consist.+DPS": {
        "output_dir": "validation_outputs_consistency_dps/eta1.0",
        "output_pattern": "output/output_{idx:04d}.png",
    },
}

# Where to find input/GT (from the baseline directory)
INPUT_DIR = "validation_outputs_scene_aware/seed_42"
INPUT_PATTERN = "input/input_{idx:04d}.png"
GT_PATTERN = "ground_truth/gt_{idx:04d}.png"

# Multi-seed directories for variance overlay
SEED_DIRS = [
    "validation_outputs_multiseed/seed_0",
    "validation_outputs_multiseed/seed_13",
    "validation_outputs_multiseed/seed_23",
    "validation_outputs_multiseed/seed_42",
    "validation_outputs_multiseed/seed_55",
    "validation_outputs_multiseed/seed_67",
    "validation_outputs_multiseed/seed_77",
    "validation_outputs_multiseed/seed_88",
    "validation_outputs_multiseed/seed_99",
    "validation_outputs_multiseed/seed_123",
]


def load_image_safe(path, size=None):
    """Load image, return None if not found."""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def compute_variance_map(base_dir, idx, size=None):
    """Compute per-pixel variance across seeds, return as a heatmap image."""
    images = []
    for seed_dir in SEED_DIRS:
        path = os.path.join(base_dir, seed_dir, f"output/output_{idx:04d}.png")
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            images.append(img)
    if len(images) < 2:
        return None
    stack = np.stack(images, axis=0)
    var_map = np.var(stack, axis=0)
    # Normalize to [0, 255] with a fixed scale (max variance = 0.05)
    var_normalized = np.clip(var_map / 0.03, 0, 1)
    # Apply a colormap (hot)
    heatmap = apply_hot_colormap(var_normalized)
    img = Image.fromarray(heatmap)
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def apply_hot_colormap(arr):
    """Apply a hot-like colormap to a [0,1] float array. Returns [H,W,3] uint8."""
    # Simple hot colormap: black -> red -> yellow -> white
    r = np.clip(arr * 3.0, 0, 1)
    g = np.clip(arr * 3.0 - 1.0, 0, 1)
    b = np.clip(arr * 3.0 - 2.0, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def add_label(img, text, font_size=14):
    """Add a text label at the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    # Draw text with background
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (img.width - text_w) // 2
    y = 2
    # Semi-transparent background
    draw.rectangle([x - 3, y - 1, x + text_w + 3, y + text_h + 3], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img


def create_montage(base_dir, idx, img_size=(256, 256), include_variance=True):
    """Create a single montage row for image index `idx`.
    
    Returns: PIL Image (horizontal strip) or None if input missing.
    """
    # Load input and GT
    input_path = os.path.join(base_dir, INPUT_DIR, INPUT_PATTERN.format(idx=idx))
    gt_path = os.path.join(base_dir, INPUT_DIR, GT_PATTERN.format(idx=idx))
    
    input_img = load_image_safe(input_path, size=img_size)
    gt_img = load_image_safe(gt_path, size=img_size)
    
    if input_img is None or gt_img is None:
        return None
    
    # Load method outputs
    method_imgs = {}
    for name, cfg in METHODS.items():
        path = os.path.join(base_dir, cfg["output_dir"], cfg["output_pattern"].format(idx=idx))
        method_imgs[name] = load_image_safe(path, size=img_size)
    
    # Build the strip
    panels = []
    labels = []
    
    # Input
    panels.append(input_img.copy())
    labels.append("SPAD Input")
    
    # Methods
    for name in METHODS:
        img = method_imgs.get(name)
        if img is not None:
            panels.append(img.copy())
        else:
            # Gray placeholder
            panels.append(Image.new("RGB", img_size, (128, 128, 128)))
        labels.append(name)
    
    # Variance (optional)
    if include_variance:
        var_img = compute_variance_map(base_dir, idx, size=img_size)
        if var_img is not None:
            panels.append(var_img)
            labels.append("Seed Var.")
    
    # Ground truth
    panels.append(gt_img.copy())
    labels.append("Ground Truth")
    
    # Add labels
    for i, (panel, label) in enumerate(zip(panels, labels)):
        add_label(panel, label)
    
    # Concatenate horizontally
    total_w = sum(p.width for p in panels)
    montage = Image.new("RGB", (total_w, img_size[1]))
    x_offset = 0
    for panel in panels:
        montage.paste(panel, (x_offset, 0))
        x_offset += panel.width
    
    return montage


def compute_diversity_scores(base_dir, n_images):
    """Score images by diversity: mix of bit density, variance, and PSNR-like spread.
    
    Returns indices sorted by diversity score (most interesting first).
    """
    scores = []
    for idx in range(n_images):
        input_path = os.path.join(base_dir, INPUT_DIR, INPUT_PATTERN.format(idx=idx))
        if not os.path.exists(input_path):
            scores.append((idx, -1))
            continue
        
        img = np.array(Image.open(input_path).convert("L"), dtype=np.float32) / 255.0
        bit_density = img.mean()
        
        # Load baseline output
        baseline_path = os.path.join(
            base_dir, METHODS["Baseline"]["output_dir"],
            METHODS["Baseline"]["output_pattern"].format(idx=idx)
        )
        if os.path.exists(baseline_path):
            out = np.array(Image.open(baseline_path).convert("L"), dtype=np.float32) / 255.0
            complexity = np.std(out)
        else:
            complexity = 0
        
        scores.append((idx, bit_density, complexity))
    
    # Sort into buckets by bit density, pick diverse from each
    scores = [(idx, bd, cx) for idx, bd, cx in scores if bd >= 0]
    scores.sort(key=lambda x: x[1])  # sort by bit density
    
    return scores


def pick_diverse(base_dir, n_images, k):
    """Pick k diverse images spanning the range of bit densities and complexities."""
    scores = compute_diversity_scores(base_dir, n_images)
    if len(scores) <= k:
        return [s[0] for s in scores]
    
    # Stratified sampling: divide into k buckets by bit density
    bucket_size = len(scores) // k
    picked = []
    for i in range(k):
        start = i * bucket_size
        end = start + bucket_size if i < k - 1 else len(scores)
        bucket = scores[start:end]
        # Pick the one with highest complexity (most interesting output)
        bucket.sort(key=lambda x: x[2], reverse=True)
        picked.append(bucket[0][0])
    
    return sorted(picked)


def create_grid_montage(base_dir, indices, img_size=(256, 256), include_variance=True, cols=1):
    """Create a grid montage with multiple rows (one per image index)."""
    rows = []
    for idx in indices:
        row = create_montage(base_dir, idx, img_size=img_size, include_variance=include_variance)
        if row is not None:
            rows.append(row)
    
    if not rows:
        return None
    
    total_h = sum(r.height for r in rows)
    max_w = max(r.width for r in rows)
    grid = Image.new("RGB", (max_w, total_h))
    y_offset = 0
    for row in rows:
        grid.paste(row, (0, y_offset))
        y_offset += row.height
    
    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate comparison montages")
    parser.add_argument("--base-dir", type=str,
                        default="/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD",
                        help="Base directory containing validation output dirs")
    parser.add_argument("--output-dir", type=str, default="./thesis_figures/montages",
                        help="Output directory for montages")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max number of montages to generate (default: all)")
    parser.add_argument("--pick-diverse", type=int, default=None,
                        help="Auto-pick N diverse images for a curated grid")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size of each panel in the montage")
    parser.add_argument("--no-variance", action="store_true",
                        help="Skip variance overlay panel")
    parser.add_argument("--grid-rows", type=int, default=None,
                        help="If set, combine this many rows into a single grid image")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = (args.img_size, args.img_size)
    include_variance = not args.no_variance

    # Count total images
    n_images = len(list(Path(os.path.join(base_dir, INPUT_DIR, "output")).glob("output_*.png")))
    print(f"Found {n_images} validation images")

    # Check which methods are available
    available_methods = {}
    for name, cfg in METHODS.items():
        method_dir = os.path.join(base_dir, cfg["output_dir"], "output")
        if os.path.exists(method_dir):
            count = len(list(Path(method_dir).glob("output_*.png")))
            available_methods[name] = count
            print(f"  ✓ {name}: {count} images")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    # Determine which images to process
    if args.pick_diverse:
        indices = pick_diverse(base_dir, n_images, args.pick_diverse)
        print(f"\nPicked {len(indices)} diverse images: {indices}")
        
        # Generate a single grid image
        grid = create_grid_montage(base_dir, indices, img_size=img_size,
                                    include_variance=include_variance)
        if grid is not None:
            grid_path = out_dir / "curated_comparison_grid.png"
            grid.save(grid_path, quality=95)
            print(f"Curated grid saved to {grid_path}")
            print(f"  Grid size: {grid.width}x{grid.height}")
    else:
        # Generate individual montages for all images
        if args.max_images:
            indices = list(range(min(args.max_images, n_images)))
        else:
            indices = list(range(n_images))
        
        print(f"\nGenerating {len(indices)} montages...")
        
        for idx in tqdm(indices, desc="Generating montages"):
            montage = create_montage(base_dir, idx, img_size=img_size,
                                     include_variance=include_variance)
            if montage is not None:
                montage.save(out_dir / f"montage_{idx:04d}.png", quality=95)
        
        # Also generate a grid of the first 12 as a quick overview
        overview_indices = indices[:12] if len(indices) >= 12 else indices
        grid = create_grid_montage(base_dir, overview_indices, img_size=img_size,
                                    include_variance=include_variance)
        if grid is not None:
            grid.save(out_dir / "overview_grid_first12.png", quality=95)
            print(f"Overview grid (first 12) saved to {out_dir / 'overview_grid_first12.png'}")

    print(f"\nDone! All montages saved to {out_dir}")


if __name__ == "__main__":
    main()
