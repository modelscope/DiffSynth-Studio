"""
Run Metrics on Generated Output Directory

This script evaluates a directory of generated images by computing:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)

Expected directory structure:
  output_dir/
    gt_*.png         (ground truth images)
    output_*.png     (generated images)

Or with subdirectories:
  output_dir/
    gt/
      image_0.png
      image_1.png
      ...
    output/
      image_0.png
      image_1.png
      ...

Note: Control/conditioning images are not needed for metrics.
"""

import os
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from metrics import ImageMetrics, MetricsTracker


def load_image(image_path, transform=None):
    """
    Load an image and convert to tensor.
    
    Args:
        image_path: Path to image file
        transform: Optional transform to apply
    
    Returns:
        Image tensor in range [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')
    
    if transform is None:
        # Default transform: to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return transform(img)


def find_images(directory, prefix=None, extensions=None):
    """
    Find all images in a directory with optional prefix filter.
    
    Args:
        directory: Directory to search
        prefix: Optional prefix filter (e.g., 'output_')
        extensions: List of file extensions to match (default: ['.png', '.jpg', '.jpeg'])
        ** NOTE: Use png for best metrics accuracy. **
    
    Returns:
        Sorted list of image paths
    """
    directory = Path(directory)
    
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg']
    
    images = []
    for ext in extensions:
        if prefix:
            pattern = f"{prefix}*{ext}"
        else:
            pattern = f"*{ext}"
        images.extend(directory.glob(pattern))
    
    return sorted(images)


def load_image_pairs(output_dir, batch_size=8, device='cuda'):
    """
    Load image pairs from directory structure for metric computation.
    Only loads output and ground truth images (control images not needed for metrics).
    
    Supports two directory structures:
    1. Flat: gt_*.png, output_*.png
    2. Nested: gt/, output/ subdirectories
    
    Args:
        output_dir: Root directory containing images
        batch_size: Number of images to load per batch
        device: Device to load tensors to
    
    Yields:
        Batches of (output, gt) image tensors
    """
    output_dir = Path(output_dir)
    
    # Detect directory structure
    if (output_dir / 'output').exists() and (output_dir / 'ground_truth').exists():
        # Nested structure
        output_images = find_images(output_dir / 'output')
        gt_images = find_images(output_dir / 'ground_truth')
    else:
        # Flat structure with prefixes
        output_images = find_images(output_dir, prefix='output_')
        gt_images = find_images(output_dir, prefix='gt_')

    if len(output_images) == 0:
        raise ValueError(f"No output images found in {output_dir}")
    if len(gt_images) == 0:
        raise ValueError(f"No ground truth images found in {output_dir}")
    
    # Match images by filename - extract numeric IDs
    def extract_id(filename_stem, prefix):
        """Extract numeric ID from filename by removing prefix and suffix."""
        # Remove prefix
        name = filename_stem.replace(f'{prefix}_', '').replace(prefix, '')
        # Remove common suffixes like _0, _1, etc.
        import re
        # Match patterns like: 000123_0 -> 000123 or 000123 -> 000123
        match = re.match(r'(\d+)', name)
        if match:
            return match.group(1)
        return name
    
    output_names = {extract_id(img.stem, 'output'): img for img in output_images}
    gt_names = {extract_id(img.stem, 'gt'): img for img in gt_images}
    
    # Find common image names
    common_names = sorted(set(output_names.keys()) & set(gt_names.keys()))
    
    if len(common_names) == 0:
        raise ValueError(f"No matching output/gt pairs found in {output_dir}")
    
    print(f"Found {len(common_names)} image pairs")
    
    # Load images in batches
    batch_outputs = []
    batch_gts = []
    
    for name in tqdm(common_names, desc="Loading images"):
        try:
            output_img = load_image(output_names[name])
            gt_img = load_image(gt_names[name])
            
            batch_outputs.append(output_img)
            batch_gts.append(gt_img)
            
            # Yield batch when full
            if len(batch_outputs) == batch_size:
                output_batch = torch.stack(batch_outputs).to(device)
                gt_batch = torch.stack(batch_gts).to(device)
                yield output_batch, gt_batch
                
                batch_outputs = []
                batch_gts = []
        
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue
    
    # Yield remaining images
    if batch_outputs:
        output_batch = torch.stack(batch_outputs).to(device)
        gt_batch = torch.stack(batch_gts).to(device)
        yield output_batch, gt_batch


def compute_metrics_on_directory(output_dir, batch_size=8, device='cuda', compute_fid=True, grayscale=True):
    """
    Compute all metrics on a directory of generated images.
    
    Args:
        output_dir: Directory containing output, gt, and optionally control images
        batch_size: Batch size for processing
        device: Device to run computations on
        compute_fid: Whether to compute FID (requires all images)
        grayscale: Whether to convert images to grayscale before computing metrics
                   (useful when control images are monochrome, e.g., SPAD sensors)
    
    Returns:
        Dictionary with average metrics
    """
    # Initialize metrics
    metrics_calc = ImageMetrics(device=device, compute_fid=compute_fid, grayscale=grayscale)
    tracker = MetricsTracker()
    
    print(f"\nComputing metrics on: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Computing FID: {compute_fid}")
    print(f"Grayscale mode: {grayscale}")
    print("-" * 80)
    
    # Process all image pairs (output and ground truth only)
    with torch.no_grad():
        for output_batch, gt_batch in load_image_pairs(output_dir, batch_size, device):
            # Compute per-image metrics (PSNR, SSIM, LPIPS)
            metrics = metrics_calc.compute_all_metrics(output_batch, gt_batch)
            tracker.update(metrics)
            
            # Update FID (accumulates features)
            if compute_fid:
                metrics_calc.update_fid(output_batch, gt_batch)
    
    # Get average metrics
    avg_metrics = tracker.compute()
    
    # Compute FID
    if compute_fid:
        fid_score = metrics_calc.compute_fid()
        avg_metrics['fid'] = fid_score
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Directory: {output_dir}")
    print(f"Number of images: {sum(tracker.counts.values()) // len(tracker.counts)}")
    print("-" * 80)
    print("Metrics:")
    print(f"  MSE:   {avg_metrics['mse']:.6f}")
    print(f"  PSNR:  {avg_metrics['psnr']:.4f} dB")
    print(f"  SSIM:  {avg_metrics['ssim']:.4f}")
    print(f"  LPIPS: {avg_metrics['lpips']:.4f}")
    if compute_fid:
        print(f"  FID:   {avg_metrics['fid']:.4f}")
    print("=" * 80 + "\n")
    
    return avg_metrics


def save_metrics_to_file(metrics, output_dir, output_file='metrics.txt'):
    """
    Save metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory where output file should be saved
        output_file: Name of output file
    """
    output_path = Path(output_dir) / output_file
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Image Quality Metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Directory: {output_dir}\n\n")
        f.write("Metrics:\n")
        f.write(f"  MSE:   {metrics['mse']:.6f}\n")
        f.write(f"  PSNR:  {metrics['psnr']:.4f} dB\n")
        f.write(f"  SSIM:  {metrics['ssim']:.4f}\n")
        f.write(f"  LPIPS: {metrics['lpips']:.4f}\n")
        if 'fid' in metrics:
            f.write(f"  FID:   {metrics['fid']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics on generated images"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory containing generated images (supports .png, .jpg, .jpeg with gt/ and output/ subdirs or gt_*/output_* prefixed files)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--no-fid',
        action='store_true',
        help='Skip FID computation (faster)'
    )
    parser.add_argument(
        '--color',
        action='store_true',
        help='Compute metrics on color images (default is grayscale for color-invariant evaluation)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save metrics to file'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='metrics.txt',
        help='Output filename for metrics (default: metrics.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory not found: {args.output_dir}")
        return
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Compute metrics
    try:
        metrics = compute_metrics_on_directory(
            args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            compute_fid=not args.no_fid,
            grayscale=not args.color  # Default is grayscale, --color flag disables it
        )
        
        # Save to file if requested
        if args.save:
            save_metrics_to_file(metrics, args.output_dir, args.output_file)
    
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
