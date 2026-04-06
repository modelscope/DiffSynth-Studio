"""
Run Metrics on Generated Output Directory

This script evaluates a directory of generated images by computing:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
- CFID (Conditional Fréchet Inception Distance) -- uses the input/conditioning
  images to measure whether generated outputs are faithful to their specific
  inputs, not just realistic in general.

Expected directory structure (nested):
  output_dir/
    input/        (conditioning SPAD images)
    output/       (generated images)
    ground_truth/ (ground truth RGB images)
"""

import os
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from metrics import ImageMetrics, MetricsTracker, CFIDAccumulator
from diffsynth.diffusion.spad_forward import SPADForwardModel, srgb_to_linear


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


def _extract_id(filename_stem, prefix):
    """Extract numeric ID from filename by removing prefix and suffix."""
    import re
    name = filename_stem.replace(f'{prefix}_', '').replace(prefix, '')
    match = re.match(r'(\d+)', name)
    return match.group(1) if match else name


def load_image_triplets(output_dir, batch_size=8, device='cuda'):
    """
    Load image triplets (input, output, gt) from directory structure.
    If input/ subdirectory exists, yields (output, gt, input) batches for CFID.
    Otherwise yields (output, gt, None).

    Args:
        output_dir: Root directory containing images
        batch_size: Number of images to load per batch
        device: Device to load tensors to

    Yields:
        Batches of (output_tensor, gt_tensor, input_tensor_or_None)
    """
    output_dir = Path(output_dir)

    if (output_dir / 'output').exists() and (output_dir / 'ground_truth').exists():
        output_images = find_images(output_dir / 'output')
        gt_images = find_images(output_dir / 'ground_truth')
        input_dir = output_dir / 'input'
        input_images = find_images(input_dir) if input_dir.exists() else []
    else:
        output_images = find_images(output_dir, prefix='output_')
        gt_images = find_images(output_dir, prefix='gt_')
        input_images = find_images(output_dir, prefix='input_')

    if not output_images:
        raise ValueError(f"No output images found in {output_dir}")
    if not gt_images:
        raise ValueError(f"No ground truth images found in {output_dir}")

    output_names = {_extract_id(img.stem, 'output'): img for img in output_images}
    gt_names = {_extract_id(img.stem, 'gt'): img for img in gt_images}
    input_names = {_extract_id(img.stem, 'input'): img for img in input_images}

    common_names = sorted(set(output_names.keys()) & set(gt_names.keys()))
    if not common_names:
        raise ValueError(f"No matching output/gt pairs found in {output_dir}")

    has_inputs = bool(input_names)
    if has_inputs:
        common_names = [n for n in common_names if n in input_names]
        print(f"Found {len(common_names)} image triplets (input + output + gt) → CFID enabled")
    else:
        print(f"Found {len(common_names)} image pairs (output + gt) — no input/ dir, CFID skipped")

    batch_out, batch_gt, batch_inp = [], [], []

    for name in tqdm(common_names, desc="Loading images"):
        try:
            batch_out.append(load_image(output_names[name]))
            batch_gt.append(load_image(gt_names[name]))
            if has_inputs:
                batch_inp.append(load_image(input_names[name]))

            if len(batch_out) == batch_size:
                out_t = torch.stack(batch_out).to(device)
                gt_t = torch.stack(batch_gt).to(device)
                inp_t = torch.stack(batch_inp).to(device) if has_inputs else None
                yield out_t, gt_t, inp_t
                batch_out, batch_gt, batch_inp = [], [], []
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            continue

    if batch_out:
        out_t = torch.stack(batch_out).to(device)
        gt_t = torch.stack(batch_gt).to(device)
        inp_t = torch.stack(batch_inp).to(device) if has_inputs else None
        yield out_t, gt_t, inp_t


def compute_metrics_on_directory(output_dir, batch_size=8, device='cuda', compute_fid=True, grayscale=True):
    """
    Compute all metrics on a directory of generated images.
    Automatically computes CFID when an input/ subdirectory is present.
    
    Args:
        output_dir: Directory containing output, gt, and optionally input images
        batch_size: Batch size for processing
        device: Device to run computations on
        compute_fid: Whether to compute FID (requires all images)
        grayscale: Whether to convert images to grayscale before computing metrics
    
    Returns:
        Dictionary with average metrics
    """
    metrics_calc = ImageMetrics(device=device, compute_fid=compute_fid, grayscale=grayscale)
    tracker = MetricsTracker()
    cfid_acc = None
    spad_model = SPADForwardModel(alpha=1.0, beta=0.0, num_frames=1)
    nll_values = []

    print(f"\nComputing metrics on: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Computing FID: {compute_fid}")
    print(f"Grayscale mode: {grayscale}")
    print("-" * 80)

    with torch.no_grad():
        for output_batch, gt_batch, input_batch in load_image_triplets(output_dir, batch_size, device):
            metrics = metrics_calc.compute_all_metrics(output_batch, gt_batch)
            tracker.update(metrics)

            if compute_fid:
                metrics_calc.update_fid(output_batch, gt_batch)

            if input_batch is not None:
                if cfid_acc is None:
                    cfid_acc = CFIDAccumulator(device=device)

                def _to_01(t):
                    return (t + 1) / 2 if t.min() < 0 else t

                out_01 = _to_01(output_batch)
                gt_01 = _to_01(gt_batch)
                inp_01 = _to_01(input_batch)

                cfid_acc.update(inp_01, gt_01, out_01)

                # Measurement NLL: forward-model the output through SPAD physics
                # and compare to the actual SPAD input observation
                linear = srgb_to_linear(out_01)
                gray = linear.mean(dim=1, keepdim=True)  # [B,1,H,W]
                inp_gray = inp_01.mean(dim=1, keepdim=True)  # SPAD is mono, avg RGB channels
                H = spad_model.intensity_to_exposure(gray)
                log_p = torch.log(-torch.expm1(-H))
                log_1mp = -H
                # Per-pixel NLL, then mean over pixels and batch
                nll_map = -(inp_gray * log_p + (1.0 - inp_gray) * log_1mp)
                nll_per_image = nll_map.flatten(1).mean(1)  # mean over pixels
                nll_values.extend(nll_per_image.cpu().tolist())

    avg_metrics = tracker.compute()

    if compute_fid:
        avg_metrics['fid'] = metrics_calc.compute_fid()

    if cfid_acc is not None:
        avg_metrics['cfid'] = cfid_acc.compute()

    if nll_values:
        avg_metrics['measurement_nll'] = float(np.mean(nll_values))

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
    if 'fid' in avg_metrics:
        print(f"  FID:   {avg_metrics['fid']:.4f}")
    if 'cfid' in avg_metrics:
        print(f"  CFID:  {avg_metrics['cfid']:.4f}")
    if 'measurement_nll' in avg_metrics:
        print(f"  Meas.NLL: {avg_metrics['measurement_nll']:.6f}  (per-pixel, lower=more physically consistent)")
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
        if 'cfid' in metrics:
            f.write(f"  CFID:  {metrics['cfid']:.4f}\n")
        if 'measurement_nll' in metrics:
            f.write(f"  Meas.NLL: {metrics['measurement_nll']:.6f}\n")
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
            # Also save JSON for programmatic access
            import json
            json_path = Path(args.output_dir) / 'metrics.json'
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"JSON saved to: {json_path}")
    
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
