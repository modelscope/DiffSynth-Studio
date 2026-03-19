"""
Image Quality Metrics for SPAD Diffusion Model

Computes various metrics to evaluate image quality:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


class ImageMetrics:
    """Calculate various image quality metrics."""
    
    def __init__(self, device='cuda', compute_fid=False, grayscale=True):
        """
        Initialize image metrics calculator.
        
        Args:
            device: Device to run computations on
            compute_fid: Whether to compute FID metric
            grayscale: If True, convert images to grayscale before computing metrics.
                      This is useful when control images are monochrome (e.g., SPAD sensors)
                      and color accuracy shouldn't affect the evaluation.
        """
        self.device = device
        self.compute_fid_flag = compute_fid
        self.grayscale = grayscale
        
        # Initialize SSIM metric
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        # Initialize LPIPS metric (perceptual similarity)
        # Uses VGG network by default
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # Initialize FID metric (requires accumulation of features over multiple images)
        # Only initialize if needed since it's memory intensive
        if compute_fid:
            self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    def to_grayscale(self, images):
        """
        Convert RGB images to grayscale using OpenCV's color conversion.
        
        Args:
            images: Tensor [B, C, H, W] with C=3 (RGB)
        
        Returns:
            Grayscale images [B, 3, H, W] (replicated across 3 channels for compatibility)
        """
        if images.shape[1] != 3:
            return images  # Already grayscale or wrong format
        
        # Convert to numpy for cv2 processing
        # images: [B, C, H, W] in range [-1, 1] or [0, 1]
        images_np = images.cpu().numpy()
        
        # Normalize to [0, 1] if needed
        if images_np.min() < 0:
            images_np = (images_np + 1) / 2
        
        # Convert each image in the batch
        gray_images = []
        for i in range(images_np.shape[0]):
            # Get single image: [C, H, W] -> [H, W, C]
            img = np.transpose(images_np[i], (1, 2, 0))
            
            # Convert to uint8 for cv2
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Convert back to [0, 1] float
            gray = gray.astype(np.float32) / 255.0
            
            # Replicate to 3 channels for compatibility with metrics
            gray_rgb = np.stack([gray, gray, gray], axis=0)  # [3, H, W]
            gray_images.append(gray_rgb)
        
        # Stack batch and convert to tensor
        gray_batch = np.stack(gray_images, axis=0)  # [B, 3, H, W]
        gray_tensor = torch.from_numpy(gray_batch).to(images.device)
        
        # Convert back to original range
        if images.min() < 0:
            gray_tensor = gray_tensor * 2 - 1  # [0, 1] -> [-1, 1]
        
        return gray_tensor
    
    def compute_mse(self, pred, target):
        """
        Compute Mean Squared Error.
        
        Args:
            pred: Predicted images [B, C, H, W] in range [-1, 1] or [0, 1]
            target: Target images [B, C, H, W] in range [-1, 1] or [0, 1]
        
        Returns:
            MSE value (scalar)
        """
        mse = F.mse_loss(pred, target)
        return mse.item()
    
    def compute_psnr(self, pred, target, data_range=2.0):
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
            data_range: Range of the data (2.0 for [-1, 1], 1.0 for [0, 1])
        
        Returns:
            PSNR value in dB (scalar)
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        return psnr.item()
    
    def compute_ssim(self, pred, target):
        """
        Compute Structural Similarity Index.
        
        Args:
            pred: Predicted images [B, C, H, W] in range [-1, 1] or [0, 1]
            target: Target images [B, C, H, W] in range [-1, 1] or [0, 1]
        
        Returns:
            SSIM value (scalar, higher is better, range [0, 1])
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2
        
        ssim = self.ssim_metric(pred, target)
        return ssim.item()
    
    def compute_lpips(self, pred, target):
        """
        Compute Learned Perceptual Image Patch Similarity.
        Lower is better (perceptually more similar).
        
        Args:
            pred: Predicted images [B, C, H, W] in range [-1, 1]
            target: Target images [B, C, H, W] in range [-1, 1]
        
        Returns:
            LPIPS value (scalar, lower is better)
        """
        lpips = self.lpips_metric(pred, target)
        return lpips.item()
    
    def update_fid(self, pred, target):
        """
        Update FID metric with a batch of images.
        FID requires accumulating features from multiple images before computing.
        Call compute_fid() after all images have been updated.
        
        Args:
            pred: Predicted images [B, C, H, W] in range [-1, 1] or [0, 1]
            target: Target images [B, C, H, W] in range [-1, 1] or [0, 1]
        """
        if not self.compute_fid_flag:
            raise RuntimeError("FID metric not initialized. Set compute_fid=True in constructor.")
        
        # Convert to grayscale if enabled
        if self.grayscale:
            pred = self.to_grayscale(pred)
            target = self.to_grayscale(target)
        
        # Convert to [0, 255] uint8 as required by FID metric
        def to_uint8(img):
            if img.min() < 0:
                img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = (img * 255).clamp(0, 255).to(torch.uint8)
            return img
        
        pred_uint8 = to_uint8(pred)
        target_uint8 = to_uint8(target)
        
        # Update FID with real and fake images
        self.fid_metric.update(target_uint8, real=True)
        self.fid_metric.update(pred_uint8, real=False)
    
    def compute_fid(self):
        """
        Compute FID score after all images have been updated.
        Lower is better (distributions are more similar).
        
        Returns:
            FID value (scalar, lower is better)
        """
        if not self.compute_fid_flag:
            raise RuntimeError("FID metric not initialized. Set compute_fid=True in constructor.")
        
        fid = self.fid_metric.compute()
        return fid.item()
    
    def reset_fid(self):
        """Reset FID metric state."""
        if self.compute_fid_flag:
            self.fid_metric.reset()
    
    def compute_all_metrics(self, pred, target):
        """
        Compute all metrics at once.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        
        Returns:
            Dictionary with all metrics
        """
        with torch.no_grad():
            # Convert to grayscale if enabled
            if self.grayscale:
                pred = self.to_grayscale(pred)
                target = self.to_grayscale(target)
            
            # Determine data range
            data_range = 2.0 if pred.min() < 0 else 1.0
            
            metrics = {
                'mse': self.compute_mse(pred, target),
                'psnr': self.compute_psnr(pred, target, data_range=data_range),
                'ssim': self.compute_ssim(pred, target),
                'lpips': self.compute_lpips(pred, target),
            }
        
        return metrics
    
    def reset(self):
        """Reset internal metric states."""
        self.ssim_metric.reset()
        # LPIPS doesn't have a state to reset
        if self.compute_fid_flag:
            self.fid_metric.reset()


class MetricsTracker:
    """Track metrics over multiple batches."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict):
        """Update running metrics."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1
    
    def compute(self):
        """Compute average metrics."""
        avg_metrics = {}
        for key in self.metrics:
            avg_metrics[key] = self.metrics[key] / self.counts[key]
        return avg_metrics
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


def compute_validation_metrics(model, dataloader, device='cuda', max_batches=None):
    """
    Compute metrics on a validation dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Validation dataloader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None for all)
    
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    metrics_calc = ImageMetrics(device=device)
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            # Get model predictions
            # This depends on your model's forward pass
            # Adjust as needed for your specific model
            try:
                # Assuming batch has 'jpg' (target) and 'hint' (control)
                target = batch.get('jpg', batch.get('target')).to(device)
                
                # Get model output (you may need to adjust this)
                # For diffusion models, you might want to use the reconstruction
                z, c = model.get_input(batch, model.first_stage_key)
                pred = model.decode_first_stage(z)
                
                # Compute metrics
                metrics = metrics_calc.compute_all_metrics(pred, target)
                tracker.update(metrics)
                
            except Exception as e:
                print(f"Error computing metrics for batch {i}: {e}")
                continue
    
    return tracker.compute()


def log_metrics_to_tensorboard(logger, metrics, step, prefix='val'):
    """
    Log metrics to TensorBoard.
    
    Args:
        logger: PyTorch Lightning logger
        metrics: Dictionary of metrics
        step: Current step/epoch
        prefix: Prefix for metric names (e.g., 'val', 'train')
    """
    for key, value in metrics.items():
        logger.experiment.add_scalar(f'{prefix}/{key}', value, step)
