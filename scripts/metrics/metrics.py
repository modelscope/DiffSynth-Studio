"""
Image Quality Metrics for SPAD Diffusion Model

Computes various metrics to evaluate image quality:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
- CFID (Conditional Fréchet Inception Distance)

CFID reference: Soloveitchik et al., "Conditional Frechet Inception Distance",
arXiv:2103.11521. Measures distance between conditional distributions P(y|x)
vs P(ŷ|x), which is strictly more informative than FID for conditional
generation tasks like SPAD→RGB.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import linalg
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
        Convert RGB images to grayscale using ITU-R BT.601 weights (same as cv2).

        Args:
            images: Tensor [B, C, H, W] with C=3 (RGB)

        Returns:
            Grayscale images [B, 3, H, W] (replicated across 3 channels for compatibility)
        """
        if images.shape[1] != 3:
            return images  # Already grayscale or wrong format

        # ITU-R BT.601 weights: R=0.299, G=0.587, B=0.114 (same as cv2.COLOR_RGB2GRAY)
        weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
        gray = (images * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return gray.expand(-1, 3, -1, -1)
    
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


class InceptionFeatureExtractor:
    """Extract pool3 (2048-d) features from InceptionV3 for FID/CFID."""

    def __init__(self, device="cuda"):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.device = device
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.eval()
        model.fc = torch.nn.Identity()
        self.model = model.to(device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d features. Input: [B,3,H,W] uint8 or float [0,1]."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = images.to(self.device)
        return self.model(images)


def _symmetric_matrix_sqrt(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute M' such that M' @ M' = mat for symmetric PSD mat."""
    s, u = np.linalg.eigh(mat)
    s = np.maximum(s, eps)
    return (u * np.sqrt(s)) @ u.T


def _trace_sqrt_product(sigma: np.ndarray, sigma_v: np.ndarray) -> float:
    """Tr(sqrt(sigma @ sigma_v)) for symmetric PSD matrices."""
    sqrt_sigma = _symmetric_matrix_sqrt(sigma)
    product = sqrt_sigma @ sigma_v @ sqrt_sigma
    s = np.linalg.eigvalsh(product)
    s = np.maximum(s, 0)
    return float(np.sum(np.sqrt(s)))


def compute_cfid(
    feats_x: np.ndarray,
    feats_y: np.ndarray,
    feats_y_hat: np.ndarray,
    reg: float = 1e-6,
) -> float:
    """Compute Conditional Fréchet Inception Distance.

    Args:
        feats_x:     [N, d] Inception features of conditioning inputs  (SPAD)
        feats_y:     [N, d] Inception features of ground truth outputs (RGB GT)
        feats_y_hat: [N, d] Inception features of generated outputs    (RGB pred)
        reg:         Ridge regularization for C_xx inversion (for N < d stability)

    Returns:
        CFID score (lower is better, non-negative for well-conditioned data).

    Reference: arXiv:2103.11521 Eq. 4
        CFID = ||m_y - m_ŷ||²
             + Tr( (C_yx - C_ŷx) C_xx⁻¹ (C_xy - C_xŷ) )
             + Tr(C_{y|x}) + Tr(C_{ŷ|x}) - 2 Tr(sqrt(C_{ŷ|x} C_{y|x}))
    """
    assert feats_x.shape[0] == feats_y.shape[0] == feats_y_hat.shape[0]
    N, d = feats_x.shape

    # Use float64 internally for numerical stability (2048x2048 matmuls)
    feats_x = feats_x.astype(np.float64)
    feats_y = feats_y.astype(np.float64)
    feats_y_hat = feats_y_hat.astype(np.float64)

    m_x = feats_x.mean(axis=0)
    m_y = feats_y.mean(axis=0)
    m_yh = feats_y_hat.mean(axis=0)

    x_c = feats_x - m_x
    y_c = feats_y - m_y
    yh_c = feats_y_hat - m_yh

    C_xx = (x_c.T @ x_c) / N + reg * np.eye(d)
    C_yy = (y_c.T @ y_c) / N
    C_yhyh = (yh_c.T @ yh_c) / N
    C_yx = (y_c.T @ x_c) / N
    C_xy = (x_c.T @ y_c) / N
    C_yhx = (yh_c.T @ x_c) / N
    C_xyh = (x_c.T @ yh_c) / N

    inv_C_xx = np.linalg.inv(C_xx)

    C_y_given_x = C_yy - C_yx @ inv_C_xx @ C_xy
    C_yh_given_x = C_yhyh - C_yhx @ inv_C_xx @ C_xyh

    C_y_given_x = (C_y_given_x + C_y_given_x.T) / 2
    C_yh_given_x = (C_yh_given_x + C_yh_given_x.T) / 2

    # Clip negative eigenvalues to zero (numerical stability when N < d)
    def _clamp_psd(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return (eigvecs * eigvals) @ eigvecs.T

    C_y_given_x = _clamp_psd(C_y_given_x)
    C_yh_given_x = _clamp_psd(C_yh_given_x)

    term_mean = float(np.sum((m_y - m_yh) ** 2))

    diff_yx = C_yx - C_yhx
    diff_xy = C_xy - C_xyh
    term_cross = float(np.trace(diff_yx @ inv_C_xx @ diff_xy))

    term_cov = (
        float(np.trace(C_y_given_x))
        + float(np.trace(C_yh_given_x))
        - 2.0 * _trace_sqrt_product(C_yh_given_x, C_y_given_x)
    )

    return term_mean + term_cross + term_cov


class CFIDAccumulator:
    """Accumulates Inception features for CFID computation."""

    def __init__(self, device="cuda"):
        self.extractor = InceptionFeatureExtractor(device)
        self.feats_x = []
        self.feats_y = []
        self.feats_y_hat = []

    def update(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        y_hat_batch: torch.Tensor,
    ):
        """Accumulate features for a batch. All inputs: [B,3,H,W] in [0,1] or uint8."""
        self.feats_x.append(self.extractor(x_batch).cpu().numpy())
        self.feats_y.append(self.extractor(y_batch).cpu().numpy())
        self.feats_y_hat.append(self.extractor(y_hat_batch).cpu().numpy())

    def compute(self) -> float:
        """Compute CFID from all accumulated features."""
        feats_x = np.concatenate(self.feats_x, axis=0)
        feats_y = np.concatenate(self.feats_y, axis=0)
        feats_y_hat = np.concatenate(self.feats_y_hat, axis=0)
        return compute_cfid(feats_x, feats_y, feats_y_hat)

    def reset(self):
        self.feats_x.clear()
        self.feats_y.clear()
        self.feats_y_hat.clear()


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
