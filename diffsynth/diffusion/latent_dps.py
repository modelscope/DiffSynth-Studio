"""
Latent-Space DPS: Measurement guidance that operates entirely in latent space.

Instead of decoding through the VAE at each step (which exceeds 32GB VRAM),
we pre-encode the SPAD measurement through the VAE encoder and compute
a latent-space consistency loss between the predicted clean latent x_0_hat
and the encoded measurement.

Loss: ||x_0_hat - z_spad||^2

NOTE: This is a heuristic approximation, NOT true Bernoulli physics guidance.
The VAE-encoded SPAD binary image has no direct physical meaning in latent
space. For physics-consistent guidance, use flow_dps.py (pixel-space DPS).

Sign convention (rectified flow scheduler):
  The scheduler step is: x_{t+1} = x_t + v * (sigma_{t+1} - sigma_t)
  Since sigma decreases during denoising, (sigma_{t+1} - sigma_t) < 0.
  Adding +grad(loss) to velocity causes latents to move in -grad direction,
  which DECREASES the loss as desired.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LatentDPSConfig:
    """Configuration for latent-space DPS guidance."""
    spad_latent: torch.Tensor = None
    guidance_scale: float = 0.05
    guidance_schedule: str = "ramp_up"
    start_step: int = 0
    stop_step: int = -1


def get_guidance_weight(progress_id, total_steps, schedule, start_step=0, stop_step=-1,
                        sigma=None, sigma_max=None):
    if stop_step < 0:
        stop_step = total_steps
    if progress_id < start_step or progress_id >= stop_step:
        return 0.0
    active_range = stop_step - start_step
    relative_pos = (progress_id - start_step) / max(active_range - 1, 1)

    if schedule == "constant":
        return 1.0
    elif schedule == "linear_decay":
        return 1.0 - relative_pos
    elif schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * relative_pos))
    elif schedule == "ramp_up":
        return relative_pos
    elif schedule == "sigma_ramp":
        if sigma is not None and sigma_max is not None and sigma_max > 0:
            return 1.0 - (sigma / sigma_max)
        return relative_pos
    return 1.0


def compute_latent_dps_correction(
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    sigma: float,
    spad_latent: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Compute latent-space DPS correction without any model forward passes.

    x_0_hat = x_t - sigma * v_theta
    loss = ||x_0_hat - z_spad||^2
    gradient = 2 * (x_0_hat - z_spad)  (trivial, no backprop needed)

    PaDIS-style preconditioning: normalize gradient by mean |grad|.

    Sign: correction = +eta * normalized_grad  (added to velocity)
    Because scheduler does x_next = x + v * (sigma_next - sigma) with sigma_next < sigma,
    adding +grad to v moves latents in -grad direction, decreasing the loss.
    """
    x0_hat = latents - sigma * noise_pred
    diff = x0_hat - spad_latent
    grad = 2.0 * diff

    # PaDIS-style normalization by mean |grad|
    mean_abs = grad.abs().mean() + 1e-8
    grad = grad / mean_abs

    # SIGN: +guidance_scale * grad added to velocity => -grad in latent space ✓
    correction = guidance_scale * grad
    return correction.to(dtype=noise_pred.dtype)
