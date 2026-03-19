"""
Phase 3a: Differentiable SPAD Forward Model

Implements the Bernoulli measurement model for single-photon avalanche diode sensors.

Physical model:
  - Scene irradiance I(x) at pixel x
  - Detection probability: p(x) = 1 - exp(-alpha * I(x))
  - Single binary frame: b(x) ~ Bernoulli(p(x))
  - N accumulated frames: y(x) = sum_{i=1}^{N} b_i(x) ~ Binomial(N, p(x))

For the single-frame case (N=1), b in {0, 1}:
  log p(b | I) = b * log(p) + (1-b) * log(1-p)
               = b * log(1 - exp(-alpha*I)) + (1-b) * (-alpha*I)

For the multi-frame case:
  log p(y | I) = y * log(p) + (N-y) * log(1-p)  (up to binomial coefficient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADForwardModel(nn.Module):
    """Differentiable SPAD measurement model with Bernoulli likelihood."""

    def __init__(self, alpha: float = 1.0, num_frames: int = 1, eps: float = 1e-6):
        """
        Args:
            alpha: Sensor sensitivity parameter. Can be estimated from data or learned.
            num_frames: Number of accumulated binary frames (N).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.num_frames = num_frames
        self.eps = eps

    def intensity_to_detection_prob(self, intensity: torch.Tensor) -> torch.Tensor:
        """Convert scene irradiance to SPAD detection probability.

        p = 1 - exp(-alpha * I), clamped for numerical stability.
        """
        intensity = intensity.clamp(min=0.0)
        p = 1.0 - torch.exp(-self.alpha * intensity)
        return p.clamp(self.eps, 1.0 - self.eps)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        """Simulate SPAD measurement: intensity -> expected detection count.

        For N=1, returns detection probability (expected binary value).
        For N>1, returns expected count = N * p.
        """
        p = self.intensity_to_detection_prob(intensity)
        return p * self.num_frames

    def log_likelihood(
        self, intensity: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p(measurement | intensity) under the Bernoulli/Binomial model.

        Args:
            intensity: Predicted scene irradiance [B, C, H, W] in [0, 1].
            measurement: Observed SPAD output [B, C, H, W].
                For N=1: binary {0, 1}.
                For N>1: counts in [0, N], normalized to [0, 1] by dividing by N.

        Returns:
            Log-likelihood per sample [B].
        """
        p = self.intensity_to_detection_prob(intensity)

        if self.num_frames == 1:
            ll = measurement * torch.log(p) + (1.0 - measurement) * torch.log(1.0 - p)
        else:
            y = measurement * self.num_frames
            n_minus_y = self.num_frames - y
            ll = y * torch.log(p) + n_minus_y * torch.log(1.0 - p)

        return ll.flatten(1).sum(1)

    def negative_log_likelihood(
        self, intensity: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        """NLL loss: -log p(y | I), averaged over batch."""
        return -self.log_likelihood(intensity, measurement).mean()

    def measurement_loss(
        self, intensity: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        """Measurement consistency loss for DPS guidance.

        Combines NLL with an L2 term comparing predicted vs observed counts.
        """
        nll = self.negative_log_likelihood(intensity, measurement)
        predicted = self.forward(intensity) / max(self.num_frames, 1)
        l2 = F.mse_loss(predicted, measurement)
        return nll + l2

    @staticmethod
    def estimate_alpha_from_data(
        rgb_images: torch.Tensor, spad_measurements: torch.Tensor, num_frames: int = 1
    ) -> float:
        """Estimate alpha from paired (RGB, SPAD) data using MLE.

        From p = y/N and p = 1 - exp(-alpha*I):
          alpha = -log(1 - y/N) / I
        """
        intensity = rgb_images.float().clamp(min=1e-6)
        if num_frames == 1:
            p_obs = spad_measurements.float().clamp(1e-6, 1.0 - 1e-6)
        else:
            p_obs = (spad_measurements.float() / num_frames).clamp(1e-6, 1.0 - 1e-6)

        alpha_per_pixel = -torch.log(1.0 - p_obs) / intensity
        return float(alpha_per_pixel.mean().item())


class SPADMeasurementConsistency(nn.Module):
    """Wrapper for computing measurement consistency between decoded latents and SPAD."""

    def __init__(self, alpha: float = 1.0, num_frames: int = 1):
        super().__init__()
        self.spad_model = SPADForwardModel(alpha=alpha, num_frames=num_frames)

    def forward(
        self,
        decoded_image: torch.Tensor,
        spad_measurement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            decoded_image: VAE-decoded image [B, 3, H, W] in [0, 1] range.
            spad_measurement: SPAD observation [B, 1, H, W] or [B, 3, H, W] in [0, 1].
        """
        intensity = decoded_image.mean(dim=1, keepdim=True)
        if spad_measurement.shape[1] == 3:
            spad_measurement = spad_measurement.mean(dim=1, keepdim=True)
        return self.spad_model.measurement_loss(intensity, spad_measurement)
