"""
Phase 3a: Differentiable SPAD Forward Model

Implements the Bernoulli measurement model for single-photon avalanche diode sensors.

Physical model:
  - Scene irradiance I(x) at pixel x (linear intensity)
  - Expected photon count: H(x) = softplus(alpha * I(x) + beta), guaranteeing H > 0
  - Detection probability: p(x) = 1 - exp(-H(x))
  - Single binary frame: b(x) ~ Bernoulli(p(x))
  - N accumulated frames: y(x) = sum_{i=1}^{N} b_i(x) ~ Binomial(N, p(x))

NLL in terms of H (numerically stable):
  For the single-frame case (N=1), b in {0, 1}:
    NLL = (1 - b) * H  - b * log(1 - exp(-H))
  where log(1 - exp(-H)) is computed via log(-expm1(-H)) for stability.

  For the multi-frame case:
    NLL = (N - y) * H  - y * log(1 - exp(-H))  (up to binomial coefficient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] values to linear intensity via inverse gamma."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _log1mexp(H: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(-H)) stably using expm1.

    For small H, 1 - exp(-H) ≈ H, so naive log(1 - exp(-H)) loses precision.
    Using log(-expm1(-H)) avoids catastrophic cancellation.
    """
    return torch.log(-torch.expm1(-H))


class SPADForwardModel(nn.Module):
    """Differentiable SPAD measurement model with Bernoulli likelihood."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0,
                 num_frames: int = 1, H_min: float = 1e-6):
        """
        Args:
            alpha: Sensor sensitivity parameter.
            beta: Offset in the exposure mapping (ensures H > 0 even at zero intensity).
            num_frames: Number of accumulated binary frames (N).
            H_min: Floor on H for numerical stability.
        """
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.num_frames = num_frames
        self.H_min = H_min

    def intensity_to_exposure(self, intensity: torch.Tensor) -> torch.Tensor:
        """Map linear intensity to expected photon count H via softplus.

        H = softplus(alpha * I + beta), clamped to >= H_min.
        This guarantees H > 0 for all inputs.
        """
        intensity = intensity.clamp(min=0.0)
        H = F.softplus(self.alpha * intensity + self.beta)
        return H.clamp(min=self.H_min)

    def intensity_to_detection_prob(self, intensity: torch.Tensor) -> torch.Tensor:
        """Convert linear intensity to SPAD detection probability.

        p = 1 - exp(-H) where H = softplus(alpha * I + beta).
        """
        H = self.intensity_to_exposure(intensity)
        return 1.0 - torch.exp(-H)

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

        Uses the H-based formulation for numerical stability:
          log(p)   = log(1 - exp(-H)) = log(-expm1(-H))   [stable via _log1mexp]
          log(1-p) = -H                                     [trivially stable]

        Args:
            intensity: Predicted linear scene irradiance [B, C, H, W] in [0, inf).
            measurement: Observed SPAD output [B, C, H, W].
                For N=1: binary {0, 1}.
                For N>1: counts in [0, N], normalized to [0, 1] by dividing by N.

        Returns:
            Log-likelihood per sample [B].
        """
        H = self.intensity_to_exposure(intensity)
        log_p = _log1mexp(H)       # log(1 - exp(-H)), stable
        log_1mp = -H               # log(exp(-H)) = -H, trivially stable

        if self.num_frames == 1:
            ll = measurement * log_p + (1.0 - measurement) * log_1mp
        else:
            y = measurement * self.num_frames
            n_minus_y = self.num_frames - y
            ll = y * log_p + n_minus_y * log_1mp

        return ll.flatten(1).sum(1)

    def negative_log_likelihood(
        self, intensity: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        """NLL loss: -log p(y | I), averaged over batch."""
        return -self.log_likelihood(intensity, measurement).mean()

    def measurement_loss(
        self, intensity: torch.Tensor, measurement: torch.Tensor,
        use_l2: bool = False,
    ) -> torch.Tensor:
        """Measurement consistency loss for DPS guidance.

        Returns pure NLL by default. Optionally adds an L2 term.
        """
        nll = self.negative_log_likelihood(intensity, measurement)
        if use_l2:
            predicted = self.forward(intensity) / max(self.num_frames, 1)
            l2 = F.mse_loss(predicted, measurement)
            return nll + l2
        return nll

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

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, num_frames: int = 1):
        super().__init__()
        self.spad_model = SPADForwardModel(alpha=alpha, beta=beta, num_frames=num_frames)

    def forward(
        self,
        decoded_image: torch.Tensor,
        spad_measurement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            decoded_image: VAE-decoded image [B, 3, H, W] in [0, 1] sRGB range.
            spad_measurement: SPAD observation [B, 1, H, W] or [B, 3, H, W] in [0, 1].
        """
        # Convert sRGB to linear intensity before applying physics model
        linear = srgb_to_linear(decoded_image.clamp(0.0, 1.0))
        intensity = linear.mean(dim=1, keepdim=True)
        if spad_measurement.shape[1] == 3:
            spad_measurement = spad_measurement.mean(dim=1, keepdim=True)
        return self.spad_model.measurement_loss(intensity, spad_measurement)
