from dataclasses import dataclass
from typing import NamedTuple, Protocol, Tuple
import torch
from torch import nn
from enum import Enum


class VideoPixelShape(NamedTuple):
    """
    Shape of the tensor representing the video pixel array. Assumes BGR channel format.
    """

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    """

    time: int
    width: int
    height: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """
    Shape of the tensor representing video in VAE latent space.
    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors[0] + 1
        height = shape.height // scale_factors[1]
        width = shape.width // scale_factors[2]

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class LatentState:
    """
    State of latents during the diffusion denoising process.
    Attributes:
        latent: The current noisy latent tensor being denoised.
        denoise_mask: Mask encoding the denoising strength for each token (1 = full denoising, 0 = no denoising).
        positions: Positional indices for each latent element, used for positional embeddings.
        clean_latent: Initial state of the latent before denoising, may include conditioning latents.
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
        )


class NormType(Enum):
    """Normalization layer types: GROUP (GroupNorm) or PIXEL (per-location RMS norm)."""

    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    For each element along the chosen dimension, this layer normalizes the tensor
    by the root-mean-square of its values across that dimension:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization along the configured dimension.
        """
        # Compute mean of squared values along `dim`, keep dimensions for broadcasting.
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        # Normalize by the root-mean-square (RMS).
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int, *, num_groups: int = 32, normtype: NormType = NormType.GROUP
) -> nn.Module:
    """
    Create a normalization layer based on the normalization type.
    Args:
        in_channels: Number of input channels
        num_groups: Number of groups for group normalization
        normtype: Type of normalization: "group" or "pixel"
    Returns:
        A normalization layer
    """
    if normtype == NormType.GROUP:
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    if normtype == NormType.PIXEL:
        return PixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)



class Patchifier(Protocol):
    """
    Protocol for patchifiers that convert latent tensors into patches and assemble them back.
    """

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        ...
        """
        Convert latent tensors into flattened patch tokens.
        Args:
            latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor.
        """

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape | VideoLatentShape,
    ) -> torch.Tensor:
        """
        Converts latent tensors between spatio-temporal formats and flattened sequence representations.
        Args:
            latents: Patch tokens that must be rearranged back into the latent grid constructed by `patchify`.
            output_shape: Shape of the output tensor. Note that output_shape is either AudioLatentShape or
            VideoLatentShape.
        Returns:
            Dense latent tensor restored from the flattened representation.
        """

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        ...
        """
        Returns the patch size as a tuple of (temporal, height, width) dimensions
        """

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...
        """
        Compute metadata describing where each latent patch resides within the
        grid specified by `output_shape`.
        Args:
            output_shape: Target grid layout for the patches.
            device: Target device for the returned tensor.
        Returns:
            Tensor containing patch coordinate metadata such as spatial or temporal intervals.
        """


def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> torch.Tensor:
    """
    Map latent-space `[start, end)` coordinates to their pixel-space equivalents by scaling
    each axis (frame/time, height, width) with the corresponding VAE downsampling factors.
    Optionally compensate for causal encoding that keeps the first frame at unit temporal scale.
    Args:
        latent_coords: Tensor of latent bounds shaped `(batch, 3, num_patches, 2)`.
        scale_factors: SpatioTemporalScaleFactors tuple `(temporal, height, width)` with integer scale factors applied
        per axis.
        causal_fix: When True, rewrites the temporal axis of the first frame so causal VAEs
            that treat frame zero differently still yield non-negative timestamps.
    """
    # Broadcast the VAE scale factors so they align with the `(batch, axis, patch, bound)` layout.
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1  # axis dimension corresponds to (frame/time, height, width)
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device).view(*broadcast_shape)

    # Apply per-axis scaling to convert latent bounds into pixel-space coordinates.
    pixel_coords = latent_coords * scale_tensor

    if causal_fix:
        # VAE temporal stride for the very first frame is 1 instead of `scale_factors[0]`.
        # Shift and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)

    return pixel_coords
