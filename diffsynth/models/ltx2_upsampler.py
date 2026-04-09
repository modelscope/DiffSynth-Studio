import math
from typing import Optional, Tuple
import torch
from einops import rearrange
import torch.nn.functional as F
from .ltx2_video_vae import LTX2VideoEncoder

class PixelShuffleND(torch.nn.Module):
    """
    N-dimensional pixel shuffle operation for upsampling tensors.
    Args:
        dims (int): Number of dimensions to apply pixel shuffle to.
            - 1: Temporal (e.g., frames)
            - 2: Spatial (e.g., height and width)
            - 3: Spatiotemporal (e.g., depth, height, width)
        upscale_factors (tuple[int, int, int], optional): Upscaling factors for each dimension.
            For dims=1, only the first value is used.
            For dims=2, the first two values are used.
            For dims=3, all three values are used.
    The input tensor is rearranged so that the channel dimension is split into
    smaller channels and upscaling factors, and the upscaling factors are moved
    into the corresponding spatial/temporal dimensions.
    Note:
    This operation is equivalent to the patchifier operation in for the models. Consider
    using this class instead.
    """

    def __init__(self, dims: int, upscale_factors: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        assert dims in [1, 2, 3], "dims must be 1, 2, or 3"
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        elif self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        elif self.dims == 1:
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )
        else:
            raise ValueError(f"Unsupported dims: {self.dims}")


class ResBlock(torch.nn.Module):
    """
    Residual block with two convolutional layers, group normalization, and SiLU activation.
    Args:
        channels (int): Number of input and output channels.
        mid_channels (Optional[int]): Number of channels in the intermediate convolution layer. Defaults to `channels`
        if not specified.
        dims (int): Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d). Defaults to 3.
    """

    def __init__(self, channels: int, mid_channels: Optional[int] = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.conv1 = conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class BlurDownsample(torch.nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.
    Applies only on H,W. Works for dims=2 or dims=3 (per-frame).
    """

    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        assert dims in (2, 3)
        assert isinstance(stride, int)
        assert stride >= 1
        assert kernel_size >= 3
        assert kernel_size % 2 == 1
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        # 5x5 separable binomial kernel using binomial coefficients [1, 4, 6, 4, 1] from
        # the 4th row of Pascal's triangle. This kernel is used for anti-aliasing and
        # provides a smooth approximation of a Gaussian filter (often called a "binomial filter").
        # The 2D kernel is constructed as the outer product and normalized.
        k = torch.tensor([math.comb(kernel_size - 1, k) for k in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()  # shape (kernel_size, kernel_size)
        self.register_buffer("kernel", k2d[None, None, :, :])  # (1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            return self._apply_2d(x)
        else:
            # dims == 3: apply per-frame on H,W
            b, _, f, _, _ = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self._apply_2d(x)
            h2, w2 = x.shape[-2:]
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)
            return x

    def _apply_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        c = x2d.shape[1]
        weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)  # depthwise
        x2d = F.conv2d(x2d, weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)
        return x2d


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
    return mapping[float(scale)]


class SpatialRationalResampler(torch.nn.Module):
    """
    Fully-learned rational spatial scaling: up by 'num' via PixelShuffle, then anti-aliased
    downsample by 'den' using fixed blur + stride. Operates on H,W only.
    For dims==3, work per-frame for spatial scaling (temporal axis untouched).
    Args:
        mid_channels (`int`): Number of intermediate channels for the convolution layer
        scale (`float`): Spatial scaling factor. Supported values are:
            - 0.75: Downsample by 3/4 (reduce spatial size)
            - 1.5: Upsample by 3/2 (increase spatial size)
            - 2.0: Upsample by 2x (double spatial size)
            - 4.0: Upsample by 4x (quadruple spatial size)
            Any other value will raise a ValueError.
    """

    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)
        self.conv = torch.nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x


class LTX2LatentUpsampler(torch.nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.
    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
        spatial_scale (`float`): Scale factor for spatial upsampling
        rational_resampler (`bool`): Whether to use a rational resampler for spatial upsampling
    """
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # remove the first frame after upsampling.
                # This is done because the first frame encodes one pixel frame.
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                x = self.upsampler(x)
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


def upsample_video(latent: torch.Tensor, video_encoder: LTX2VideoEncoder, upsampler: "LTX2LatentUpsampler") -> torch.Tensor:
    """
    Apply upsampling to the latent representation using the provided upsampler,
    with normalization and un-normalization based on the video encoder's per-channel statistics.
    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LTX2LatentUpsampler module to perform upsampling.
    Returns:
        torch.Tensor: Upsampled and re-normalized latent tensor.
    """
    latent = video_encoder.per_channel_statistics.un_normalize(latent)
    latent = upsampler(latent)
    latent = video_encoder.per_channel_statistics.normalize(latent)
    return latent
