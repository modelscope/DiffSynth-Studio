import itertools
import math
import einops
from dataclasses import replace, dataclass
from typing import Any, Callable, Iterator, List, NamedTuple, Tuple, Union, Optional
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from enum import Enum
from .ltx2_common import PixelNorm, SpatioTemporalScaleFactors, VideoLatentShape, Patchifier, AudioLatentShape
from .ltx2_dit import PixArtAlphaCombinedTimestepSizeEmbeddings

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8


class VideoLatentPatchifier(Patchifier):
    def __init__(self, patch_size: int):
        # Patch sizes for video latents.
        self._patch_size = (
            1,  # temporal dimension
            patch_size,  # height dimension
            patch_size,  # width dimension
        )

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = einops.rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )

        return latents

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: VideoLatentShape,
    ) -> torch.Tensor:
        assert self._patch_size[0] == 1, "Temporal patch size must be 1 for symmetric patchifier"

        patch_grid_frames = output_shape.frames // self._patch_size[0]
        patch_grid_height = output_shape.height // self._patch_size[1]
        patch_grid_width = output_shape.width // self._patch_size[2]

        latents = einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )

        return latents

    def unpatchify_video(
        self,
        latents: torch.Tensor,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        latents = einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=frames,
            h=height // self._patch_size[1],
            w=width // self._patch_size[2],
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the per-dimension bounds [inclusive start, exclusive end) for every
        patch produced by `patchify`. The bounds are expressed in the original
        video grid coordinates: frame/time, height, and width.
        The resulting tensor is shaped `[batch_size, 3, num_patches, 2]`, where:
            - axis 1 (size 3) enumerates (frame/time, height, width) dimensions
            - axis 3 (size 2) stores `[start, end)` indices within each dimension
        Args:
            output_shape: Video grid description containing frames, height, and width.
            device: Device of the latent tensor.
        """
        if not isinstance(output_shape, VideoLatentShape):
            raise ValueError("VideoLatentPatchifier expects VideoLatentShape when computing coordinates")

        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch

        # Validate inputs to ensure positive dimensions
        assert frames > 0, f"frames must be positive, got {frames}"
        assert height > 0, f"height must be positive, got {height}"
        assert width > 0, f"width must be positive, got {width}"
        assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

        # Generate grid coordinates for each dimension (frame, height, width)
        # We use torch.arange to create the starting coordinates for each patch.
        # indexing='ij' ensures the dimensions are in the order (frame, height, width).
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )

        # Stack the grid coordinates to create the start coordinates tensor.
        # Shape becomes (3, grid_f, grid_h, grid_w)
        patch_starts = torch.stack(grid_coords, dim=0)

        # Create a tensor containing the size of a single patch:
        # (frame_patch_size, height_patch_size, width_patch_size).
        # Reshape to (3, 1, 1, 1) to enable broadcasting when adding to the start coordinates.
        patch_size_delta = torch.tensor(
            self._patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)

        # Calculate end coordinates: start + patch_size
        # Shape becomes (3, grid_f, grid_h, grid_w)
        patch_ends = patch_starts + patch_size_delta

        # Stack start and end coordinates together along the last dimension
        # Shape becomes (3, grid_f, grid_h, grid_w, 2), where the last dimension is [start, end]
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)

        # Broadcast to batch size and flatten all spatial/temporal dimensions into one sequence.
        # Final Shape: (batch_size, 3, num_patches, 2)
        latent_coords = einops.repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=batch_size,
            bounds=2,
        )

        return latent_coords


class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class DualConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super(DualConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Ensure kernel_size, stride, padding, and dilation are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError("kernel_size must be greater than 1. Use make_linear_nd instead.")
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Set parameters for convolutions
        self.groups = groups
        self.bias = bias

        # Define the size of the channels after the first convolution
        intermediate_channels = out_channels if in_channels < out_channels else in_channels

        # Define parameters for the first convolution
        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            ))
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Define parameters for the second convolution
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, intermediate_channels // groups, kernel_size[0], 1, 1))
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight1, a=torch.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=torch.sqrt(5))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / torch.sqrt(fan_in1)
            nn.init.uniform_(self.bias1, -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / torch.sqrt(fan_in2)
            nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(
        self,
        x: torch.Tensor,
        use_conv3d: bool = False,
        skip_time_conv: bool = False,
    ) -> torch.Tensor:
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        # First convolution
        x = F.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        if skip_time_conv:
            return x

        # Second convolution
        x = F.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )

        return x

    def forward_with_2d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        b, _, _, h, w = x.shape

        # First 2D convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        # Squeeze the depth dimension out of weight1 since it's 1
        weight1 = self.weight1.squeeze(2)
        # Select stride, padding, and dilation for the 2D convolution
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x,
            weight1,
            self.bias1,
            stride1,
            padding1,
            dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second convolution which is essentially treated as a 1D convolution across the 'd' dimension
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)

        # Reshape weight2 to match the expected dimensions for conv1d
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        # Use only the relevant dimension for stride, padding, and dilation for the 1D convolution
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x,
            weight2,
            self.bias2,
            stride2,
            padding2,
            dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.weight2


class CausalConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode.value,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight


def make_conv_nd(  # noqa: PLR0913
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    temporal_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    if dims == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    elif dims in (3, (2, 1)):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange spatial dimensions into channels. Divides image into patch_size x patch_size blocks
    and moves pixels from each block into separate channels (space-to-depth).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, divides HxW into 4x4 blocks.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal patching).
    For 5D: (B, C, F, H, W) -> (B, Cx(patch_size_hw^2)x(patch_size_t), F/patch_size_t, H/patch_size_hw, W/patch_size_hw)
    Example: (B, 3, 33, 512, 512) with patch_size_hw=4, patch_size_t=1 -> (B, 48, 33, 128, 128)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange channels back into spatial dimensions. Inverse of patchify - moves pixels from
    channels back into patch_size x patch_size blocks (depth-to-space).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, expands HxW by 4x.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal expansion).
    For 5D: (B, Cx(patch_size_hw^2)x(patch_size_t), F, H, W) -> (B, C, Fxpatch_size_t, Hxpatch_size_hw, Wxpatch_size_hw)
    Example: (B, 48, 33, 128, 128) with patch_size_hw=4, patch_size_t=1 -> (B, 3, 33, 512, 512)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statics is computed over the entire dataset and stored in model's checkpoint under VAE state_dict.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-stds", torch.empty(latent_channels))
        self.register_buffer("mean-of-stds_over_std-of-means", torch.empty(latent_channels))
        self.register_buffer("channel", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1).to(x)


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.
    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        if norm_layer == NormLayerType.GROUP_NORM:
            self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.norm2 = PixelNorm()

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))

        self.conv_shortcut = (make_linear_nd(dims=dims, in_channels=in_channels, out_channels=out_channels)
                              if in_channels != out_channels else nn.Identity())

        # Using GroupNorm with 1 group is equivalent to LayerNorm but works with (B, C, ...) layout
        # avoiding the need for dimension rearrangement used in standard nn.LayerNorm
        self.norm3 = (nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps, affine=True)
                      if in_channels != out_channels else nn.Identity())

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.zeros(4, in_channels))

    def _feed_spatial_noise(
        self,
        hidden_states: torch.Tensor,
        per_channel_scale: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # similar to the "explicit noise inputs" method in style-gan
        spatial_noise = torch.randn(spatial_shape, device=device, dtype=dtype, generator=generator)[None]
        scaled_noise = (spatial_noise * per_channel_scale)[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise

        return hidden_states

    def forward(
        self,
        input_tensor: torch.Tensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]

        hidden_states = self.norm1(hidden_states)
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            ada_values = self.scale_shift_table[None, ..., None, None, None].to(
                device=hidden_states.device, dtype=hidden_states.dtype) + timestep.reshape(
                    batch_size,
                    4,
                    -1,
                    timestep.shape[-3],
                    timestep.shape[-2],
                    timestep.shape[-1],
                )
            shift1, scale1, shift2, scale2 = ada_values.unbind(dim=1)

            hidden_states = hidden_states * (1 + scale1) + shift1

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale1.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        hidden_states = self.norm2(hidden_states)

        if self.timestep_conditioning:
            hidden_states = hidden_states * (1 + scale2) + shift2

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(
                hidden_states,
                self.per_channel_scale2.to(device=hidden_states.device, dtype=hidden_states.dtype),
                generator=generator,
            )

        input_tensor = self.norm3(input_tensor)

        batch_size = input_tensor.shape[0]

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.
    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        inject_noise (`bool`, *optional*, defaults to `False`):
            Whether to inject noise into the hidden states.
        timestep_conditioning (`bool`, *optional*, defaults to `False`):
            Whether to condition the hidden states on the timestep.
    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: NormLayerType = NormLayerType.GROUP_NORM,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.timestep_conditioning = timestep_conditioning

        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=in_channels * 4,
                                                                           size_emb_dim=0)

        self.res_blocks = nn.ModuleList([
            ResnetBlock3D(
                dims=dims,
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal: bool = True,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        timestep_embed = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            batch_size = hidden_states.shape[0]
            timestep_embed = self.time_embedder(
                timestep=timestep.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

        for resnet in self.res_blocks:
            hidden_states = resnet(
                hidden_states,
                causal=causal,
                timestep=timestep_embed,
                generator=generator,
            )

        return hidden_states


class SpaceToDepthDownsample(nn.Module):

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)  # duplicate first frames for padding

        # skip connection
        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        # conv
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):

    def __init__(
        self,
        dims: int | Tuple[int, int],
        in_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.residual:
            # Reshape and duplicate the input to match the output shape
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.
    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.
    Returns:
        A 1D tensor of shape `(length,)` with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.
    Args:
        tile_size_in_pixels (int): Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels (int, optional): Overlap between tiles in pixels. Must be divisible by 32. Defaults to 0.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap.
    Args:
        tile_size_in_frames (int): Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames (int, optional): Number of overlapping frames between consecutive tiles.
            Must be divisible by 8. Defaults to 0.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.
    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    """Intervals which a single dimension of the latent space is split into.
    Each interval is defined by its start, end, left ramp, and right ramp.
    The start and end are the indices of the first and last element (exclusive) in the interval.
    Ramps are regions of the interval where the value of the mask tensor is
    interpolated between 0 and 1 for blending with neighboring intervals.
    The left ramp and right ramp values are the lengths of the left and right ramps.
    """

    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


@dataclass(frozen=True)
class LatentIntervals:
    """Intervals which the latent tensor of given shape is split into.
    Each dimension of the latent space is split into intervals based on the length along said dimension.
    """

    original_shape: torch.Size
    dimension_intervals: Tuple[DimensionIntervals, ...]


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(_intervals: DimensionIntervals,) -> tuple[list[slice], list[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


class Tile(NamedTuple):
    """
    Represents a single tile.
    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.
        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.
        masks_1d:
            Per-dimension masks in OUTPUT units.
            These are used to create all-dimensional blending mask.
    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: Tuple[slice, ...]
    out_coords: Tuple[slice, ...]
    masks_1d: Tuple[Tuple[torch.Tensor, ...]]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: List[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                # Broadcast mask along this dimension (length 1).
                one = torch.ones(1)

                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: List[MappingOperation],
) -> List[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        starts = dimension_intervals.starts
        ends = dimension_intervals.ends
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True):
        tiles.append(Tile(
            in_coords=in_coord,
            out_coords=out_coord,
            masks_1d=mask_1d,
        ))
    return tiles


def create_tiles(
    latent_shape: torch.Size,
    splitters: List[SplitOperation],
    mappers: List[MappingOperation],
) -> List[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(f"Number of splitters must be equal to number of dimensions in latent shape, "
                         f"got {len(splitters)} and {len(latent_shape)}")
    if len(mappers) != len(latent_shape):
        raise ValueError(f"Number of mappers must be equal to number of dimensions in latent shape, "
                         f"got {len(mappers)} and {len(latent_shape)}")
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class LTX2VideoEncoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        encoder_blocks = [['res_x', {
            'num_layers': 4
        }], ['compress_space_res', {
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 6
        }], ['compress_time_res', {
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 6
        }], ['compress_all_res', {
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 2
        }], ['compress_all_res', {
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 2
        }]]
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError("Invalid number of frames: Encode input must have 1 + 8 * x frames "
                             "(e.g., 1, 9, 17, ...). Please check your input.")

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                                 f"(N means + 1 logvar), got {sample.shape[1]}")

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)


    def tiled_encode_video(
        self,
        video: torch.Tensor,
        tile_size: int = 512,
        tile_overlap: int = 128,
    ) -> torch.Tensor:
        """Encode video using spatial tiling for memory efficiency.
        Splits the video into overlapping spatial tiles, encodes each tile separately,
        and blends the results using linear feathering in the overlap regions.
        Args:
            video: Input tensor of shape [B, C, F, H, W]
            tile_size: Tile size in pixels (must be divisible by 32)
            tile_overlap: Overlap between tiles in pixels (must be divisible by 32)
        Returns:
            Encoded latent tensor [B, C_latent, F_latent, H_latent, W_latent]
        """
        batch, _channels, frames, height, width = video.shape
        device = video.device
        dtype = video.dtype

        # Validate tile parameters
        if tile_size % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(f"tile_size must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_size}")
        if tile_overlap % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(f"tile_overlap must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_overlap}")
        if tile_overlap >= tile_size:
            raise ValueError(f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")

        # If video fits in a single tile, use regular encoding
        if height <= tile_size and width <= tile_size:
            return self.forward(video)

        # Calculate output dimensions
        # VAE compresses: H -> H/32, W -> W/32, F -> 1 + (F-1)/8
        output_height = height // VAE_SPATIAL_FACTOR
        output_width = width // VAE_SPATIAL_FACTOR
        output_frames = 1 + (frames - 1) // VAE_TEMPORAL_FACTOR

        # Latent channels (128 for LTX-2)
        # Get from a small test encode or assume 128
        latent_channels = 128

        # Initialize output and weight tensors
        output = torch.zeros(
            (batch, latent_channels, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )
        weights = torch.zeros(
            (batch, 1, output_frames, output_height, output_width),
            device=device,
            dtype=dtype,
        )

        # Calculate tile positions with overlap
        # Step size is tile_size - tile_overlap
        step_h = tile_size - tile_overlap
        step_w = tile_size - tile_overlap

        h_positions = list(range(0, max(1, height - tile_overlap), step_h))
        w_positions = list(range(0, max(1, width - tile_overlap), step_w))

        # Ensure last tile covers the edge
        if h_positions[-1] + tile_size < height:
            h_positions.append(height - tile_size)
        if w_positions[-1] + tile_size < width:
            w_positions.append(width - tile_size)

        # Remove duplicates and sort
        h_positions = sorted(set(h_positions))
        w_positions = sorted(set(w_positions))

        # Overlap in latent space
        overlap_out_h = tile_overlap // VAE_SPATIAL_FACTOR
        overlap_out_w = tile_overlap // VAE_SPATIAL_FACTOR

        # Process each tile
        for h_pos in h_positions:
            for w_pos in w_positions:
                # Calculate tile boundaries in input space
                h_start = max(0, h_pos)
                w_start = max(0, w_pos)
                h_end = min(h_start + tile_size, height)
                w_end = min(w_start + tile_size, width)

                # Ensure tile dimensions are divisible by VAE_SPATIAL_FACTOR
                tile_h = ((h_end - h_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR
                tile_w = ((w_end - w_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR

                if tile_h < VAE_SPATIAL_FACTOR or tile_w < VAE_SPATIAL_FACTOR:
                    continue

                # Adjust end positions
                h_end = h_start + tile_h
                w_end = w_start + tile_w

                # Extract tile
                tile = video[:, :, :, h_start:h_end, w_start:w_end]

                # Encode tile
                encoded_tile = self.forward(tile)

                # Get actual encoded dimensions
                _, _, tile_out_frames, tile_out_height, tile_out_width = encoded_tile.shape

                # Calculate output positions
                out_h_start = h_start // VAE_SPATIAL_FACTOR
                out_w_start = w_start // VAE_SPATIAL_FACTOR
                out_h_end = min(out_h_start + tile_out_height, output_height)
                out_w_end = min(out_w_start + tile_out_width, output_width)

                # Trim encoded tile if necessary
                actual_tile_h = out_h_end - out_h_start
                actual_tile_w = out_w_end - out_w_start
                encoded_tile = encoded_tile[:, :, :, :actual_tile_h, :actual_tile_w]

                # Create blending mask with linear feathering at edges
                mask = torch.ones(
                    (1, 1, tile_out_frames, actual_tile_h, actual_tile_w),
                    device=device,
                    dtype=dtype,
                )

                # Apply feathering at edges (linear blend in overlap regions)
                # Left edge
                if h_pos > 0 and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                    fade_in = torch.linspace(0.0, 1.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                    mask[:, :, :, :overlap_out_h, :] *= fade_in.view(1, 1, 1, -1, 1)

                # Right edge (bottom in height dimension)
                if h_end < height and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                    fade_out = torch.linspace(1.0, 0.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                    mask[:, :, :, -overlap_out_h:, :] *= fade_out.view(1, 1, 1, -1, 1)

                # Top edge (left in width dimension)
                if w_pos > 0 and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                    fade_in = torch.linspace(0.0, 1.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                    mask[:, :, :, :, :overlap_out_w] *= fade_in.view(1, 1, 1, 1, -1)

                # Bottom edge (right in width dimension)
                if w_end < width and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                    fade_out = torch.linspace(1.0, 0.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                    mask[:, :, :, :, -overlap_out_w:] *= fade_out.view(1, 1, 1, 1, -1)

                # Accumulate weighted results
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += encoded_tile * mask
                weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += mask

        # Normalize by weights (avoid division by zero)
        output = output / (weights + 1e-8)

        return output

    def encode(
        self,
        video: torch.Tensor,
        tiled=False,
        tile_size_in_pixels: Optional[int] = 512,
        tile_overlap_in_pixels: Optional[int] = 128,
        **kwargs,
    ) -> torch.Tensor:
        if video.ndim == 4:
            video = video.unsqueeze(0)  # [C, F, H, W] -> [B, C, F, H, W]
        # Choose encoding method based on tiling flag
        if tiled:
            latents = self.tiled_encode_video(
                video=video,
                tile_size=tile_size_in_pixels,
                tile_overlap=tile_overlap_in_pixels,
            )
        else:
            # Encode video - VAE expects [B, C, F, H, W], returns [B, C, F', H', W']
            latents = self.forward(video)
        return latents


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class LTX2VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
    ):
        super().__init__()

        # Spatiotemporal downscaling between decoded video space and VAE latents.
        # According to the LTXV paper, the standard configuration downsamples
        # video inputs by a factor of 8 in the temporal dimension and 32 in
        # each spatial dimension (height and width). This parameter determines how
        # many video frames and pixels correspond to a single latent cell.
        decoder_blocks = [['res_x', {
            'num_layers': 5,
            'inject_noise': False
        }], ['compress_all', {
            'residual': True,
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 5,
            'inject_noise': False
        }], ['compress_all', {
            'residual': True,
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 5,
            'inject_noise': False
        }], ['compress_all', {
            'residual': True,
            'multiplier': 2
        }], ['res_x', {
            'num_layers': 5,
            'inject_noise': False
        }]]
        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8,
            width=32,
            height=32,
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # Compute initial feature_channels by going through blocks in reverse
        # This determines the channel width at the start of the decoder
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=feature_channels * 2,
                                                                                size_emb_dim=0)
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (torch.randn(
                sample.size(),
                generator=generator,
                dtype=sample.dtype,
                device=sample.device,
            ) * self.decode_noise_scale)

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype) + embedded_timestep.reshape(
                    batch_size,
                    2,
                    -1,
                    embedded_timestep.shape[-3],
                    embedded_timestep.shape[-2],
                    embedded_timestep.shape[-1],
                )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is not None and tiling_config.spatial_config is not None:
            cfg = tiling_config.spatial_config
            long_side = max(latent.shape[3], latent.shape[4])

            def enable_on_axis(axis_idx: int, factor: int) -> None:
                size = cfg.tile_size_in_pixels // factor
                overlap = cfg.tile_overlap_in_pixels // factor
                axis_length = latent.shape[axis_idx]
                lower_threshold = max(2, overlap + 1)
                tile_size = max(lower_threshold, round(size * axis_length / long_side))
                splitters[axis_idx] = split_in_spatial(tile_size, overlap)
                mappers[axis_idx] = to_mapping_operation(map_spatial_slice, factor)

            enable_on_axis(3, self.video_downscale_factors.height)
            enable_on_axis(4, self.video_downscale_factors.width)

        if tiling_config is not None and tiling_config.temporal_config is not None:
            cfg = tiling_config.temporal_config
            tile_size = cfg.tile_size_in_frames // self.video_downscale_factors.time
            overlap = cfg.tile_overlap_in_frames // self.video_downscale_factors.time
            splitters[2] = split_in_temporal(tile_size, overlap)
            mappers[2] = to_mapping_operation(map_temporal_slice, self.video_downscale_factors.time)

        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices;
        """

        # Calculate full video shape from latent shape to get spatial dimensions
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
        tiles = self._prepare_tiles(latent, tiling_config)

        temporal_groups = self._group_tiles_by_temporal_slice(tiles)

        # State for temporal overlap handling
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]

            # Calculate the shape of the temporal buffer for this group of tiles.
            # The temporal length depends on whether this is the first tile (starts at 0) or not.
            # - First tile: (frames - 1) * scale + 1
            # - Subsequent tiles: frames * scale
            # This logic is handled by TemporalAxisMapping and reflected in out_coords.
            temporal_tile_buffer_shape = full_video_shape._replace(frames=curr_temporal_slice.stop -
                                                                   curr_temporal_slice.start,)

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
            )

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                # Check if current temporal slice overlaps with previous temporal slice
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    # The overlap is already masked before it reaches this step. Each tile is accumulated into buffer
                    # with its trapezoidal mask, and curr_weights accumulates the same mask. In the overlap blend we add
                    # the masked values (buffer[...]) and the corresponding weights (curr_weights[...]) into the
                    # previous buffers, then later normalize by weights.
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[:, :,
                                                                                         slice(0, overlap_len), :, :]

                    buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                    curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[:, :,
                                                                                       temporal_overlap_slice, :, :]

                # Yield the non-overlapping part of the previous chunk
                previous_weights = previous_weights.clamp(min=1e-8)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            previous_weights = previous_weights.clamp(min=1e-8)
            yield previous_chunk / previous_weights

    def _group_tiles_by_temporal_slice(self, tiles: List[Tile]) -> List[List[Tile]]:
        """Group tiles by their temporal output slice."""
        if not tiles:
            return []

        groups = []
        current_slice = tiles[0].out_coords[2]
        current_group = []

        for tile in tiles:
            tile_slice = tile.out_coords[2]
            if tile_slice == current_slice:
                current_group.append(tile)
            else:
                groups.append(current_group)
                current_slice = tile_slice
                current_group = [tile]

        # Add the final group
        if current_group:
            groups.append(current_group)

        return groups

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """
        Decode and accumulate all tiles of a temporal group into a local buffer.
        The buffer is local to the group and always starts at time 0; temporal coordinates
        are rebased by subtracting temporal_slice.start.
        """
        temporal_slice = group_tiles[0].out_coords[2]

        weights = torch.zeros_like(buffer)

        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            mask = tile.blend_mask.to(device=buffer.device, dtype=buffer.dtype)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            # Use the tile's output coordinate length, not the decoded tile's length,
            # as the decoder may produce a different number of frames than expected
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]

            # Ensure we don't exceed the buffer or decoded tile bounds
            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            # Slice decoded_tile and mask to match the actual length we're writing
            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask

            buffer[chunk_coords] += decoded_slice * mask_slice
            weights[chunk_coords] += mask_slice

        return weights

    def decode(
        self,
        latent: torch.Tensor,
        tiled=False,
        tile_size_in_pixels: Optional[int] = 512,
        tile_overlap_in_pixels: Optional[int] = 128,
        tile_size_in_frames: Optional[int] = 128,
        tile_overlap_in_frames: Optional[int] = 24,
    ) -> torch.Tensor:
        if tiled:
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tile_size_in_pixels,
                    tile_overlap_in_pixels=tile_overlap_in_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tile_size_in_frames,
                    tile_overlap_in_frames=tile_overlap_in_frames,
                ),
            )
            tiles = self.tiled_decode(latent, tiling_config)
            return torch.cat(list(tiles), dim=2)
        else:
            return self.forward(latent)

def decode_video(
    latent: torch.Tensor,
    video_decoder: LTX2VideoDecoder,
    tiling_config: TilingConfig | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """
    Decode a video latent tensor with the given decoder.
    Args:
        latent: Tensor [c, f, h, w]
        video_decoder: Decoder module.
        tiling_config: Optional tiling settings.
        generator: Optional random generator for deterministic decoding.
    Yields:
        Decoded chunk [f, h, w, c], uint8 in [0, 255].
    """

    def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
        frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames

    if tiling_config is not None:
        for frames in video_decoder.tiled_decode(latent, tiling_config, generator=generator):
            return convert_to_uint8(frames)
    else:
        decoded_video = video_decoder(latent, generator=generator)
        return convert_to_uint8(decoded_video)


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    frame_stride = cfg.tile_size_in_frames - cfg.tile_overlap_in_frames
    return (num_frames - 1 + frame_stride - 1) // frame_stride


def split_in_spatial(size: int, overlap: int) -> SplitOperation:

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_in_temporal(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_in_spatial(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:

    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
        number_of_slices = len(intervals.starts)
        for i in range(number_of_slices):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True)


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False)
