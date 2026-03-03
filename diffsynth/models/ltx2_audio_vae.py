from typing import Set, Tuple, Optional, List
from enum import Enum
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .ltx2_common import VideoLatentShape, AudioLatentShape, Patchifier, NormType, build_normalization_layer


class AudioProcessor(nn.Module):
    """Converts audio waveforms to log-mel spectrograms with optional resampling."""

    def __init__(
        self,
        sample_rate: int = 16000,
        mel_bins: int = 64,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample waveform to target sample rate if needed."""
        if source_rate == target_rate:
            return waveform
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(
        self,
        waveform: torch.Tensor,
        waveform_sample_rate: int,
    ) -> torch.Tensor:
        """Convert waveform to log-mel spectrogram [batch, channels, time, n_mels]."""
        waveform = self.resample_waveform(waveform, waveform_sample_rate, self.sample_rate)

        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class AudioPatchifier(Patchifier):
    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        """
        Patchifier tailored for spectrogram/audio latents.
        Args:
            patch_size: Number of mel bins combined into a single patch. This
                controls the resolution along the frequency axis.
            sample_rate: Original waveform sampling rate. Used to map latent
                indices back to seconds so downstream consumers can align audio
                and video cues.
            hop_length: Window hop length used for the spectrogram. Determines
                how many real-time samples separate two consecutive latent frames.
            audio_latent_downsample_factor: Ratio between spectrogram frames and
                latent frames; compensates for additional downsampling inside the
                VAE encoder.
            is_causal: When True, timing is shifted to account for causal
                receptive fields so timestamps do not peek into the future.
            shift: Integer offset applied to the latent indices. Enables
                constructing overlapping windows from the same latent sequence.
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Converts latent indices into real-time seconds while honoring causal
        offsets and the configured hop length.
        Args:
            start_latent: Inclusive start index inside the latent sequence. This
                sets the first timestamp returned.
            end_latent: Exclusive end index. Determines how many timestamps get
                generated.
            dtype: Floating-point dtype used for the returned tensor, allowing
                callers to control precision.
            device: Target device for the timestamp tensor. When omitted the
                computation occurs on CPU to avoid surprising GPU allocations.
        """
        if device is None:
            device = torch.device("cpu")

        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)

        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor

        if self.is_causal:
            # Frame offset for causal alignment.
            # The "+1" ensures the timestamp corresponds to the first sample that is fully available.
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.audio_latent_downsample_factor).clip(min=0)

        return audio_mel_frame * self.hop_length / self.sample_rate

    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Builds a `(B, 1, T, 2)` tensor containing timestamps for each latent frame.
        This helper method underpins `get_patch_grid_bounds` for the audio patchifier.
        Args:
            batch_size: Number of sequences to broadcast the timings over.
            num_steps: Number of latent frames (time steps) to convert into timestamps.
            device: Device on which the resulting tensor should reside.
        """
        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cpu")

        start_timings = self._get_audio_latent_time_in_sec(
            self.shift,
            num_steps + self.shift,
            torch.float32,
            resolved_device,
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1,
            num_steps + self.shift + 1,
            torch.float32,
            resolved_device,
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)

        return torch.stack([start_timings, end_timings], dim=-1)

    def patchify(
        self,
        audio_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flattens the audio latent tensor along time. Use `get_patch_grid_bounds`
        to derive timestamps for each latent frame based on the configured hop
        length and downsampling.
        Args:
            audio_latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor. Use `get_patch_grid_bounds` to compute the
            corresponding timing metadata when needed.
        """
        audio_latents = einops.rearrange(
            audio_latents,
            "b c t f -> b t (c f)",
        )

        return audio_latents

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Restores the `(B, C, T, F)` spectrogram tensor from flattened patches.
        Use `get_patch_grid_bounds` to recompute the timestamps that describe each
        frame's position in real time.
        Args:
            audio_latents: Latent tensor to unpatchify.
            output_shape: Shape of the unpatched output tensor.
        Returns:
            Unpatched latent tensor. Use `get_patch_grid_bounds` to compute the timing
            metadata associated with the restored latents.
        """
        # audio_latents shape: (batch, time, freq * channels)
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

        return audio_latents

    def unpatchify_audio(
        self,
        audio_latents: torch.Tensor,
        channels: int,
        mel_bins: int
    ) -> torch.Tensor:
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=channels,
            f=mel_bins,
        )
        return audio_latents

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return the temporal bounds `[inclusive start, exclusive end)` for every
        patch emitted by `patchify`. For audio this corresponds to timestamps in
        seconds aligned with the original spectrogram grid.
        The returned tensor has shape `[batch_size, 1, time_steps, 2]`, where:
            - axis 1 (size 1) represents the temporal dimension
            - axis 3 (size 2) stores the `[start, end)` timestamps per patch
        Args:
            output_shape: Audio grid specification describing the number of time steps.
            device: Target device for the returned tensor.
        """
        if not isinstance(output_shape, AudioLatentShape):
            raise ValueError("AudioPatchifier expects AudioLatentShape when computing coordinates")

        return self._compute_audio_timings(output_shape.batch, output_shape.frames, device)


class AttentionType(Enum):
    """Enum for specifying the attention mechanism type."""

    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class AttnBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w).contiguous()  # b,c,hw
        w_ = torch.bmm(q, k).contiguous()  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_).contiguous()  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w).contiguous()

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> torch.nn.Module:
    match attn_type:
        case AttentionType.VANILLA:
            return AttnBlock(in_channels, norm_type=norm_type)
        case AttentionType.NONE:
            return torch.nn.Identity()
        case AttentionType.LINEAR:
            raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")


class CausalityAxis(Enum):
    """Enum for specifying the causality axis in causal convolutions."""

    NONE = None
    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_COMPATIBILITY = "width-compatibility"


class CausalConv2d(torch.nn.Module):
    """
    A causal 2D convolution.
    This layer ensures that the output at time `t` only depends on inputs
    at time `t` and earlier. It achieves this by applying asymmetric padding
    to the time dimension (width) before the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis

        # Ensure kernel_size and dilation are tuples
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        dilation = torch.nn.modules.utils._pair(dilation)

        # Calculate padding dimensions
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # The padding tuple for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        match self.causality_axis:
            case CausalityAxis.NONE:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.WIDTH | CausalityAxis.WIDTH_COMPATIBILITY:
                self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.HEIGHT:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
            case _:
                raise ValueError(f"Invalid causality_axis: {causality_axis}")

        # The internal convolution layer uses no padding, as we handle it manually
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal padding before convolution
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: tuple[int, int, int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> torch.nn.Module:
    """
    Create a 2D convolution layer that can be either causal or non-causal.
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Convolution stride
        padding: Padding (if None, will be calculated based on causal flag)
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        causality_axis: Dimension along which to apply causality.
    Returns:
        Either a regular Conv2d or CausalConv2d layer
    """
    if causality_axis is not None:
        # For causal convolution, padding is handled internally by CausalConv2d
        return CausalConv2d(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis)
    else:
        # For non-causal convolution, use symmetric padding if not specified
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        return torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )



LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding="same",
                ),
            ]
        )

        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int] = (1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = torch.nn.SiLU()
        self.conv1 = make_conv2d(in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv2d(out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


class Downsample(torch.nn.Module):
    """
    A downsampling layer that can use either a strided convolution
    or average pooling. Supports standard and causal padding for the
    convolutional mode.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Padding tuple is in the order: (left, right, top, bottom).
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pad = (0, 1, 0, 1)
                case CausalityAxis.WIDTH:
                    pad = (2, 0, 0, 1)
                case CausalityAxis.HEIGHT:
                    pad = (0, 1, 2, 0)
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pad = (1, 0, 0, 1)
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # This branch is only taken if with_conv=False, which implies causality_axis is NONE.
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x


def build_downsampling_path(  # noqa: PLR0913
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
) -> tuple[torch.nn.ModuleList, int]:
    """Build the downsampling path with residual blocks, attention, and downsampling layers."""
    down_modules = torch.nn.ModuleList()
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch

    for i_level in range(num_resolutions):
        block = torch.nn.ModuleList()
        attn = torch.nn.ModuleList()
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]

        for _ in range(num_res_blocks):
            block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        down = torch.nn.Module()
        down.block = block
        down.attn = attn
        if i_level != num_resolutions - 1:
            down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res = curr_res // 2
        down_modules.append(down)

    return down_modules, block_in


class Upsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element in the causal axis to undo encoder's padding, while keeping the length 1 + 2 * n.
            # For example, if the input is [0, 1, 2], after interpolation, the output is [0, 0, 1, 1, 2, 2].
            # The causal convolution will pad the first element as [-, -, 0, 0, 1, 1, 2, 2],
            # So the output elements rely on the following windows:
            # 0: [-,-,0]
            # 1: [-,0,0]
            # 2: [0,0,1]
            # 3: [0,1,1]
            # 4: [1,1,2]
            # 5: [1,2,2]
            # Notice that the first and second elements in the output rely only on the first element in the input,
            # while all other elements rely on two elements in the input.
            # So we can drop the first element to undo the padding (rather than the last element).
            # This is a no-op for non-causal convolutions.
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pass  # x remains unchanged
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pass  # x remains unchanged
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


def build_upsampling_path(  # noqa: PLR0913
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> tuple[torch.nn.ModuleList, int]:
    """Build the upsampling path with residual blocks, attention, and upsampling layers."""
    up_modules = torch.nn.ModuleList()
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = torch.nn.Module()
        stage.block = torch.nn.ModuleList()
        stage.attn = torch.nn.ModuleList()
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks + 1):
            stage.block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage.attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        if level != 0:
            stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res *= 2

        up_modules.insert(0, stage)

    return up_modules, block_in


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statics is computed over the entire dataset and stored in model's checkpoint under AudioVAE state_dict.
    """

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)


LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> torch.nn.Module:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = torch.nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid.attn_1 = make_attn(channels, attn_type=attn_type, norm_type=norm_type) if add_attention else torch.nn.Identity()
    mid.block_2 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


class LTX2AudioEncoder(torch.nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    The encoder uses a series of downsampling blocks with residual connections,
    attention mechanisms, and configurable causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = set(),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 2,
        resolution: int = 256,
        z_channels: int = 8,
        double_z: bool = True,
        attn_type: AttentionType = AttentionType.VANILLA,
        mid_block_add_attention: bool = False,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
        **_ignore_kwargs,
    ) -> None:
        """
        Initialize the Encoder.
        Args:
            Arguments are configuration parameters, loaded from the audio VAE checkpoint config
            (audio_vae.model.params.ddconfig):
            ch: Base number of feature channels used in the first convolution layer.
            ch_mult: Multiplicative factors for the number of channels at each resolution level.
            num_res_blocks: Number of residual blocks to use at each resolution level.
            attn_resolutions: Spatial resolutions (e.g., in time/frequency) at which to apply attention.
            resolution: Input spatial resolution of the spectrogram (height, width).
            z_channels: Number of channels in the latent representation.
            norm_type: Normalization layer type to use within the network (e.g., group, batch).
            causality_axis: Axis along which convolutions should be causal (e.g., time axis).
            sample_rate: Audio sample rate in Hz for the input signals.
            mel_hop_length: Hop length used when computing the mel spectrogram.
            n_fft: FFT size used to compute the spectrogram.
            mel_bins: Number of mel-frequency bins in the input spectrogram.
            in_channels: Number of channels in the input spectrogram tensor.
            double_z: If True, predict both mean and log-variance (doubling latent channels).
            is_causal: If True, use causal convolutions suitable for streaming setups.
            dropout: Dropout probability used in residual and mid blocks.
            attn_type: Type of attention mechanism to use in attention blocks.
            resamp_with_conv: If True, perform resolution changes using strided convolutions.
            mid_block_add_attention: If True, add an attention block in the mid-level of the encoder.
        """
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        # downsampling
        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.non_linearity = torch.nn.SiLU()

        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram into latent representations.
        Args:
            spectrogram: Input spectrogram of shape (batch, channels, time, frequency)
        Returns:
            Encoded latent representation of shape (batch, channels, frames, mel_bins)
        """
        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)

        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != self.num_resolutions - 1:
                h = stage.downsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        """
        Normalize encoder latents using per-channel statistics.
        When the encoder is configured with ``double_z=True``, the final
        convolution produces twice the number of latent channels, typically
        interpreted as two concatenated tensors along the channel dimension
        (e.g., mean and variance or other auxiliary parameters).
        This method intentionally uses only the first half of the channels
        (the "mean" component) as input to the patchifier and normalization
        logic. The remaining channels are left unchanged by this method and
        are expected to be consumed elsewhere in the VAE pipeline.
        If ``double_z=False``, the encoder output already contains only the
        mean latents and the chunking operation simply returns that tensor.
        """
        means = torch.chunk(latent_output, 2, dim=1)[0]
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        latent_patched = self.patchifier.patchify(means)
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        return self.patchifier.unpatchify(latent_normalized, latent_shape)


class LTX2AudioDecoder(torch.nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int = 128,
        out_ch: int = 2,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = set(),
        resolution: int=256,
        z_channels: int=8,
        norm_type: NormType = NormType.PIXEL,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = 64,
    ) -> None:
        """
        Initialize the Decoder.
        Args:
            Arguments are configuration parameters, loaded from the audio VAE checkpoint config
            (audio_vae.model.params.ddconfig):
            - ch, out_ch, ch_mult, num_res_blocks, attn_resolutions
            - resolution, z_channels
            - norm_type, causality_axis
        """
        super().__init__()

        # Internal behavioural defaults that are not driven by the checkpoint.
        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = torch.nn.SiLU()
        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )
        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (batch, channels, frames, mel_bins)
        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(self, sample: torch.Tensor) -> tuple[torch.Tensor, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Adjust output shape to match target dimensions for variable-length audio.
        This function handles the common case where decoded audio spectrograms need to be
        resized to match a specific target shape.
        Args:
            decoded_output: Tensor of shape (batch, channels, time, frequency)
            target_shape: AudioLatentShape describing (batch, channels, time, mel bins)
        Returns:
            Tensor adjusted to match target_shape exactly
        """
        # Current output shape: (batch, channels, time, frequency)
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        # Step 1: Crop first to avoid exceeding target dimensions
        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        # Step 2: Calculate padding needed for time and frequency dimensions
        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        # Step 3: Apply padding if needed
        if time_padding_needed > 0 or freq_padding_needed > 0:
            # PyTorch padding format: (pad_left, pad_right, pad_top, pad_bottom)
            # For audio: pad_left/right = frequency, pad_top/bottom = time
            padding = (
                0,
                max(freq_padding_needed, 0),  # frequency padding (left, right)
                0,
                max(time_padding_needed, 0),  # time padding (top, bottom)
            )
            decoded_output = F.pad(decoded_output, padding)

        # Step 4: Final safety crop to ensure exact target shape
        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def _run_upsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != 0 and hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.non_linearity(h)
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h


class LTX2Vocoder(torch.nn.Module):
    """
    Vocoder model for synthesizing audio from Mel spectrograms.
    Args:
        resblock_kernel_sizes: List of kernel sizes for the residual blocks.
                               This value is read from the checkpoint at `config.vocoder.resblock_kernel_sizes`.
        upsample_rates: List of upsampling rates.
                               This value is read from the checkpoint at `config.vocoder.upsample_rates`.
        upsample_kernel_sizes: List of kernel sizes for the upsampling layers.
                               This value is read from the checkpoint at `config.vocoder.upsample_kernel_sizes`.
        resblock_dilation_sizes: List of dilation sizes for the residual blocks.
                               This value is read from the checkpoint at `config.vocoder.resblock_dilation_sizes`.
        upsample_initial_channel: Initial number of channels for the upsampling layers.
                               This value is read from the checkpoint at `config.vocoder.upsample_initial_channel`.
        stereo: Whether to use stereo output.
                               This value is read from the checkpoint at `config.vocoder.stereo`.
        resblock: Type of residual block to use.
                                This value is read from the checkpoint at `config.vocoder.resblock`.
        output_sample_rate: Waveform sample rate.
                               This value is read from the checkpoint at `config.vocoder.output_sample_rate`.
    """

    def __init__(
        self,
        resblock_kernel_sizes: List[int] | None = [3, 7, 11],
        upsample_rates: List[int] | None = [6, 5, 2, 2, 2],
        upsample_kernel_sizes: List[int] | None = [16, 15, 8, 4, 4],
        resblock_dilation_sizes: List[List[int]] | None = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel: int = 1024,
        stereo: bool = True,
        resblock: str = "1",
        output_sample_rate: int = 24000,
    ):
        super().__init__()

        # Initialize default values if not provided. Note that mutable default values are not supported.
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sample_rate = output_sample_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        in_channels = 128 if stereo else 64
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i, _ in enumerate(self.ups):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_class(ch, kernel_size, dilations))

        out_channels = 2 if stereo else 1
        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = nn.Conv1d(final_channels, out_channels, 7, 1, padding=3)

        self.upsample_factor = math.prod(layer.stride[0] for layer in self.ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vocoder.
        Args:
            x: Input Mel spectrogram tensor. Can be either:
               - 3D: (batch_size, time, mel_bins) for mono
               - 4D: (batch_size, 2, time, mel_bins) for stereo
        Returns:
            Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        x = x.transpose(2, 3)  # (batch, channels, time, mel_bins) -> (batch, channels, mel_bins, time)

        if x.dim() == 4:  # stereo
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels

            # Evaluate all resblocks with the same input tensor so they can run
            # independently (and thus in parallel on accelerator hardware) before
            # aggregating their outputs via mean.
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )

            x = block_outputs.mean(dim=0)

        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


def decode_audio(latent: torch.Tensor, audio_decoder: "LTX2AudioDecoder", vocoder: "LTX2Vocoder") -> torch.Tensor:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    Args:
        latent: Input audio latent tensor.
        audio_decoder: Model to decode the latent to waveform features.
        vocoder: Model to convert decoded features to audio waveform.
    Returns:
        Decoded audio as a float tensor.
    """
    decoded_audio = audio_decoder(latent)
    decoded_audio = vocoder(decoded_audio).squeeze(0).float()
    return decoded_audio
