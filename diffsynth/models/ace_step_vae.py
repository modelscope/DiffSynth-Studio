# Copyright 2025 The ACESTEO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm


class Snake1d(nn.Module):
    """Snake activation: x + 1/(beta+eps) * sin(alpha*x)^2."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class OobleckResidualUnit(nn.Module):
    """Residual unit: Snake1d → Conv1d(dilated) → Snake1d → Conv1d(1×1) + skip."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output = self.conv1(self.snake1(hidden_state))
        output = self.conv2(self.snake2(output))
        padding = (hidden_state.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output


class OobleckEncoderBlock(nn.Module):
    """Encoder block: 3 residual units + downsampling conv."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = weight_norm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        return self.conv1(hidden_state)


class OobleckDecoderBlock(nn.Module):
    """Decoder block: upsampling conv + 3 residual units."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


class OobleckEncoder(nn.Module):
    """Full encoder: audio → latent representation [B, encoder_hidden_size, T'].

    conv1 → [blocks] → snake1 → conv2
    """

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        audio_channels: int = 2,
        downsampling_ratios: list = None,
        channel_multiples: list = None,
    ):
        super().__init__()
        downsampling_ratios = downsampling_ratios or [2, 4, 4, 6, 10]
        channel_multiples = channel_multiples or [1, 2, 4, 8, 16]
        channel_multiples = [1] + channel_multiples

        self.conv1 = weight_norm(nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3))

        self.block = nn.ModuleList()
        for stride_index, stride in enumerate(downsampling_ratios):
            self.block.append(
                OobleckEncoderBlock(
                    input_dim=encoder_hidden_size * channel_multiples[stride_index],
                    output_dim=encoder_hidden_size * channel_multiples[stride_index + 1],
                    stride=stride,
                )
            )

        d_model = encoder_hidden_size * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for block in self.block:
            hidden_state = block(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class OobleckDecoder(nn.Module):
    """Full decoder: latent → audio waveform [B, audio_channels, T].

    conv1 → [blocks] → snake1 → conv2(no bias)
    """

    def __init__(
        self,
        channels: int = 128,
        input_channels: int = 64,
        audio_channels: int = 2,
        upsampling_ratios: list = None,
        channel_multiples: list = None,
    ):
        super().__init__()
        upsampling_ratios = upsampling_ratios or [10, 6, 4, 4, 2]
        channel_multiples = channel_multiples or [1, 2, 4, 8, 16]
        channel_multiples = [1] + channel_multiples

        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        self.block = nn.ModuleList()
        for stride_index, stride in enumerate(upsampling_ratios):
            self.block.append(
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index],
                    output_dim=channels * channel_multiples[len(upsampling_ratios) - stride_index - 1],
                    stride=stride,
                )
            )

        self.snake1 = Snake1d(channels)
        # conv2 has no bias (matches checkpoint: only weight_g/weight_v, no bias key)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for block in self.block:
            hidden_state = block(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class OobleckDiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = torch.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "OobleckDiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return (self.mean * self.mean + self.var - self.logvar - 1.0).sum(1).mean()
            else:
                normalized_diff = torch.pow(self.mean - other.mean, 2) / other.var
                var_ratio = self.var / other.var
                logvar_diff = self.logvar - other.logvar

                kl = normalized_diff + var_ratio + logvar_diff - 1

                kl = kl.sum(1).mean()
                return kl


class AceStepVAE(nn.Module):
    """Audio VAE for ACE-Step (AutoencoderOobleck architecture).

    Encodes audio waveform → latent, decodes latent → audio waveform.
    Uses Snake1d activations and weight-normalized convolutions.
    """

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        downsampling_ratios: list = None,
        channel_multiples: list = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        audio_channels: int = 2,
        sampling_rate: int = 48000,
    ):
        super().__init__()
        downsampling_ratios = downsampling_ratios or [2, 4, 4, 6, 10]
        channel_multiples = channel_multiples or [1, 2, 4, 8, 16]
        upsampling_ratios = downsampling_ratios[::-1]

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=upsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.sampling_rate = sampling_rate

    def tiled_encode(self, x: torch.Tensor, tile_size: int = 10240, tile_stride: int = 5120) -> torch.Tensor:
        batch_size, audio_channels, T_audio = x.shape
        up = self.upsampling_factor

        lat_T = round(T_audio / up)

        tiles = []
        for t in range(0, T_audio, tile_stride):
            if t - tile_stride >= 0 and t - tile_stride + tile_size >= T_audio:
                continue
            tiles.append((t, min(t + tile_size, T_audio)))

        encoder_out_channels = self.encoder.conv2.out_channels
        values = torch.zeros((batch_size, encoder_out_channels, lat_T), dtype=x.dtype, device="cpu")
        weight = torch.zeros((1, 1, lat_T), dtype=x.dtype, device="cpu")

        for t_start, t_end in tiles:
            tile_audio = x[:, :, t_start:t_end]
            tile_latent = self.encoder(tile_audio).to("cpu")
            lat_len = tile_latent.shape[-1]

            lat_start = round(t_start / up)
            lat_end = lat_start + lat_len

            border = (tile_size - tile_stride) // up
            mask = self._build_blend_mask_1d(
                lat_len, border,
                is_left_bound=(t_start == 0),
                is_right_bound=(t_end >= T_audio),
                dtype=x.dtype, device="cpu",
            )

            values[:, :, lat_start:lat_end] += tile_latent * mask
            weight[:, :, lat_start:lat_end] += mask

        weight = weight.clamp(min=1e-8)
        latent = values / weight
        output = OobleckDiagonalGaussianDistribution(latent.to(x.device)).sample()
        return output

    def encode(self, x: torch.Tensor, tiled: bool = False, tile_size: int = 10240, tile_stride: int = 5120) -> torch.Tensor:
        """Audio waveform [B, audio_channels, T] → latent [B, decoder_input_channels, T"]."""
        if tiled:
            return self.tiled_encode(x, tile_size=tile_size, tile_stride=tile_stride)
        h = self.encoder(x)
        output = OobleckDiagonalGaussianDistribution(h).sample()
        return output

    @property
    def upsampling_factor(self) -> int:
        ratios = [10, 6, 4, 4, 2]
        result = 1
        for r in ratios:
            result *= r
        return result

    def _build_blend_mask_1d(self, length: int, border_width: int, is_left_bound: bool, is_right_bound: bool,
                              dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.ones(length, dtype=dtype, device=device)
        if border_width <= 0:
            return mask
        if not is_left_bound:
            ramp = torch.linspace(0.0, 1.0, border_width, dtype=dtype, device=device)
            mask[:border_width] = ramp
        if not is_right_bound:
            ramp = torch.linspace(1.0, 0.0, border_width, dtype=dtype, device=device)
            mask[-border_width:] = ramp
        return mask.unsqueeze(0).unsqueeze(0)

    def tiled_decode(self, z: torch.Tensor, tile_size: int = 512, tile_stride: int = 256) -> torch.Tensor:
        batch_size, channels, T = z.shape
        up = self.upsampling_factor
        out_length = up * T

        tiles = []
        for t in range(0, T, tile_stride):
            if t - tile_stride >= 0 and t - tile_stride + tile_size >= T:
                continue
            tiles.append((t, min(t + tile_size, T)))

        audio_channels = self.decoder.conv2.out_channels
        values = torch.zeros((batch_size, audio_channels, out_length), dtype=z.dtype, device="cpu")
        weight = torch.zeros((1, 1, out_length), dtype=z.dtype, device="cpu")

        for t_start, t_end in tiles:
            tile_latent = z[:, :, t_start:t_end]
            tile_output = self.decoder(tile_latent).to("cpu")
            tile_len = tile_output.shape[-1]

            out_start = t_start * up
            out_end = out_start + tile_len

            border = (tile_size - tile_stride) * up
            mask = self._build_blend_mask_1d(
                tile_len, border,
                is_left_bound=(t_start == 0),
                is_right_bound=(t_end >= T),
                dtype=z.dtype, device="cpu",
            )

            values[:, :, out_start:out_end] += tile_output * mask
            weight[:, :, out_start:out_end] += mask

        weight = weight.clamp(min=1e-8)
        return values / weight

    def decode(self, z: torch.Tensor, tiled: bool = False, tile_size: int = 512, tile_stride: int = 256) -> torch.Tensor:
        """Latent [B, decoder_input_channels, T] → audio waveform [B, audio_channels, T]."""
        if tiled:
            return self.tiled_decode(z, tile_size=tile_size, tile_stride=tile_stride)
        return self.decoder(z)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Full round-trip: encode → decode."""
        z = self.encode(sample)
        return self.decode(z)

    def remove_weight_norm(self):
        """Remove weight normalization from all conv layers (for export/inference)."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                remove_weight_norm(module)
