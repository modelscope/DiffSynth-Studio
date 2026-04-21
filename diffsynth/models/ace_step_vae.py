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
"""ACE-Step Audio VAE (AutoencoderOobleck CNN architecture).

This is a CNN-based VAE for audio waveform encoding/decoding.
It uses weight-normalized convolutions and Snake1d activations.
Does NOT depend on diffusers — pure nn.Module implementation.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Audio waveform [B, audio_channels, T] → latent [B, encoder_hidden_size, T']."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent [B, encoder_hidden_size, T] → audio waveform [B, audio_channels, T']."""
        return self.decoder(z)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Full round-trip: encode → decode."""
        z = self.encode(sample)
        return self.decoder(z)
