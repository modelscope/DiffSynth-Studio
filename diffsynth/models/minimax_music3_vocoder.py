import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.ModuleList([
            MiniMaxMusic3Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            MiniMaxMusic3Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        for module in self.block:
            hidden_states = module(hidden_states)
        return residual + hidden_states


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.ModuleList([
            MiniMaxMusic3Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1),
            MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3),
            MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.block:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxMusic3VocoderDecoder(nn.Module):
    def __init__(self, decoder_input_dim: int, decoder_hidden_dim: int, upsampling_ratios):
        super().__init__()
        model = [WNConv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3)]
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2 ** index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            model.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        model.append(MiniMaxMusic3Snake1d(output_dim))
        model.append(WNConv1d(output_dim, 1, kernel_size=7, padding=3))
        self.model = nn.ModuleList(model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.model:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxMusic3Vocoder(nn.Module):

    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple = (8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.sampling_rate = sampling_rate
        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.decoder = MiniMaxMusic3VocoderDecoder(decoder_input_dim, decoder_hidden_dim, upsampling_ratios)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.latent_channels // 2, length)
        hidden_states = self.dec_in_proj(hidden_states)
        hidden_states = self.decoder(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform.reshape(batch_size, 2, -1)
