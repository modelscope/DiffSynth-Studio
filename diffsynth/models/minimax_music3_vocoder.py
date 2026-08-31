import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMaxMusic3WeightNormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty((out_channels, 1, 1)))
        self.weight_v = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size)))
        self.bias = nn.Parameter(torch.empty((out_channels,))) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = self.weight_g.dtype
        weight = (self.weight_g.float() * self.weight_v.float() / torch.norm(self.weight_v.float(), dim=(1, 2), keepdim=True)).to(dtype)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MiniMaxMusic3WeightNormConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty((in_channels, 1, 1)))
        self.weight_v = nn.Parameter(torch.empty((in_channels, out_channels, kernel_size)))
        self.bias = nn.Parameter(torch.empty((out_channels,))) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = self.weight_g.dtype
        weight = (self.weight_g.float() * self.weight_v.float() / torch.norm(self.weight_v.float(), dim=(1, 2), keepdim=True)).to(dtype)
        return F.conv_transpose1d(
            input, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )


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
        self.snake1 = MiniMaxMusic3Snake1d(dim)
        self.conv1 = MiniMaxMusic3WeightNormConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = MiniMaxMusic3Snake1d(dim)
        self.conv2 = MiniMaxMusic3WeightNormConv1d(dim, dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim)
        self.conv_t1 = MiniMaxMusic3WeightNormConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


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
        self.conv_in = MiniMaxMusic3WeightNormConv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3)
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2 ** index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(output_dim)
        self.conv_out = MiniMaxMusic3WeightNormConv1d(output_dim, 1, kernel_size=7, padding=3)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.latent_channels // 2, length)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch_size, 2, -1)
