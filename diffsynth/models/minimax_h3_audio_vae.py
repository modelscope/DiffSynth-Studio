# SPDX-License-Identifier: Apache-2.0
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..core.attention import attention_forward


class WarpedTensor(torch.nn.Module):
    def __init__(self, weight=None, shape=None):
        super().__init__()
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.empty(shape))

    def forward(self):
        return self.weight


class WeightNormedConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty((out_channels, 1, 1)))
        self.weight_v = torch.nn.Parameter(torch.empty((out_channels, in_channels, kernel_size)))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_channels,)))
        else:
            self.bias = None

    def forward(self, input):
        dtype = self.weight_g.dtype
        weight = (self.weight_g.float() * self.weight_v.float() / torch.norm(self.weight_v.float(), dim=(1, 2), keepdim=True)).to(dtype)
        return torch.nn.functional.conv1d(
            input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class WeightNormedConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty((in_channels, 1, 1)))
        self.weight_v = torch.nn.Parameter(torch.empty((in_channels, out_channels, kernel_size)))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding,)
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_channels,)))
        else:
            self.bias = None

    def _output_padding(
        self,
        input,
        output_size: list[int] | None,
        stride: list[int],
        padding: list[int],
        kernel_size: list[int],
        num_spatial_dims: int,
        dilation: list[int] | None = None,
    ) -> list[int]:
        if output_size is None:
            ret = self.output_padding
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} "
                    f"or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )

            min_sizes = torch.jit.annotate(list[int], [])
            max_sizes = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})"
                    )

            res = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def forward(self, input, output_size=None):
        dtype = self.weight_g.dtype
        weight = (self.weight_g.float() * self.weight_v.float() / torch.norm(self.weight_v.float(), dim=(1, 2), keepdim=True)).to(dtype)
        assert isinstance(self.padding, tuple)
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )
        return F.conv_transpose1d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class Snake(nn.Module):
    def __init__(self, channels, beta=False):
        super().__init__()
        if not beta:
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))
            self.beta = None
        else:
            self.alpha = nn.Parameter(torch.ones(channels))
            self.beta = Parameter(torch.zeros(channels)) if beta else None

    def forward(self, x):
        alpha = self.alpha
        if len(alpha.shape) == 1: alpha = alpha.unsqueeze(0).unsqueeze(-1).exp()
        beta = self.alpha if self.beta is None else self.beta
        if len(beta.shape) == 1: beta = beta.unsqueeze(0).unsqueeze(-1).exp()
        x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.25, half_width=0.3, stride: int = 2, kernel_size: int = 12):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter = torch.nn.Parameter(kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        x = F.pad(x, (self.kernel_size // 2 - 1, self.kernel_size // 2), mode="replicate")
        out = F.conv1d(x, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, groups=x.shape[1])
        return out


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.filter = torch.nn.Parameter(filter)

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1).to(dtype=x.dtype, device=x.device), stride=self.stride, groups=C)
        x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size)

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.proj = nn.Linear(out_dim, out_dim)
        self.q_bias = WarpedTensor(torch.empty(in_dim))
        self.zero_k_bias = WarpedTensor(torch.empty(in_dim))
        self.v_bias = WarpedTensor(torch.empty(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = q + self.q_bias(), k + self.zero_k_bias(), v + self.v_bias()
        x = attention_forward(q, k, v, q_pattern="b s (n d)", k_pattern="b s (n d)", v_pattern="b s (n d)", out_pattern="b n s d", dims={"n": self.num_heads}, is_causal=True)
        x = torch.mean(x, dim=1)
        x = F.adaptive_avg_pool1d(x, self.out_dim)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=out_dim * mlp_ratio)

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake(dim),
            WeightNormedConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake(dim),
            WeightNormedConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake(dim // 2),
            WeightNormedConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: list = [2, 4, 8, 8], d_latent: int = 64):
        super().__init__()
        self.block = [WeightNormedConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake(d_model),
            WeightNormedConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([WeightNormedConv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=int((kernel_size * d - d) / 2)) for d in dilation])
        self.convs2 = nn.ModuleList([WeightNormedConv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=int((kernel_size - 1) / 2)) for _ in range(len(dilation))])
        self.activations = nn.ModuleList([Activation1d(activation=Snake(channels, beta=True)) for _ in range(2 * len(dilation))])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = WeightNormedConv1d(2048, 1024, 7, 1, padding=3)
        self.ups = nn.ModuleList([
            nn.ModuleList([WeightNormedConvTranspose1d(1024 // (2**i), 1024 // (2**(i+1)), k, u, padding=(k-u)//2)])
            for i, (u, k) in enumerate(zip([5, 5, 2, 2, 2, 2, 2], [9, 9, 4, 4, 4, 4, 4]))
        ])
        self.resblocks = nn.ModuleList()
        for i in range(7):
            ch = 1024 // (2 ** (i + 1))
            for k in [3, 7, 11]:
                self.resblocks.append(AMPBlock1(ch, k, (1, 3, 5)))
        self.activation_post = Activation1d(activation=Snake(ch, beta=True))
        self.conv_post = WeightNormedConv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up[0](x)
            xs = self.resblocks[i * 3](x)
            for j in range(1, 3):
                xs += self.resblocks[i * 3 + j](x)
            x = xs / 3
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x


_AUDIO_LATENTS_MEAN = [-0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885, -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391, -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163, -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344, 0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797]
_AUDIO_LATENTS_STD = [1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178, 1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373, 1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093, 1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912, 1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363]


class MiniMaxH3AudioVAE(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 4, 5, 5],
        decoder_dim: int = 1024,
        decoder_rates: List[int] = [5, 5, 2, 2, 2, 2, 2],
        sample_rate: int = 32000,
        vae_latent_channels: int = 32,
        attn_proj: bool = True,
        decoder_type: str = "bigvgan",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.attn_proj = attn_proj
        latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.hop_length = int(np.prod(encoder_rates))
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.mean_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.logs_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN()
        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)

    def _latent_stats(self, device):
        mean = torch.tensor(_AUDIO_LATENTS_MEAN, device=device, dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(_AUDIO_LATENTS_STD, device=device, dtype=torch.float32).view(1, -1, 1)
        return mean, std

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        z = self.encoder(audio_data)
        if self.attn_proj:
            z = self.pre_block(z.transpose(1, 2)).transpose(1, 2)
        mean = self.mean_proj(z)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec_in_proj(z)
        return self.decoder(z)

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor, dtype=None) -> torch.Tensor:
        if waveform.dim() != 2:
            raise ValueError(f"expected waveform [C, L], got {list(waveform.shape)}")
        out_dtype = waveform.dtype
        audio_data = waveform.unsqueeze(1).to(dtype if dtype is not None else out_dtype)
        latent = self.encode(self.preprocess(audio_data))
        latent_channels = len(_AUDIO_LATENTS_MEAN)
        if latent.dim() != 3 or latent.shape[1] != latent_channels:
            raise ValueError(
                f"expected audio latent [C, {latent_channels}, T], got {list(latent.shape)}"
            )
        mean, std = self._latent_stats(latent.device)
        return ((latent.to(torch.float32) - mean) / std).to(out_dtype)

    @torch.no_grad()
    def decode_audio(self, latents: torch.Tensor, dtype=None) -> torch.Tensor:
        mean, std = self._latent_stats(latents.device)
        z = (latents.to(latents.device, torch.float32) * std + mean).to(dtype if dtype is not None else latents.dtype)
        waveform = self.decode(z)
        return waveform.transpose(0, 1).to(latents.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)
