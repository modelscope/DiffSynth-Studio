# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ===== Time Embedding =====

class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos=True, freq_shift=0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim + self.freq_shift
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        if self.flip_sin_to_cos:
            emb = torch.cat([cos_emb, sin_emb], dim=-1)
        else:
            emb = torch.cat([sin_emb, cos_emb], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, act_fn="silu", out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        out_dim = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, out_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# ===== ResNet Blocks =====

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels or in_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels or in_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * (out_channels or in_channels))

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels or in_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels or in_channels, out_channels or in_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()
        elif non_linearity == "relu":
            self.nonlinearity = nn.ReLU()

        self.use_conv_shortcut = conv_shortcut
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels or in_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels or in_channels, kernel_size=1, stride=1, padding=0) if in_channels != (out_channels or in_channels) else None

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb).unsqueeze(-1).unsqueeze(-1)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


# ===== Transformer Blocks =====

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, dropout=0.0):
        super().__init__()
        self.net = nn.ModuleList([
            GEGLU(dim, dim * 4),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim if dim_out is None else dim_out),
        ])

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Attention block matching diffusers checkpoint key format.
    Keys: to_q.weight, to_k.weight, to_v.weight, to_out.0.weight, to_out.0.bias
    """
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        bias=False,
        upcast_attention=False,
        cross_attention_dim=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.inner_dim = inner_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Dropout(dropout),
        ])

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Query
        query = self.to_q(hidden_states)
        batch_size, seq_len, _ = query.shape

        # Key/Value
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        head_dim = self.inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        cross_attention_dim=None,
        upcast_attention=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Self-attention
        attn_output = self.attn1(self.norm1(hidden_states))
        hidden_states = attn_output + hidden_states
        # Cross-attention
        attn_output = self.attn2(self.norm2(hidden_states), encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states
        # Feed-forward
        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    """2D Transformer block wrapper matching diffusers checkpoint structure.
    Keys: norm.weight/bias, proj_in.weight/bias, transformer_blocks.X.*, proj_out.weight/bias
    """
    def __init__(
        self,
        num_attention_heads=16,
        attention_head_dim=64,
        in_channels=320,
        num_layers=1,
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        upcast_attention=False,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, num_attention_heads * attention_head_dim, kernel_size=1, bias=True)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=num_attention_heads * attention_head_dim,
                n_heads=num_attention_heads,
                d_head=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv2d(num_attention_heads * attention_head_dim, in_channels, kernel_size=1, bias=True)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        # Normalize and project to sequence
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, -1, channel)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        # Project back to 2D
        hidden_states = hidden_states.reshape(batch, height, width, channel).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


# ===== Down/Up Blocks =====

class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels=1280,
        dropout=0.0,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        cross_attention_dim=768,
        attention_head_dim=1,
        downsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True

        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels_i = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels_i,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attention_head_dim,
                    attention_head_dim=out_channels // attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    dropout=dropout,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels, padding=1)
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, tuple(output_states)


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels=1280,
        dropout=0.0,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        downsample=True,
    ):
        super().__init__()
        self.has_cross_attention = False

        resnets = []
        for i in range(num_layers):
            in_channels_i = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels_i,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels, padding=1)
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, tuple(output_states)


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels=1280,
        dropout=0.0,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        cross_attention_dim=768,
        attention_head_dim=1,
        upsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True

        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attention_head_dim,
                    attention_head_dim=out_channels // attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    dropout=dropout,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, upsample_size=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            # Pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size=upsample_size)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels=1280,
        dropout=0.0,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        upsample=True,
    ):
        super().__init__()
        self.has_cross_attention = False

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, out_channels)
            ])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, upsample_size=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size=upsample_size)

        return hidden_states


# ===== UNet Mid Block =====

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        temb_channels=1280,
        dropout=0.0,
        num_layers=1,
        transformer_layers_per_block=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        cross_attention_dim=768,
        attention_head_dim=1,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # There is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=1.0,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    num_attention_heads=attention_head_dim,
                    attention_head_dim=in_channels // attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    dropout=dropout,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=cross_attention_dim,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=1.0,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


# ===== Downsample / Upsample =====

class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding)
        self.padding = padding

    def forward(self, hidden_states):
        if self.padding == 0:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states, upsample_size=None):
        if upsample_size is not None:
            hidden_states = F.interpolate(hidden_states, size=upsample_size, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


# ===== UNet2DConditionModel =====

class UNet2DConditionModel(nn.Module):
    """Stable Diffusion UNet with cross-attention conditioning.
    state_dict keys match the diffusers UNet2DConditionModel checkpoint format.
    """
    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        cross_attention_dim=768,
        attention_head_dim=8,
        norm_num_groups=32,
        norm_eps=1e-5,
        dropout=0.0,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=True,
        freq_shift=0,
        time_embedding_dim=None,
        resnet_time_scale_shift="default",
        upcast_attention=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size

        # Time embedding
        timestep_embedding_dim = time_embedding_dim or block_out_channels[0]
        self.time_proj = Timesteps(timestep_embedding_dim, flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift)
        time_embed_dim = block_out_channels[0] * 4
        self.time_embedding = TimestepEmbedding(timestep_embedding_dim, time_embed_dim)

        # Input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if "CrossAttn" in down_block_type:
                down_block = CrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=1,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_head_dim=attention_head_dim,
                    downsample=not is_final_block,
                )
            else:
                down_block = DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample=not is_final_block,
                )
            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            num_layers=1,
            transformer_layers_per_block=1,
            resnet_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            # in_channels for up blocks: diffusers uses reversed_block_out_channels[min(i+1, len-1)]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if "CrossAttn" in up_block_type:
                up_block = CrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=1,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_head_dim=attention_head_dim,
                    upsample=not is_final_block,
                )
            else:
                up_block = UpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    upsample=not is_final_block,
                )
            self.up_blocks.append(up_block)

        # Output
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, timestep_cond=None, added_cond_kwargs=None, return_dict=True):
        # 1. Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Pre-process
        sample = self.conv_in(sample)

        # 3. Down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
            down_block_res_samples += res_samples

        # 4. Mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. Up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            upsample_size = down_block_res_samples[-1].shape[2:] if down_block_res_samples else None
            sample = up_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        return sample
