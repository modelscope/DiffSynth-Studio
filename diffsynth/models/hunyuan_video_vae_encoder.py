import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from .hunyuan_video_vae_decoder import CausalConv3d, ResnetBlockCausal3D, UNetMidBlockCausal3D


class DownsampleCausal3D(nn.Module):

    def __init__(self, channels, out_channels, kernel_size=3, bias=True, stride=2):
        super().__init__()
        self.conv = CausalConv3d(channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0,
        num_layers=1,
        eps=1e-6,
        num_groups=32,
        add_downsample=True,
        downsample_stride=2,
    ):

        super().__init__()
        resnets = []
        for i in range(num_layers):
            cur_in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=cur_in_channel,
                    out_channels=out_channels,
                    groups=num_groups,
                    dropout=dropout,
                    eps=eps,
                ))
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([DownsampleCausal3D(
                out_channels,
                out_channels,
                stride=downsample_stride,
            )])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class EncoderCausal3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        eps=1e-6,
        dropout=0.0,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        num_groups=32,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            add_spatial_downsample = bool(i < num_spatial_downsample_layers)
            add_time_downsample = bool(i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block)

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = DownEncoderBlockCausal3D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                num_layers=layers_per_block,
                eps=eps,
                num_groups=num_groups,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            dropout=dropout,
            eps=eps,
            num_groups=num_groups,
            attention_head_dim=block_out_channels[-1],
        )
        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=num_groups, eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[-1], 2 * out_channels, kernel_size=3)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.conv_in(hidden_states)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    hidden_states,
                    use_reentrant=False,
                )
            # middle
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                use_reentrant=False,
            )
        else:
            # down
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states)
            # middle
            hidden_states = self.mid_block(hidden_states)
        # post-process
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoVAEEncoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=16,
        eps=1e-6,
        dropout=0.0,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        num_groups=32,
        time_compression_ratio=4,
        spatial_compression_ratio=8,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.encoder = EncoderCausal3D(
            in_channels=in_channels,
            out_channels=out_channels,
            eps=eps,
            dropout=dropout,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            num_groups=num_groups,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.quant_conv = nn.Conv3d(2 * out_channels, 2 * out_channels, kernel_size=1)

    def encode_video(self, latents, use_temporal_tiling=False, use_spatial_tiling=False, sample_ssize=256, sample_tsize=64):
        if use_temporal_tiling:
            raise NotImplementedError
        if use_spatial_tiling:
            raise NotImplementedError
        # no tiling
        latents = self.decoder(latents)
        dec = self.quant_conv(latents)
        return dec

    @staticmethod
    def state_dict_converter():
        return HunyuanVideoVAEEncoderStateDictConverter()


class HunyuanVideoVAEEncoderStateDictConverter:

    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for name in state_dict:
            if name.startswith('encoder.') or name.startswith('quant_conv.'):
                state_dict_[name] = state_dict[name]
        return state_dict_
