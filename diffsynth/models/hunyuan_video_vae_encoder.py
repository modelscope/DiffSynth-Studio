import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
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

    def forward(self, hidden_states):
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
        self.scaling_factor = 0.476986


    def forward(self, images):
        latents = self.encoder(images)
        latents = self.quant_conv(latents)
        latents = latents[:, :16]
        latents = latents * self.scaling_factor
        return latents
    

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x
    

    def build_mask(self, data, is_bound, border_width):
        _, _, T, H, W = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        h = self.build_1d_mask(H, is_bound[2], is_bound[3], border_width[1])
        w = self.build_1d_mask(W, is_bound[4], is_bound[5], border_width[2])

        t = repeat(t, "T -> T H W", T=T, H=H, W=W)
        h = repeat(h, "H -> T H W", T=T, H=H, W=W)
        w = repeat(w, "W -> T H W", T=T, H=H, W=W)

        mask = torch.stack([t, h, w]).min(dim=0).values
        mask = rearrange(mask, "T H W -> 1 1 T H W")
        return mask
    

    def tile_forward(self, hidden_states, tile_size, tile_stride):
        B, C, T, H, W = hidden_states.shape
        size_t, size_h, size_w = tile_size
        stride_t, stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for t in range(0, T, stride_t):
            if (t-stride_t >= 0 and t-stride_t+size_t >= T): continue
            for h in range(0, H, stride_h):
                if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
                for w in range(0, W, stride_w):
                    if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                    t_, h_, w_ = t + size_t, h + size_h, w + size_w
                    tasks.append((t, t_, h, h_, w, w_))

        # Run
        torch_dtype = self.quant_conv.weight.dtype
        data_device = hidden_states.device
        computation_device = self.quant_conv.weight.device

        weight = torch.zeros((1, 1,  (T - 1) // 4 + 1, H // 8, W // 8), dtype=torch_dtype, device=data_device)
        values = torch.zeros((B, 16, (T - 1) // 4 + 1, H // 8, W // 8), dtype=torch_dtype, device=data_device)

        for t, t_, h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = hidden_states[:, :, t:t_, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.forward(hidden_states_batch).to(data_device)
            if t > 0:
                hidden_states_batch = hidden_states_batch[:, :, 1:]

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(t==0, t_>=T, h==0, h_>=H, w==0, w_>=W),
                border_width=((size_t - stride_t) // 4, (size_h - stride_h) // 8, (size_w - stride_w) // 8)
            ).to(dtype=torch_dtype, device=data_device)

            target_t = 0 if t==0 else t // 4 + 1
            target_h = h // 8
            target_w = w // 8
            values[
                :,
                :,
                target_t: target_t + hidden_states_batch.shape[2],
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                target_t: target_t + hidden_states_batch.shape[2],
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        return values / weight


    def encode_video(self, latents, tile_size=(65, 256, 256), tile_stride=(48, 192, 192)):
        latents = latents.to(self.quant_conv.weight.dtype)
        return self.tile_forward(latents, tile_size=tile_size, tile_stride=tile_stride)


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
