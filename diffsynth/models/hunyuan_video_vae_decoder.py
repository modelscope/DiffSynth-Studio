import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from tqdm import tqdm
from einops import repeat


class CausalConv3d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0
                                   )  # W, H, T
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class UpsampleCausal3D(nn.Module):

    def __init__(self, channels, use_conv=False, out_channels=None, kernel_size=None, bias=True, upsample_factor=(2, 2, 2)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample_factor = upsample_factor
        self.conv = None
        if use_conv:
            kernel_size = 3 if kernel_size is None else kernel_size
            self.conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, hidden_states):
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # interpolate
        B, C, T, H, W = hidden_states.shape
        first_h, other_h = hidden_states.split((1, T - 1), dim=2)
        if T > 1:
            other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")
        first_h = F.interpolate(first_h.squeeze(2), scale_factor=self.upsample_factor[1:], mode="nearest").unsqueeze(2)
        hidden_states = torch.cat((first_h, other_h), dim=2) if T > 1 else first_h

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlockCausal3D(nn.Module):

    def __init__(self, in_channels, out_channels=None, dropout=0.0, groups=32, eps=1e-6, conv_shortcut_bias=True):
        super().__init__()
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1)

        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=conv_shortcut_bias)

    def forward(self, input_tensor):
        hidden_states = input_tensor
        # conv1
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # conv2
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        # shortcut
        if self.conv_shortcut is not None:
            input_tensor = (self.conv_shortcut(input_tensor))
        # shortcut and scale
        output_tensor = input_tensor + hidden_states

        return output_tensor


def prepare_causal_attention_mask(n_frame, n_hw, dtype, device, batch_size=None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, :(i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class Attention(nn.Module):

    def __init__(self,
                 in_channels,
                 num_heads,
                 head_dim,
                 num_groups=32,
                 dropout=0.0,
                 eps=1e-6,
                 bias=True,
                 residual_connection=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.residual_connection = residual_connection
        dim_inner = head_dim * num_heads
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.to_q = nn.Linear(in_channels, dim_inner, bias=bias)
        self.to_k = nn.Linear(in_channels, dim_inner, bias=bias)
        self.to_v = nn.Linear(in_channels, dim_inner, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, in_channels, bias=bias), nn.Dropout(dropout))

    def forward(self, input_tensor, attn_mask=None):
        hidden_states = self.group_norm(input_tensor.transpose(1, 2)).transpose(1, 2)
        batch_size = hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, self.num_heads, -1, attn_mask.shape[-1])
        hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = self.to_out(hidden_states)
        if self.residual_connection:
            output_tensor = input_tensor + hidden_states
        return output_tensor


class UNetMidBlockCausal3D(nn.Module):

    def __init__(self, in_channels, dropout=0.0, num_layers=1, eps=1e-6, num_groups=32, attention_head_dim=None):
        super().__init__()
        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout=dropout,
                groups=num_groups,
                eps=eps,
            )
        ]
        attentions = []
        attention_head_dim = attention_head_dim or in_channels

        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    num_heads=in_channels // attention_head_dim,
                    head_dim=attention_head_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                    eps=eps,
                    bias=True,
                    residual_connection=True,
                ))

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    groups=num_groups,
                    eps=eps,
                ))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            B, C, T, H, W = hidden_states.shape
            hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
            attn_mask = prepare_causal_attention_mask(T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B)
            hidden_states = attn(hidden_states, attn_mask=attn_mask)
            hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
            hidden_states = resnet(hidden_states)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dropout=0.0,
            num_layers=1,
            eps=1e-6,
            num_groups=32,
            add_upsample=True,
            upsample_scale_factor=(2, 2, 2),
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

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpsampleCausal3D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    upsample_factor=upsample_scale_factor,
                )
            ])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class DecoderCausal3D(nn.Module):

    def __init__(
        self,
        in_channels=16,
        out_channels=3,
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
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1)
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            dropout=dropout,
            eps=eps,
            num_groups=num_groups,
            attention_head_dim=block_out_channels[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            add_spatial_upsample = bool(i < num_spatial_upsample_layers)
            add_time_upsample = bool(i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block)

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)

            up_block = UpDecoderBlockCausal3D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                eps=eps,
                num_groups=num_groups,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups, eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_states,
                use_reentrant=False,
            )
            # up
            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    hidden_states,
                    use_reentrant=False,
                )
        else:
            # middle
            hidden_states = self.mid_block(hidden_states)
            # up
            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states)
        # post-process
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoVAEDecoder(nn.Module):

    def __init__(
        self,
        in_channels=16,
        out_channels=3,
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
        self.decoder = DecoderCausal3D(
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
        self.post_quant_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.scaling_factor = 0.476986


    def forward(self, latents):
        latents = latents / self.scaling_factor
        latents = self.post_quant_conv(latents)
        dec = self.decoder(latents)
        return dec
    

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
        torch_dtype = self.post_quant_conv.weight.dtype
        data_device = hidden_states.device
        computation_device = self.post_quant_conv.weight.device

        weight = torch.zeros((1, 1, (T - 1) * 4 + 1, H * 8, W * 8), dtype=torch_dtype, device=data_device)
        values = torch.zeros((B, 3, (T - 1) * 4 + 1, H * 8, W * 8), dtype=torch_dtype, device=data_device)

        for t, t_, h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, t:t_, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.forward(hidden_states_batch).to(data_device)
            if t > 0:
                hidden_states_batch = hidden_states_batch[:, :, 1:]

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(t==0, t_>=T, h==0, h_>=H, w==0, w_>=W),
                border_width=((size_t - stride_t) * 4, (size_h - stride_h) * 8, (size_w - stride_w) * 8)
            ).to(dtype=torch_dtype, device=data_device)

            target_t = 0 if t==0 else t * 4 + 1
            target_h = h * 8
            target_w = w * 8
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


    def decode_video(self, latents, tile_size=(17, 32, 32), tile_stride=(12, 24, 24)):
        latents = latents.to(self.post_quant_conv.weight.dtype)
        return self.tile_forward(latents, tile_size=tile_size, tile_stride=tile_stride)

    @staticmethod
    def state_dict_converter():
        return HunyuanVideoVAEDecoderStateDictConverter()


class HunyuanVideoVAEDecoderStateDictConverter:

    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for name in state_dict:
            if name.startswith('decoder.') or name.startswith('post_quant_conv.'):
                state_dict_[name] = state_dict[name]
        return state_dict_
