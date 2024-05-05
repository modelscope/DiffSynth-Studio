import torch
from .attention import Attention
from .sd_unet import ResnetBlock, UpSampler
from .tiler import TileWorker
from einops import rearrange, repeat


class VAEAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        self.transformer_blocks = torch.nn.ModuleList([
            Attention(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                bias_q=True,
                bias_kv=True,
                bias_out=True
            )
            for d in range(num_layers)
        ])

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack
    

class TemporalResnetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.nonlinearity = torch.nn.SiLU()
        self.mix_factor = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x_spatial = hidden_states
        x = rearrange(hidden_states, "T C H W -> 1 C T H W")
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x_temporal = hidden_states + x[0].permute(1, 0, 2, 3)
        alpha = torch.sigmoid(self.mix_factor)
        hidden_states = alpha * x_temporal + (1 - alpha) * x_spatial
        return hidden_states, time_emb, text_emb, res_stack
    

class SVDVAEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.18215
        self.conv_in = torch.nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # UNetMidBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            TemporalResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock
            ResnetBlock(512, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            TemporalResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock
            ResnetBlock(256, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            TemporalResnetBlock(128, 128, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.time_conv_out = torch.nn.Conv3d(3, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0))


    def forward(self, sample):
        # 1. pre-process
        hidden_states = rearrange(sample, "C T H W -> T C H W")
        hidden_states = hidden_states / self.scaling_factor
        hidden_states = self.conv_in(hidden_states)
        time_emb, text_emb, res_stack = None, None, None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = rearrange(hidden_states, "T C H W -> C T H W")
        hidden_states = self.time_conv_out(hidden_states)

        return hidden_states
    
    
    def build_mask(self, data, is_bound):
        _, T, H, W = data.shape
        t = repeat(torch.arange(T), "T -> T H W", T=T, H=H, W=W)
        h = repeat(torch.arange(H), "H -> T H W", T=T, H=H, W=W)
        w = repeat(torch.arange(W), "W -> T H W", T=T, H=H, W=W)
        border_width = (T + H + W) // 6
        pad = torch.ones_like(t) * border_width
        mask = torch.stack([
            pad if is_bound[0] else t + 1,
            pad if is_bound[1] else T - t,
            pad if is_bound[2] else h + 1,
            pad if is_bound[3] else H - h,
            pad if is_bound[4] else w + 1,
            pad if is_bound[5] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=data.dtype, device=data.device)
        mask = rearrange(mask, "T H W -> 1 T H W")
        return mask
    

    def decode_video(
        self, sample,
        batch_time=8, batch_height=128, batch_width=128,
        stride_time=4, stride_height=32, stride_width=32,
        progress_bar=lambda x:x
    ):
        sample = sample.permute(1, 0, 2, 3)
        data_device = sample.device
        computation_device = self.conv_in.weight.device
        torch_dtype = sample.dtype
        _, T, H, W = sample.shape

        weight = torch.zeros((1, T, H*8, W*8), dtype=torch_dtype, device=data_device)
        values = torch.zeros((3, T, H*8, W*8), dtype=torch_dtype, device=data_device)

        # Split tasks
        tasks = []
        for t in range(0, T, stride_time):
            for h in range(0, H, stride_height):
                for w in range(0, W, stride_width):
                    if (t-stride_time >= 0 and t-stride_time+batch_time >= T)\
                        or (h-stride_height >= 0 and h-stride_height+batch_height >= H)\
                        or (w-stride_width >= 0 and w-stride_width+batch_width >= W):
                        continue
                    tasks.append((t, t+batch_time, h, h+batch_height, w, w+batch_width))
        
        # Run
        for tl, tr, hl, hr, wl, wr in progress_bar(tasks):
            sample_batch = sample[:, tl:tr, hl:hr, wl:wr].to(computation_device)
            sample_batch = self.forward(sample_batch).to(data_device)
            mask = self.build_mask(sample_batch, is_bound=(tl==0, tr>=T, hl==0, hr>=H, wl==0, wr>=W))
            values[:, tl:tr, hl*8:hr*8, wl*8:wr*8] += sample_batch * mask
            weight[:, tl:tr, hl*8:hr*8, wl*8:wr*8] += mask
        values /= weight
        return values
    
    
    def state_dict_converter(self):
        return SVDVAEDecoderStateDictConverter()
    

class SVDVAEDecoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        static_rename_dict = {
            "decoder.conv_in":  "conv_in",
            "decoder.mid_block.attentions.0.group_norm": "blocks.2.norm",
            "decoder.mid_block.attentions.0.to_q": "blocks.2.transformer_blocks.0.to_q",
            "decoder.mid_block.attentions.0.to_k": "blocks.2.transformer_blocks.0.to_k",
            "decoder.mid_block.attentions.0.to_v": "blocks.2.transformer_blocks.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0": "blocks.2.transformer_blocks.0.to_out",
            "decoder.up_blocks.0.upsamplers.0.conv": "blocks.11.conv",
            "decoder.up_blocks.1.upsamplers.0.conv": "blocks.18.conv",
            "decoder.up_blocks.2.upsamplers.0.conv": "blocks.25.conv",
            "decoder.conv_norm_out": "conv_norm_out",
            "decoder.conv_out": "conv_out",
            "decoder.time_conv_out": "time_conv_out"
        }
        prefix_rename_dict = {
            "decoder.mid_block.resnets.0.spatial_res_block": "blocks.0",
            "decoder.mid_block.resnets.0.temporal_res_block": "blocks.1",
            "decoder.mid_block.resnets.0.time_mixer": "blocks.1",
            "decoder.mid_block.resnets.1.spatial_res_block": "blocks.3",
            "decoder.mid_block.resnets.1.temporal_res_block": "blocks.4",
            "decoder.mid_block.resnets.1.time_mixer": "blocks.4",

            "decoder.up_blocks.0.resnets.0.spatial_res_block": "blocks.5",
            "decoder.up_blocks.0.resnets.0.temporal_res_block": "blocks.6",
            "decoder.up_blocks.0.resnets.0.time_mixer": "blocks.6",
            "decoder.up_blocks.0.resnets.1.spatial_res_block": "blocks.7",
            "decoder.up_blocks.0.resnets.1.temporal_res_block": "blocks.8",
            "decoder.up_blocks.0.resnets.1.time_mixer": "blocks.8",
            "decoder.up_blocks.0.resnets.2.spatial_res_block": "blocks.9",
            "decoder.up_blocks.0.resnets.2.temporal_res_block": "blocks.10",
            "decoder.up_blocks.0.resnets.2.time_mixer": "blocks.10",

            "decoder.up_blocks.1.resnets.0.spatial_res_block": "blocks.12",
            "decoder.up_blocks.1.resnets.0.temporal_res_block": "blocks.13",
            "decoder.up_blocks.1.resnets.0.time_mixer": "blocks.13",
            "decoder.up_blocks.1.resnets.1.spatial_res_block": "blocks.14",
            "decoder.up_blocks.1.resnets.1.temporal_res_block": "blocks.15",
            "decoder.up_blocks.1.resnets.1.time_mixer": "blocks.15",
            "decoder.up_blocks.1.resnets.2.spatial_res_block": "blocks.16",
            "decoder.up_blocks.1.resnets.2.temporal_res_block": "blocks.17",
            "decoder.up_blocks.1.resnets.2.time_mixer": "blocks.17",

            "decoder.up_blocks.2.resnets.0.spatial_res_block": "blocks.19",
            "decoder.up_blocks.2.resnets.0.temporal_res_block": "blocks.20",
            "decoder.up_blocks.2.resnets.0.time_mixer": "blocks.20",
            "decoder.up_blocks.2.resnets.1.spatial_res_block": "blocks.21",
            "decoder.up_blocks.2.resnets.1.temporal_res_block": "blocks.22",
            "decoder.up_blocks.2.resnets.1.time_mixer": "blocks.22",
            "decoder.up_blocks.2.resnets.2.spatial_res_block": "blocks.23",
            "decoder.up_blocks.2.resnets.2.temporal_res_block": "blocks.24",
            "decoder.up_blocks.2.resnets.2.time_mixer": "blocks.24",

            "decoder.up_blocks.3.resnets.0.spatial_res_block": "blocks.26",
            "decoder.up_blocks.3.resnets.0.temporal_res_block": "blocks.27",
            "decoder.up_blocks.3.resnets.0.time_mixer": "blocks.27",
            "decoder.up_blocks.3.resnets.1.spatial_res_block": "blocks.28",
            "decoder.up_blocks.3.resnets.1.temporal_res_block": "blocks.29",
            "decoder.up_blocks.3.resnets.1.time_mixer": "blocks.29",
            "decoder.up_blocks.3.resnets.2.spatial_res_block": "blocks.30",
            "decoder.up_blocks.3.resnets.2.temporal_res_block": "blocks.31",
            "decoder.up_blocks.3.resnets.2.time_mixer": "blocks.31",
        }
        suffix_rename_dict = {
            "norm1.weight": "norm1.weight",
            "conv1.weight": "conv1.weight",
            "norm2.weight": "norm2.weight",
            "conv2.weight": "conv2.weight",
            "conv_shortcut.weight": "conv_shortcut.weight",
            "norm1.bias": "norm1.bias",
            "conv1.bias": "conv1.bias",
            "norm2.bias": "norm2.bias",
            "conv2.bias": "conv2.bias",
            "conv_shortcut.bias": "conv_shortcut.bias",
            "mix_factor": "mix_factor",
        }

        state_dict_ = {}
        for name in static_rename_dict:
            state_dict_[static_rename_dict[name] + ".weight"] = state_dict[name + ".weight"]
            state_dict_[static_rename_dict[name] + ".bias"] = state_dict[name + ".bias"]
        for prefix_name in prefix_rename_dict:
            for suffix_name in suffix_rename_dict:
                name = prefix_name + "." + suffix_name
                name_ = prefix_rename_dict[prefix_name] + "." + suffix_rename_dict[suffix_name]
                if name in state_dict:
                    state_dict_[name_] = state_dict[name]
        
        return state_dict_
    

    def from_civitai(self, state_dict):
        rename_dict = {
            "first_stage_model.decoder.conv_in.bias": "conv_in.bias",
            "first_stage_model.decoder.conv_in.weight": "conv_in.weight",
            "first_stage_model.decoder.conv_out.bias": "conv_out.bias",
            "first_stage_model.decoder.conv_out.time_mix_conv.bias": "time_conv_out.bias",
            "first_stage_model.decoder.conv_out.time_mix_conv.weight": "time_conv_out.weight",
            "first_stage_model.decoder.conv_out.weight": "conv_out.weight",
            "first_stage_model.decoder.mid.attn_1.k.bias": "blocks.2.transformer_blocks.0.to_k.bias",
            "first_stage_model.decoder.mid.attn_1.k.weight": "blocks.2.transformer_blocks.0.to_k.weight",
            "first_stage_model.decoder.mid.attn_1.norm.bias": "blocks.2.norm.bias",
            "first_stage_model.decoder.mid.attn_1.norm.weight": "blocks.2.norm.weight",
            "first_stage_model.decoder.mid.attn_1.proj_out.bias": "blocks.2.transformer_blocks.0.to_out.bias",
            "first_stage_model.decoder.mid.attn_1.proj_out.weight": "blocks.2.transformer_blocks.0.to_out.weight",
            "first_stage_model.decoder.mid.attn_1.q.bias": "blocks.2.transformer_blocks.0.to_q.bias",
            "first_stage_model.decoder.mid.attn_1.q.weight": "blocks.2.transformer_blocks.0.to_q.weight",
            "first_stage_model.decoder.mid.attn_1.v.bias": "blocks.2.transformer_blocks.0.to_v.bias",
            "first_stage_model.decoder.mid.attn_1.v.weight": "blocks.2.transformer_blocks.0.to_v.weight",
            "first_stage_model.decoder.mid.block_1.conv1.bias": "blocks.0.conv1.bias",
            "first_stage_model.decoder.mid.block_1.conv1.weight": "blocks.0.conv1.weight",
            "first_stage_model.decoder.mid.block_1.conv2.bias": "blocks.0.conv2.bias",
            "first_stage_model.decoder.mid.block_1.conv2.weight": "blocks.0.conv2.weight",
            "first_stage_model.decoder.mid.block_1.mix_factor": "blocks.1.mix_factor",
            "first_stage_model.decoder.mid.block_1.norm1.bias": "blocks.0.norm1.bias",
            "first_stage_model.decoder.mid.block_1.norm1.weight": "blocks.0.norm1.weight",
            "first_stage_model.decoder.mid.block_1.norm2.bias": "blocks.0.norm2.bias",
            "first_stage_model.decoder.mid.block_1.norm2.weight": "blocks.0.norm2.weight",
            "first_stage_model.decoder.mid.block_1.time_stack.in_layers.0.bias": "blocks.1.norm1.bias",
            "first_stage_model.decoder.mid.block_1.time_stack.in_layers.0.weight": "blocks.1.norm1.weight",
            "first_stage_model.decoder.mid.block_1.time_stack.in_layers.2.bias": "blocks.1.conv1.bias",
            "first_stage_model.decoder.mid.block_1.time_stack.in_layers.2.weight": "blocks.1.conv1.weight",
            "first_stage_model.decoder.mid.block_1.time_stack.out_layers.0.bias": "blocks.1.norm2.bias",
            "first_stage_model.decoder.mid.block_1.time_stack.out_layers.0.weight": "blocks.1.norm2.weight",
            "first_stage_model.decoder.mid.block_1.time_stack.out_layers.3.bias": "blocks.1.conv2.bias",
            "first_stage_model.decoder.mid.block_1.time_stack.out_layers.3.weight": "blocks.1.conv2.weight",
            "first_stage_model.decoder.mid.block_2.conv1.bias": "blocks.3.conv1.bias",
            "first_stage_model.decoder.mid.block_2.conv1.weight": "blocks.3.conv1.weight",
            "first_stage_model.decoder.mid.block_2.conv2.bias": "blocks.3.conv2.bias",
            "first_stage_model.decoder.mid.block_2.conv2.weight": "blocks.3.conv2.weight",
            "first_stage_model.decoder.mid.block_2.mix_factor": "blocks.4.mix_factor",
            "first_stage_model.decoder.mid.block_2.norm1.bias": "blocks.3.norm1.bias",
            "first_stage_model.decoder.mid.block_2.norm1.weight": "blocks.3.norm1.weight",
            "first_stage_model.decoder.mid.block_2.norm2.bias": "blocks.3.norm2.bias",
            "first_stage_model.decoder.mid.block_2.norm2.weight": "blocks.3.norm2.weight",
            "first_stage_model.decoder.mid.block_2.time_stack.in_layers.0.bias": "blocks.4.norm1.bias",
            "first_stage_model.decoder.mid.block_2.time_stack.in_layers.0.weight": "blocks.4.norm1.weight",
            "first_stage_model.decoder.mid.block_2.time_stack.in_layers.2.bias": "blocks.4.conv1.bias",
            "first_stage_model.decoder.mid.block_2.time_stack.in_layers.2.weight": "blocks.4.conv1.weight",
            "first_stage_model.decoder.mid.block_2.time_stack.out_layers.0.bias": "blocks.4.norm2.bias",
            "first_stage_model.decoder.mid.block_2.time_stack.out_layers.0.weight": "blocks.4.norm2.weight",
            "first_stage_model.decoder.mid.block_2.time_stack.out_layers.3.bias": "blocks.4.conv2.bias",
            "first_stage_model.decoder.mid.block_2.time_stack.out_layers.3.weight": "blocks.4.conv2.weight",
            "first_stage_model.decoder.norm_out.bias": "conv_norm_out.bias",
            "first_stage_model.decoder.norm_out.weight": "conv_norm_out.weight",
            "first_stage_model.decoder.up.0.block.0.conv1.bias": "blocks.26.conv1.bias",
            "first_stage_model.decoder.up.0.block.0.conv1.weight": "blocks.26.conv1.weight",
            "first_stage_model.decoder.up.0.block.0.conv2.bias": "blocks.26.conv2.bias",
            "first_stage_model.decoder.up.0.block.0.conv2.weight": "blocks.26.conv2.weight",
            "first_stage_model.decoder.up.0.block.0.mix_factor": "blocks.27.mix_factor",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.bias": "blocks.26.conv_shortcut.bias",
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.weight": "blocks.26.conv_shortcut.weight",
            "first_stage_model.decoder.up.0.block.0.norm1.bias": "blocks.26.norm1.bias",
            "first_stage_model.decoder.up.0.block.0.norm1.weight": "blocks.26.norm1.weight",
            "first_stage_model.decoder.up.0.block.0.norm2.bias": "blocks.26.norm2.bias",
            "first_stage_model.decoder.up.0.block.0.norm2.weight": "blocks.26.norm2.weight",
            "first_stage_model.decoder.up.0.block.0.time_stack.in_layers.0.bias": "blocks.27.norm1.bias",
            "first_stage_model.decoder.up.0.block.0.time_stack.in_layers.0.weight": "blocks.27.norm1.weight",
            "first_stage_model.decoder.up.0.block.0.time_stack.in_layers.2.bias": "blocks.27.conv1.bias",
            "first_stage_model.decoder.up.0.block.0.time_stack.in_layers.2.weight": "blocks.27.conv1.weight",
            "first_stage_model.decoder.up.0.block.0.time_stack.out_layers.0.bias": "blocks.27.norm2.bias",
            "first_stage_model.decoder.up.0.block.0.time_stack.out_layers.0.weight": "blocks.27.norm2.weight",
            "first_stage_model.decoder.up.0.block.0.time_stack.out_layers.3.bias": "blocks.27.conv2.bias",
            "first_stage_model.decoder.up.0.block.0.time_stack.out_layers.3.weight": "blocks.27.conv2.weight",
            "first_stage_model.decoder.up.0.block.1.conv1.bias": "blocks.28.conv1.bias",
            "first_stage_model.decoder.up.0.block.1.conv1.weight": "blocks.28.conv1.weight",
            "first_stage_model.decoder.up.0.block.1.conv2.bias": "blocks.28.conv2.bias",
            "first_stage_model.decoder.up.0.block.1.conv2.weight": "blocks.28.conv2.weight",
            "first_stage_model.decoder.up.0.block.1.mix_factor": "blocks.29.mix_factor",
            "first_stage_model.decoder.up.0.block.1.norm1.bias": "blocks.28.norm1.bias",
            "first_stage_model.decoder.up.0.block.1.norm1.weight": "blocks.28.norm1.weight",
            "first_stage_model.decoder.up.0.block.1.norm2.bias": "blocks.28.norm2.bias",
            "first_stage_model.decoder.up.0.block.1.norm2.weight": "blocks.28.norm2.weight",
            "first_stage_model.decoder.up.0.block.1.time_stack.in_layers.0.bias": "blocks.29.norm1.bias",
            "first_stage_model.decoder.up.0.block.1.time_stack.in_layers.0.weight": "blocks.29.norm1.weight",
            "first_stage_model.decoder.up.0.block.1.time_stack.in_layers.2.bias": "blocks.29.conv1.bias",
            "first_stage_model.decoder.up.0.block.1.time_stack.in_layers.2.weight": "blocks.29.conv1.weight",
            "first_stage_model.decoder.up.0.block.1.time_stack.out_layers.0.bias": "blocks.29.norm2.bias",
            "first_stage_model.decoder.up.0.block.1.time_stack.out_layers.0.weight": "blocks.29.norm2.weight",
            "first_stage_model.decoder.up.0.block.1.time_stack.out_layers.3.bias": "blocks.29.conv2.bias",
            "first_stage_model.decoder.up.0.block.1.time_stack.out_layers.3.weight": "blocks.29.conv2.weight",
            "first_stage_model.decoder.up.0.block.2.conv1.bias": "blocks.30.conv1.bias",
            "first_stage_model.decoder.up.0.block.2.conv1.weight": "blocks.30.conv1.weight",
            "first_stage_model.decoder.up.0.block.2.conv2.bias": "blocks.30.conv2.bias",
            "first_stage_model.decoder.up.0.block.2.conv2.weight": "blocks.30.conv2.weight",
            "first_stage_model.decoder.up.0.block.2.mix_factor": "blocks.31.mix_factor",
            "first_stage_model.decoder.up.0.block.2.norm1.bias": "blocks.30.norm1.bias",
            "first_stage_model.decoder.up.0.block.2.norm1.weight": "blocks.30.norm1.weight",
            "first_stage_model.decoder.up.0.block.2.norm2.bias": "blocks.30.norm2.bias",
            "first_stage_model.decoder.up.0.block.2.norm2.weight": "blocks.30.norm2.weight",
            "first_stage_model.decoder.up.0.block.2.time_stack.in_layers.0.bias": "blocks.31.norm1.bias",
            "first_stage_model.decoder.up.0.block.2.time_stack.in_layers.0.weight": "blocks.31.norm1.weight",
            "first_stage_model.decoder.up.0.block.2.time_stack.in_layers.2.bias": "blocks.31.conv1.bias",
            "first_stage_model.decoder.up.0.block.2.time_stack.in_layers.2.weight": "blocks.31.conv1.weight",
            "first_stage_model.decoder.up.0.block.2.time_stack.out_layers.0.bias": "blocks.31.norm2.bias",
            "first_stage_model.decoder.up.0.block.2.time_stack.out_layers.0.weight": "blocks.31.norm2.weight",
            "first_stage_model.decoder.up.0.block.2.time_stack.out_layers.3.bias": "blocks.31.conv2.bias",
            "first_stage_model.decoder.up.0.block.2.time_stack.out_layers.3.weight": "blocks.31.conv2.weight",
            "first_stage_model.decoder.up.1.block.0.conv1.bias": "blocks.19.conv1.bias",
            "first_stage_model.decoder.up.1.block.0.conv1.weight": "blocks.19.conv1.weight",
            "first_stage_model.decoder.up.1.block.0.conv2.bias": "blocks.19.conv2.bias",
            "first_stage_model.decoder.up.1.block.0.conv2.weight": "blocks.19.conv2.weight",
            "first_stage_model.decoder.up.1.block.0.mix_factor": "blocks.20.mix_factor",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.bias": "blocks.19.conv_shortcut.bias",
            "first_stage_model.decoder.up.1.block.0.nin_shortcut.weight": "blocks.19.conv_shortcut.weight",
            "first_stage_model.decoder.up.1.block.0.norm1.bias": "blocks.19.norm1.bias",
            "first_stage_model.decoder.up.1.block.0.norm1.weight": "blocks.19.norm1.weight",
            "first_stage_model.decoder.up.1.block.0.norm2.bias": "blocks.19.norm2.bias",
            "first_stage_model.decoder.up.1.block.0.norm2.weight": "blocks.19.norm2.weight",
            "first_stage_model.decoder.up.1.block.0.time_stack.in_layers.0.bias": "blocks.20.norm1.bias",
            "first_stage_model.decoder.up.1.block.0.time_stack.in_layers.0.weight": "blocks.20.norm1.weight",
            "first_stage_model.decoder.up.1.block.0.time_stack.in_layers.2.bias": "blocks.20.conv1.bias",
            "first_stage_model.decoder.up.1.block.0.time_stack.in_layers.2.weight": "blocks.20.conv1.weight",
            "first_stage_model.decoder.up.1.block.0.time_stack.out_layers.0.bias": "blocks.20.norm2.bias",
            "first_stage_model.decoder.up.1.block.0.time_stack.out_layers.0.weight": "blocks.20.norm2.weight",
            "first_stage_model.decoder.up.1.block.0.time_stack.out_layers.3.bias": "blocks.20.conv2.bias",
            "first_stage_model.decoder.up.1.block.0.time_stack.out_layers.3.weight": "blocks.20.conv2.weight",
            "first_stage_model.decoder.up.1.block.1.conv1.bias": "blocks.21.conv1.bias",
            "first_stage_model.decoder.up.1.block.1.conv1.weight": "blocks.21.conv1.weight",
            "first_stage_model.decoder.up.1.block.1.conv2.bias": "blocks.21.conv2.bias",
            "first_stage_model.decoder.up.1.block.1.conv2.weight": "blocks.21.conv2.weight",
            "first_stage_model.decoder.up.1.block.1.mix_factor": "blocks.22.mix_factor",
            "first_stage_model.decoder.up.1.block.1.norm1.bias": "blocks.21.norm1.bias",
            "first_stage_model.decoder.up.1.block.1.norm1.weight": "blocks.21.norm1.weight",
            "first_stage_model.decoder.up.1.block.1.norm2.bias": "blocks.21.norm2.bias",
            "first_stage_model.decoder.up.1.block.1.norm2.weight": "blocks.21.norm2.weight",
            "first_stage_model.decoder.up.1.block.1.time_stack.in_layers.0.bias": "blocks.22.norm1.bias",
            "first_stage_model.decoder.up.1.block.1.time_stack.in_layers.0.weight": "blocks.22.norm1.weight",
            "first_stage_model.decoder.up.1.block.1.time_stack.in_layers.2.bias": "blocks.22.conv1.bias",
            "first_stage_model.decoder.up.1.block.1.time_stack.in_layers.2.weight": "blocks.22.conv1.weight",
            "first_stage_model.decoder.up.1.block.1.time_stack.out_layers.0.bias": "blocks.22.norm2.bias",
            "first_stage_model.decoder.up.1.block.1.time_stack.out_layers.0.weight": "blocks.22.norm2.weight",
            "first_stage_model.decoder.up.1.block.1.time_stack.out_layers.3.bias": "blocks.22.conv2.bias",
            "first_stage_model.decoder.up.1.block.1.time_stack.out_layers.3.weight": "blocks.22.conv2.weight",
            "first_stage_model.decoder.up.1.block.2.conv1.bias": "blocks.23.conv1.bias",
            "first_stage_model.decoder.up.1.block.2.conv1.weight": "blocks.23.conv1.weight",
            "first_stage_model.decoder.up.1.block.2.conv2.bias": "blocks.23.conv2.bias",
            "first_stage_model.decoder.up.1.block.2.conv2.weight": "blocks.23.conv2.weight",
            "first_stage_model.decoder.up.1.block.2.mix_factor": "blocks.24.mix_factor",
            "first_stage_model.decoder.up.1.block.2.norm1.bias": "blocks.23.norm1.bias",
            "first_stage_model.decoder.up.1.block.2.norm1.weight": "blocks.23.norm1.weight",
            "first_stage_model.decoder.up.1.block.2.norm2.bias": "blocks.23.norm2.bias",
            "first_stage_model.decoder.up.1.block.2.norm2.weight": "blocks.23.norm2.weight",
            "first_stage_model.decoder.up.1.block.2.time_stack.in_layers.0.bias": "blocks.24.norm1.bias",
            "first_stage_model.decoder.up.1.block.2.time_stack.in_layers.0.weight": "blocks.24.norm1.weight",
            "first_stage_model.decoder.up.1.block.2.time_stack.in_layers.2.bias": "blocks.24.conv1.bias",
            "first_stage_model.decoder.up.1.block.2.time_stack.in_layers.2.weight": "blocks.24.conv1.weight",
            "first_stage_model.decoder.up.1.block.2.time_stack.out_layers.0.bias": "blocks.24.norm2.bias",
            "first_stage_model.decoder.up.1.block.2.time_stack.out_layers.0.weight": "blocks.24.norm2.weight",
            "first_stage_model.decoder.up.1.block.2.time_stack.out_layers.3.bias": "blocks.24.conv2.bias",
            "first_stage_model.decoder.up.1.block.2.time_stack.out_layers.3.weight": "blocks.24.conv2.weight",
            "first_stage_model.decoder.up.1.upsample.conv.bias": "blocks.25.conv.bias",
            "first_stage_model.decoder.up.1.upsample.conv.weight": "blocks.25.conv.weight",
            "first_stage_model.decoder.up.2.block.0.conv1.bias": "blocks.12.conv1.bias",
            "first_stage_model.decoder.up.2.block.0.conv1.weight": "blocks.12.conv1.weight",
            "first_stage_model.decoder.up.2.block.0.conv2.bias": "blocks.12.conv2.bias",
            "first_stage_model.decoder.up.2.block.0.conv2.weight": "blocks.12.conv2.weight",
            "first_stage_model.decoder.up.2.block.0.mix_factor": "blocks.13.mix_factor",
            "first_stage_model.decoder.up.2.block.0.norm1.bias": "blocks.12.norm1.bias",
            "first_stage_model.decoder.up.2.block.0.norm1.weight": "blocks.12.norm1.weight",
            "first_stage_model.decoder.up.2.block.0.norm2.bias": "blocks.12.norm2.bias",
            "first_stage_model.decoder.up.2.block.0.norm2.weight": "blocks.12.norm2.weight",
            "first_stage_model.decoder.up.2.block.0.time_stack.in_layers.0.bias": "blocks.13.norm1.bias",
            "first_stage_model.decoder.up.2.block.0.time_stack.in_layers.0.weight": "blocks.13.norm1.weight",
            "first_stage_model.decoder.up.2.block.0.time_stack.in_layers.2.bias": "blocks.13.conv1.bias",
            "first_stage_model.decoder.up.2.block.0.time_stack.in_layers.2.weight": "blocks.13.conv1.weight",
            "first_stage_model.decoder.up.2.block.0.time_stack.out_layers.0.bias": "blocks.13.norm2.bias",
            "first_stage_model.decoder.up.2.block.0.time_stack.out_layers.0.weight": "blocks.13.norm2.weight",
            "first_stage_model.decoder.up.2.block.0.time_stack.out_layers.3.bias": "blocks.13.conv2.bias",
            "first_stage_model.decoder.up.2.block.0.time_stack.out_layers.3.weight": "blocks.13.conv2.weight",
            "first_stage_model.decoder.up.2.block.1.conv1.bias": "blocks.14.conv1.bias",
            "first_stage_model.decoder.up.2.block.1.conv1.weight": "blocks.14.conv1.weight",
            "first_stage_model.decoder.up.2.block.1.conv2.bias": "blocks.14.conv2.bias",
            "first_stage_model.decoder.up.2.block.1.conv2.weight": "blocks.14.conv2.weight",
            "first_stage_model.decoder.up.2.block.1.mix_factor": "blocks.15.mix_factor",
            "first_stage_model.decoder.up.2.block.1.norm1.bias": "blocks.14.norm1.bias",
            "first_stage_model.decoder.up.2.block.1.norm1.weight": "blocks.14.norm1.weight",
            "first_stage_model.decoder.up.2.block.1.norm2.bias": "blocks.14.norm2.bias",
            "first_stage_model.decoder.up.2.block.1.norm2.weight": "blocks.14.norm2.weight",
            "first_stage_model.decoder.up.2.block.1.time_stack.in_layers.0.bias": "blocks.15.norm1.bias",
            "first_stage_model.decoder.up.2.block.1.time_stack.in_layers.0.weight": "blocks.15.norm1.weight",
            "first_stage_model.decoder.up.2.block.1.time_stack.in_layers.2.bias": "blocks.15.conv1.bias",
            "first_stage_model.decoder.up.2.block.1.time_stack.in_layers.2.weight": "blocks.15.conv1.weight",
            "first_stage_model.decoder.up.2.block.1.time_stack.out_layers.0.bias": "blocks.15.norm2.bias",
            "first_stage_model.decoder.up.2.block.1.time_stack.out_layers.0.weight": "blocks.15.norm2.weight",
            "first_stage_model.decoder.up.2.block.1.time_stack.out_layers.3.bias": "blocks.15.conv2.bias",
            "first_stage_model.decoder.up.2.block.1.time_stack.out_layers.3.weight": "blocks.15.conv2.weight",
            "first_stage_model.decoder.up.2.block.2.conv1.bias": "blocks.16.conv1.bias",
            "first_stage_model.decoder.up.2.block.2.conv1.weight": "blocks.16.conv1.weight",
            "first_stage_model.decoder.up.2.block.2.conv2.bias": "blocks.16.conv2.bias",
            "first_stage_model.decoder.up.2.block.2.conv2.weight": "blocks.16.conv2.weight",
            "first_stage_model.decoder.up.2.block.2.mix_factor": "blocks.17.mix_factor",
            "first_stage_model.decoder.up.2.block.2.norm1.bias": "blocks.16.norm1.bias",
            "first_stage_model.decoder.up.2.block.2.norm1.weight": "blocks.16.norm1.weight",
            "first_stage_model.decoder.up.2.block.2.norm2.bias": "blocks.16.norm2.bias",
            "first_stage_model.decoder.up.2.block.2.norm2.weight": "blocks.16.norm2.weight",
            "first_stage_model.decoder.up.2.block.2.time_stack.in_layers.0.bias": "blocks.17.norm1.bias",
            "first_stage_model.decoder.up.2.block.2.time_stack.in_layers.0.weight": "blocks.17.norm1.weight",
            "first_stage_model.decoder.up.2.block.2.time_stack.in_layers.2.bias": "blocks.17.conv1.bias",
            "first_stage_model.decoder.up.2.block.2.time_stack.in_layers.2.weight": "blocks.17.conv1.weight",
            "first_stage_model.decoder.up.2.block.2.time_stack.out_layers.0.bias": "blocks.17.norm2.bias",
            "first_stage_model.decoder.up.2.block.2.time_stack.out_layers.0.weight": "blocks.17.norm2.weight",
            "first_stage_model.decoder.up.2.block.2.time_stack.out_layers.3.bias": "blocks.17.conv2.bias",
            "first_stage_model.decoder.up.2.block.2.time_stack.out_layers.3.weight": "blocks.17.conv2.weight",
            "first_stage_model.decoder.up.2.upsample.conv.bias": "blocks.18.conv.bias",
            "first_stage_model.decoder.up.2.upsample.conv.weight": "blocks.18.conv.weight",
            "first_stage_model.decoder.up.3.block.0.conv1.bias": "blocks.5.conv1.bias",
            "first_stage_model.decoder.up.3.block.0.conv1.weight": "blocks.5.conv1.weight",
            "first_stage_model.decoder.up.3.block.0.conv2.bias": "blocks.5.conv2.bias",
            "first_stage_model.decoder.up.3.block.0.conv2.weight": "blocks.5.conv2.weight",
            "first_stage_model.decoder.up.3.block.0.mix_factor": "blocks.6.mix_factor",
            "first_stage_model.decoder.up.3.block.0.norm1.bias": "blocks.5.norm1.bias",
            "first_stage_model.decoder.up.3.block.0.norm1.weight": "blocks.5.norm1.weight",
            "first_stage_model.decoder.up.3.block.0.norm2.bias": "blocks.5.norm2.bias",
            "first_stage_model.decoder.up.3.block.0.norm2.weight": "blocks.5.norm2.weight",
            "first_stage_model.decoder.up.3.block.0.time_stack.in_layers.0.bias": "blocks.6.norm1.bias",
            "first_stage_model.decoder.up.3.block.0.time_stack.in_layers.0.weight": "blocks.6.norm1.weight",
            "first_stage_model.decoder.up.3.block.0.time_stack.in_layers.2.bias": "blocks.6.conv1.bias",
            "first_stage_model.decoder.up.3.block.0.time_stack.in_layers.2.weight": "blocks.6.conv1.weight",
            "first_stage_model.decoder.up.3.block.0.time_stack.out_layers.0.bias": "blocks.6.norm2.bias",
            "first_stage_model.decoder.up.3.block.0.time_stack.out_layers.0.weight": "blocks.6.norm2.weight",
            "first_stage_model.decoder.up.3.block.0.time_stack.out_layers.3.bias": "blocks.6.conv2.bias",
            "first_stage_model.decoder.up.3.block.0.time_stack.out_layers.3.weight": "blocks.6.conv2.weight",
            "first_stage_model.decoder.up.3.block.1.conv1.bias": "blocks.7.conv1.bias",
            "first_stage_model.decoder.up.3.block.1.conv1.weight": "blocks.7.conv1.weight",
            "first_stage_model.decoder.up.3.block.1.conv2.bias": "blocks.7.conv2.bias",
            "first_stage_model.decoder.up.3.block.1.conv2.weight": "blocks.7.conv2.weight",
            "first_stage_model.decoder.up.3.block.1.mix_factor": "blocks.8.mix_factor",
            "first_stage_model.decoder.up.3.block.1.norm1.bias": "blocks.7.norm1.bias",
            "first_stage_model.decoder.up.3.block.1.norm1.weight": "blocks.7.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.norm2.bias": "blocks.7.norm2.bias",
            "first_stage_model.decoder.up.3.block.1.norm2.weight": "blocks.7.norm2.weight",
            "first_stage_model.decoder.up.3.block.1.time_stack.in_layers.0.bias": "blocks.8.norm1.bias",
            "first_stage_model.decoder.up.3.block.1.time_stack.in_layers.0.weight": "blocks.8.norm1.weight",
            "first_stage_model.decoder.up.3.block.1.time_stack.in_layers.2.bias": "blocks.8.conv1.bias",
            "first_stage_model.decoder.up.3.block.1.time_stack.in_layers.2.weight": "blocks.8.conv1.weight",
            "first_stage_model.decoder.up.3.block.1.time_stack.out_layers.0.bias": "blocks.8.norm2.bias",
            "first_stage_model.decoder.up.3.block.1.time_stack.out_layers.0.weight": "blocks.8.norm2.weight",
            "first_stage_model.decoder.up.3.block.1.time_stack.out_layers.3.bias": "blocks.8.conv2.bias",
            "first_stage_model.decoder.up.3.block.1.time_stack.out_layers.3.weight": "blocks.8.conv2.weight",
            "first_stage_model.decoder.up.3.block.2.conv1.bias": "blocks.9.conv1.bias",
            "first_stage_model.decoder.up.3.block.2.conv1.weight": "blocks.9.conv1.weight",
            "first_stage_model.decoder.up.3.block.2.conv2.bias": "blocks.9.conv2.bias",
            "first_stage_model.decoder.up.3.block.2.conv2.weight": "blocks.9.conv2.weight",
            "first_stage_model.decoder.up.3.block.2.mix_factor": "blocks.10.mix_factor",
            "first_stage_model.decoder.up.3.block.2.norm1.bias": "blocks.9.norm1.bias",
            "first_stage_model.decoder.up.3.block.2.norm1.weight": "blocks.9.norm1.weight",
            "first_stage_model.decoder.up.3.block.2.norm2.bias": "blocks.9.norm2.bias",
            "first_stage_model.decoder.up.3.block.2.norm2.weight": "blocks.9.norm2.weight",
            "first_stage_model.decoder.up.3.block.2.time_stack.in_layers.0.bias": "blocks.10.norm1.bias",
            "first_stage_model.decoder.up.3.block.2.time_stack.in_layers.0.weight": "blocks.10.norm1.weight",
            "first_stage_model.decoder.up.3.block.2.time_stack.in_layers.2.bias": "blocks.10.conv1.bias",
            "first_stage_model.decoder.up.3.block.2.time_stack.in_layers.2.weight": "blocks.10.conv1.weight",
            "first_stage_model.decoder.up.3.block.2.time_stack.out_layers.0.bias": "blocks.10.norm2.bias",
            "first_stage_model.decoder.up.3.block.2.time_stack.out_layers.0.weight": "blocks.10.norm2.weight",
            "first_stage_model.decoder.up.3.block.2.time_stack.out_layers.3.bias": "blocks.10.conv2.bias",
            "first_stage_model.decoder.up.3.block.2.time_stack.out_layers.3.weight": "blocks.10.conv2.weight",
            "first_stage_model.decoder.up.3.upsample.conv.bias": "blocks.11.conv.bias",
            "first_stage_model.decoder.up.3.upsample.conv.weight": "blocks.11.conv.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "blocks.2.transformer_blocks.0" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_
