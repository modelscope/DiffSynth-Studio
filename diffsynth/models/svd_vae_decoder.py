import torch
from .attention import Attention
from .sd_unet import ResnetBlock, UpSampler
from .tiler import TileWorker
from einops import rearrange


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
        hidden_states = sample.flatten(0, 1)
        hidden_states = hidden_states / self.scaling_factor
        hidden_states = self.conv_in(hidden_states)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = rearrange(hidden_states, "T C H W -> 1 C T H W")
        hidden_states = self.time_conv_out(hidden_states)

        return hidden_states
    
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
