from .sd_vae_encoder import SDVAEEncoderStateDictConverter, SDVAEEncoder


class SVDVAEEncoder(SDVAEEncoder):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.13025
    
    @staticmethod
    def state_dict_converter():
        return SVDVAEEncoderStateDictConverter()


class SVDVAEEncoderStateDictConverter(SDVAEEncoderStateDictConverter):
    def __init__(self):
        super().__init__()

    def from_diffusers(self, state_dict):
        return super().from_diffusers(state_dict)
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "conditioner.embedders.3.encoder.encoder.conv_in.bias": "conv_in.bias",
            "conditioner.embedders.3.encoder.encoder.conv_in.weight": "conv_in.weight",
            "conditioner.embedders.3.encoder.encoder.conv_out.bias": "conv_out.bias",
            "conditioner.embedders.3.encoder.encoder.conv_out.weight": "conv_out.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.conv1.bias": "blocks.0.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.conv1.weight": "blocks.0.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.conv2.bias": "blocks.0.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.conv2.weight": "blocks.0.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.norm1.bias": "blocks.0.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.norm1.weight": "blocks.0.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.norm2.bias": "blocks.0.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.0.norm2.weight": "blocks.0.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.conv1.bias": "blocks.1.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.conv1.weight": "blocks.1.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.conv2.bias": "blocks.1.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.conv2.weight": "blocks.1.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.norm1.bias": "blocks.1.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.norm1.weight": "blocks.1.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.norm2.bias": "blocks.1.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.block.1.norm2.weight": "blocks.1.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.0.downsample.conv.bias": "blocks.2.conv.bias",
            "conditioner.embedders.3.encoder.encoder.down.0.downsample.conv.weight": "blocks.2.conv.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.conv1.bias": "blocks.3.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.conv1.weight": "blocks.3.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.conv2.bias": "blocks.3.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.conv2.weight": "blocks.3.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.nin_shortcut.bias": "blocks.3.conv_shortcut.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.nin_shortcut.weight": "blocks.3.conv_shortcut.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.norm1.bias": "blocks.3.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.norm1.weight": "blocks.3.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.norm2.bias": "blocks.3.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.0.norm2.weight": "blocks.3.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.conv1.bias": "blocks.4.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.conv1.weight": "blocks.4.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.conv2.bias": "blocks.4.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.conv2.weight": "blocks.4.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.norm1.bias": "blocks.4.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.norm1.weight": "blocks.4.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.norm2.bias": "blocks.4.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.block.1.norm2.weight": "blocks.4.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.1.downsample.conv.bias": "blocks.5.conv.bias",
            "conditioner.embedders.3.encoder.encoder.down.1.downsample.conv.weight": "blocks.5.conv.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.conv1.bias": "blocks.6.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.conv1.weight": "blocks.6.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.conv2.bias": "blocks.6.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.conv2.weight": "blocks.6.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.nin_shortcut.bias": "blocks.6.conv_shortcut.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.nin_shortcut.weight": "blocks.6.conv_shortcut.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.norm1.bias": "blocks.6.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.norm1.weight": "blocks.6.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.norm2.bias": "blocks.6.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.0.norm2.weight": "blocks.6.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.conv1.bias": "blocks.7.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.conv1.weight": "blocks.7.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.conv2.bias": "blocks.7.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.conv2.weight": "blocks.7.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.norm1.bias": "blocks.7.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.norm1.weight": "blocks.7.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.norm2.bias": "blocks.7.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.block.1.norm2.weight": "blocks.7.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.2.downsample.conv.bias": "blocks.8.conv.bias",
            "conditioner.embedders.3.encoder.encoder.down.2.downsample.conv.weight": "blocks.8.conv.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.conv1.bias": "blocks.9.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.conv1.weight": "blocks.9.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.conv2.bias": "blocks.9.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.conv2.weight": "blocks.9.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.norm1.bias": "blocks.9.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.norm1.weight": "blocks.9.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.norm2.bias": "blocks.9.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.0.norm2.weight": "blocks.9.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.conv1.bias": "blocks.10.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.conv1.weight": "blocks.10.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.conv2.bias": "blocks.10.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.conv2.weight": "blocks.10.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.norm1.bias": "blocks.10.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.norm1.weight": "blocks.10.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.norm2.bias": "blocks.10.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.down.3.block.1.norm2.weight": "blocks.10.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.k.bias": "blocks.12.transformer_blocks.0.to_k.bias",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.k.weight": "blocks.12.transformer_blocks.0.to_k.weight",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.norm.bias": "blocks.12.norm.bias",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.norm.weight": "blocks.12.norm.weight",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.proj_out.bias": "blocks.12.transformer_blocks.0.to_out.bias",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.proj_out.weight": "blocks.12.transformer_blocks.0.to_out.weight",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.q.bias": "blocks.12.transformer_blocks.0.to_q.bias",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.q.weight": "blocks.12.transformer_blocks.0.to_q.weight",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.v.bias": "blocks.12.transformer_blocks.0.to_v.bias",
            "conditioner.embedders.3.encoder.encoder.mid.attn_1.v.weight": "blocks.12.transformer_blocks.0.to_v.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.conv1.bias": "blocks.11.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.conv1.weight": "blocks.11.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.conv2.bias": "blocks.11.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.conv2.weight": "blocks.11.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.norm1.bias": "blocks.11.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.norm1.weight": "blocks.11.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.norm2.bias": "blocks.11.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_1.norm2.weight": "blocks.11.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.conv1.bias": "blocks.13.conv1.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.conv1.weight": "blocks.13.conv1.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.conv2.bias": "blocks.13.conv2.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.conv2.weight": "blocks.13.conv2.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.norm1.bias": "blocks.13.norm1.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.norm1.weight": "blocks.13.norm1.weight",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.norm2.bias": "blocks.13.norm2.bias",
            "conditioner.embedders.3.encoder.encoder.mid.block_2.norm2.weight": "blocks.13.norm2.weight",
            "conditioner.embedders.3.encoder.encoder.norm_out.bias": "conv_norm_out.bias",
            "conditioner.embedders.3.encoder.encoder.norm_out.weight": "conv_norm_out.weight",
            "conditioner.embedders.3.encoder.quant_conv.bias": "quant_conv.bias",
            "conditioner.embedders.3.encoder.quant_conv.weight": "quant_conv.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if "transformer_blocks" in rename_dict[name]:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_
