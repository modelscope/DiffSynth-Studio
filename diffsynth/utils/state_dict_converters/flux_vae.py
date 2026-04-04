def FluxVAEEncoderStateDictConverter(state_dict):
    rename_dict = {
        "encoder.conv_in.bias": "conv_in.bias",
        "encoder.conv_in.weight": "conv_in.weight",
        "encoder.conv_out.bias": "conv_out.bias",
        "encoder.conv_out.weight": "conv_out.weight",
        "encoder.down.0.block.0.conv1.bias": "blocks.0.conv1.bias",
        "encoder.down.0.block.0.conv1.weight": "blocks.0.conv1.weight",
        "encoder.down.0.block.0.conv2.bias": "blocks.0.conv2.bias",
        "encoder.down.0.block.0.conv2.weight": "blocks.0.conv2.weight",
        "encoder.down.0.block.0.norm1.bias": "blocks.0.norm1.bias",
        "encoder.down.0.block.0.norm1.weight": "blocks.0.norm1.weight",
        "encoder.down.0.block.0.norm2.bias": "blocks.0.norm2.bias",
        "encoder.down.0.block.0.norm2.weight": "blocks.0.norm2.weight",
        "encoder.down.0.block.1.conv1.bias": "blocks.1.conv1.bias",
        "encoder.down.0.block.1.conv1.weight": "blocks.1.conv1.weight",
        "encoder.down.0.block.1.conv2.bias": "blocks.1.conv2.bias",
        "encoder.down.0.block.1.conv2.weight": "blocks.1.conv2.weight",
        "encoder.down.0.block.1.norm1.bias": "blocks.1.norm1.bias",
        "encoder.down.0.block.1.norm1.weight": "blocks.1.norm1.weight",
        "encoder.down.0.block.1.norm2.bias": "blocks.1.norm2.bias",
        "encoder.down.0.block.1.norm2.weight": "blocks.1.norm2.weight",
        "encoder.down.0.downsample.conv.bias": "blocks.2.conv.bias",
        "encoder.down.0.downsample.conv.weight": "blocks.2.conv.weight",
        "encoder.down.1.block.0.conv1.bias": "blocks.3.conv1.bias",
        "encoder.down.1.block.0.conv1.weight": "blocks.3.conv1.weight",
        "encoder.down.1.block.0.conv2.bias": "blocks.3.conv2.bias",
        "encoder.down.1.block.0.conv2.weight": "blocks.3.conv2.weight",
        "encoder.down.1.block.0.nin_shortcut.bias": "blocks.3.conv_shortcut.bias",
        "encoder.down.1.block.0.nin_shortcut.weight": "blocks.3.conv_shortcut.weight",
        "encoder.down.1.block.0.norm1.bias": "blocks.3.norm1.bias",
        "encoder.down.1.block.0.norm1.weight": "blocks.3.norm1.weight",
        "encoder.down.1.block.0.norm2.bias": "blocks.3.norm2.bias",
        "encoder.down.1.block.0.norm2.weight": "blocks.3.norm2.weight",
        "encoder.down.1.block.1.conv1.bias": "blocks.4.conv1.bias",
        "encoder.down.1.block.1.conv1.weight": "blocks.4.conv1.weight",
        "encoder.down.1.block.1.conv2.bias": "blocks.4.conv2.bias",
        "encoder.down.1.block.1.conv2.weight": "blocks.4.conv2.weight",
        "encoder.down.1.block.1.norm1.bias": "blocks.4.norm1.bias",
        "encoder.down.1.block.1.norm1.weight": "blocks.4.norm1.weight",
        "encoder.down.1.block.1.norm2.bias": "blocks.4.norm2.bias",
        "encoder.down.1.block.1.norm2.weight": "blocks.4.norm2.weight",
        "encoder.down.1.downsample.conv.bias": "blocks.5.conv.bias",
        "encoder.down.1.downsample.conv.weight": "blocks.5.conv.weight",
        "encoder.down.2.block.0.conv1.bias": "blocks.6.conv1.bias",
        "encoder.down.2.block.0.conv1.weight": "blocks.6.conv1.weight",
        "encoder.down.2.block.0.conv2.bias": "blocks.6.conv2.bias",
        "encoder.down.2.block.0.conv2.weight": "blocks.6.conv2.weight",
        "encoder.down.2.block.0.nin_shortcut.bias": "blocks.6.conv_shortcut.bias",
        "encoder.down.2.block.0.nin_shortcut.weight": "blocks.6.conv_shortcut.weight",
        "encoder.down.2.block.0.norm1.bias": "blocks.6.norm1.bias",
        "encoder.down.2.block.0.norm1.weight": "blocks.6.norm1.weight",
        "encoder.down.2.block.0.norm2.bias": "blocks.6.norm2.bias",
        "encoder.down.2.block.0.norm2.weight": "blocks.6.norm2.weight",
        "encoder.down.2.block.1.conv1.bias": "blocks.7.conv1.bias",
        "encoder.down.2.block.1.conv1.weight": "blocks.7.conv1.weight",
        "encoder.down.2.block.1.conv2.bias": "blocks.7.conv2.bias",
        "encoder.down.2.block.1.conv2.weight": "blocks.7.conv2.weight",
        "encoder.down.2.block.1.norm1.bias": "blocks.7.norm1.bias",
        "encoder.down.2.block.1.norm1.weight": "blocks.7.norm1.weight",
        "encoder.down.2.block.1.norm2.bias": "blocks.7.norm2.bias",
        "encoder.down.2.block.1.norm2.weight": "blocks.7.norm2.weight",
        "encoder.down.2.downsample.conv.bias": "blocks.8.conv.bias",
        "encoder.down.2.downsample.conv.weight": "blocks.8.conv.weight",
        "encoder.down.3.block.0.conv1.bias": "blocks.9.conv1.bias",
        "encoder.down.3.block.0.conv1.weight": "blocks.9.conv1.weight",
        "encoder.down.3.block.0.conv2.bias": "blocks.9.conv2.bias",
        "encoder.down.3.block.0.conv2.weight": "blocks.9.conv2.weight",
        "encoder.down.3.block.0.norm1.bias": "blocks.9.norm1.bias",
        "encoder.down.3.block.0.norm1.weight": "blocks.9.norm1.weight",
        "encoder.down.3.block.0.norm2.bias": "blocks.9.norm2.bias",
        "encoder.down.3.block.0.norm2.weight": "blocks.9.norm2.weight",
        "encoder.down.3.block.1.conv1.bias": "blocks.10.conv1.bias",
        "encoder.down.3.block.1.conv1.weight": "blocks.10.conv1.weight",
        "encoder.down.3.block.1.conv2.bias": "blocks.10.conv2.bias",
        "encoder.down.3.block.1.conv2.weight": "blocks.10.conv2.weight",
        "encoder.down.3.block.1.norm1.bias": "blocks.10.norm1.bias",
        "encoder.down.3.block.1.norm1.weight": "blocks.10.norm1.weight",
        "encoder.down.3.block.1.norm2.bias": "blocks.10.norm2.bias",
        "encoder.down.3.block.1.norm2.weight": "blocks.10.norm2.weight",
        "encoder.mid.attn_1.k.bias": "blocks.12.transformer_blocks.0.to_k.bias",
        "encoder.mid.attn_1.k.weight": "blocks.12.transformer_blocks.0.to_k.weight",
        "encoder.mid.attn_1.norm.bias": "blocks.12.norm.bias",
        "encoder.mid.attn_1.norm.weight": "blocks.12.norm.weight",
        "encoder.mid.attn_1.proj_out.bias": "blocks.12.transformer_blocks.0.to_out.bias",
        "encoder.mid.attn_1.proj_out.weight": "blocks.12.transformer_blocks.0.to_out.weight",
        "encoder.mid.attn_1.q.bias": "blocks.12.transformer_blocks.0.to_q.bias",
        "encoder.mid.attn_1.q.weight": "blocks.12.transformer_blocks.0.to_q.weight",
        "encoder.mid.attn_1.v.bias": "blocks.12.transformer_blocks.0.to_v.bias",
        "encoder.mid.attn_1.v.weight": "blocks.12.transformer_blocks.0.to_v.weight",
        "encoder.mid.block_1.conv1.bias": "blocks.11.conv1.bias",
        "encoder.mid.block_1.conv1.weight": "blocks.11.conv1.weight",
        "encoder.mid.block_1.conv2.bias": "blocks.11.conv2.bias",
        "encoder.mid.block_1.conv2.weight": "blocks.11.conv2.weight",
        "encoder.mid.block_1.norm1.bias": "blocks.11.norm1.bias",
        "encoder.mid.block_1.norm1.weight": "blocks.11.norm1.weight",
        "encoder.mid.block_1.norm2.bias": "blocks.11.norm2.bias",
        "encoder.mid.block_1.norm2.weight": "blocks.11.norm2.weight",
        "encoder.mid.block_2.conv1.bias": "blocks.13.conv1.bias",
        "encoder.mid.block_2.conv1.weight": "blocks.13.conv1.weight",
        "encoder.mid.block_2.conv2.bias": "blocks.13.conv2.bias",
        "encoder.mid.block_2.conv2.weight": "blocks.13.conv2.weight",
        "encoder.mid.block_2.norm1.bias": "blocks.13.norm1.bias",
        "encoder.mid.block_2.norm1.weight": "blocks.13.norm1.weight",
        "encoder.mid.block_2.norm2.bias": "blocks.13.norm2.bias",
        "encoder.mid.block_2.norm2.weight": "blocks.13.norm2.weight",
        "encoder.norm_out.bias": "conv_norm_out.bias",
        "encoder.norm_out.weight": "conv_norm_out.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            param = state_dict[name]
            state_dict_[rename_dict[name]] = param
    return state_dict_


def FluxVAEDecoderStateDictConverter(state_dict):
    rename_dict = {
        "decoder.conv_in.bias": "conv_in.bias",
        "decoder.conv_in.weight": "conv_in.weight",
        "decoder.conv_out.bias": "conv_out.bias",
        "decoder.conv_out.weight": "conv_out.weight",
        "decoder.mid.attn_1.k.bias": "blocks.1.transformer_blocks.0.to_k.bias",
        "decoder.mid.attn_1.k.weight": "blocks.1.transformer_blocks.0.to_k.weight",
        "decoder.mid.attn_1.norm.bias": "blocks.1.norm.bias",
        "decoder.mid.attn_1.norm.weight": "blocks.1.norm.weight",
        "decoder.mid.attn_1.proj_out.bias": "blocks.1.transformer_blocks.0.to_out.bias",
        "decoder.mid.attn_1.proj_out.weight": "blocks.1.transformer_blocks.0.to_out.weight",
        "decoder.mid.attn_1.q.bias": "blocks.1.transformer_blocks.0.to_q.bias",
        "decoder.mid.attn_1.q.weight": "blocks.1.transformer_blocks.0.to_q.weight",
        "decoder.mid.attn_1.v.bias": "blocks.1.transformer_blocks.0.to_v.bias",
        "decoder.mid.attn_1.v.weight": "blocks.1.transformer_blocks.0.to_v.weight",
        "decoder.mid.block_1.conv1.bias": "blocks.0.conv1.bias",
        "decoder.mid.block_1.conv1.weight": "blocks.0.conv1.weight",
        "decoder.mid.block_1.conv2.bias": "blocks.0.conv2.bias",
        "decoder.mid.block_1.conv2.weight": "blocks.0.conv2.weight",
        "decoder.mid.block_1.norm1.bias": "blocks.0.norm1.bias",
        "decoder.mid.block_1.norm1.weight": "blocks.0.norm1.weight",
        "decoder.mid.block_1.norm2.bias": "blocks.0.norm2.bias",
        "decoder.mid.block_1.norm2.weight": "blocks.0.norm2.weight",
        "decoder.mid.block_2.conv1.bias": "blocks.2.conv1.bias",
        "decoder.mid.block_2.conv1.weight": "blocks.2.conv1.weight",
        "decoder.mid.block_2.conv2.bias": "blocks.2.conv2.bias",
        "decoder.mid.block_2.conv2.weight": "blocks.2.conv2.weight",
        "decoder.mid.block_2.norm1.bias": "blocks.2.norm1.bias",
        "decoder.mid.block_2.norm1.weight": "blocks.2.norm1.weight",
        "decoder.mid.block_2.norm2.bias": "blocks.2.norm2.bias",
        "decoder.mid.block_2.norm2.weight": "blocks.2.norm2.weight",
        "decoder.norm_out.bias": "conv_norm_out.bias",
        "decoder.norm_out.weight": "conv_norm_out.weight",
        "decoder.up.0.block.0.conv1.bias": "blocks.15.conv1.bias",
        "decoder.up.0.block.0.conv1.weight": "blocks.15.conv1.weight",
        "decoder.up.0.block.0.conv2.bias": "blocks.15.conv2.bias",
        "decoder.up.0.block.0.conv2.weight": "blocks.15.conv2.weight",
        "decoder.up.0.block.0.nin_shortcut.bias": "blocks.15.conv_shortcut.bias",
        "decoder.up.0.block.0.nin_shortcut.weight": "blocks.15.conv_shortcut.weight",
        "decoder.up.0.block.0.norm1.bias": "blocks.15.norm1.bias",
        "decoder.up.0.block.0.norm1.weight": "blocks.15.norm1.weight",
        "decoder.up.0.block.0.norm2.bias": "blocks.15.norm2.bias",
        "decoder.up.0.block.0.norm2.weight": "blocks.15.norm2.weight",
        "decoder.up.0.block.1.conv1.bias": "blocks.16.conv1.bias",
        "decoder.up.0.block.1.conv1.weight": "blocks.16.conv1.weight",
        "decoder.up.0.block.1.conv2.bias": "blocks.16.conv2.bias",
        "decoder.up.0.block.1.conv2.weight": "blocks.16.conv2.weight",
        "decoder.up.0.block.1.norm1.bias": "blocks.16.norm1.bias",
        "decoder.up.0.block.1.norm1.weight": "blocks.16.norm1.weight",
        "decoder.up.0.block.1.norm2.bias": "blocks.16.norm2.bias",
        "decoder.up.0.block.1.norm2.weight": "blocks.16.norm2.weight",
        "decoder.up.0.block.2.conv1.bias": "blocks.17.conv1.bias",
        "decoder.up.0.block.2.conv1.weight": "blocks.17.conv1.weight",
        "decoder.up.0.block.2.conv2.bias": "blocks.17.conv2.bias",
        "decoder.up.0.block.2.conv2.weight": "blocks.17.conv2.weight",
        "decoder.up.0.block.2.norm1.bias": "blocks.17.norm1.bias",
        "decoder.up.0.block.2.norm1.weight": "blocks.17.norm1.weight",
        "decoder.up.0.block.2.norm2.bias": "blocks.17.norm2.bias",
        "decoder.up.0.block.2.norm2.weight": "blocks.17.norm2.weight",
        "decoder.up.1.block.0.conv1.bias": "blocks.11.conv1.bias",
        "decoder.up.1.block.0.conv1.weight": "blocks.11.conv1.weight",
        "decoder.up.1.block.0.conv2.bias": "blocks.11.conv2.bias",
        "decoder.up.1.block.0.conv2.weight": "blocks.11.conv2.weight",
        "decoder.up.1.block.0.nin_shortcut.bias": "blocks.11.conv_shortcut.bias",
        "decoder.up.1.block.0.nin_shortcut.weight": "blocks.11.conv_shortcut.weight",
        "decoder.up.1.block.0.norm1.bias": "blocks.11.norm1.bias",
        "decoder.up.1.block.0.norm1.weight": "blocks.11.norm1.weight",
        "decoder.up.1.block.0.norm2.bias": "blocks.11.norm2.bias",
        "decoder.up.1.block.0.norm2.weight": "blocks.11.norm2.weight",
        "decoder.up.1.block.1.conv1.bias": "blocks.12.conv1.bias",
        "decoder.up.1.block.1.conv1.weight": "blocks.12.conv1.weight",
        "decoder.up.1.block.1.conv2.bias": "blocks.12.conv2.bias",
        "decoder.up.1.block.1.conv2.weight": "blocks.12.conv2.weight",
        "decoder.up.1.block.1.norm1.bias": "blocks.12.norm1.bias",
        "decoder.up.1.block.1.norm1.weight": "blocks.12.norm1.weight",
        "decoder.up.1.block.1.norm2.bias": "blocks.12.norm2.bias",
        "decoder.up.1.block.1.norm2.weight": "blocks.12.norm2.weight",
        "decoder.up.1.block.2.conv1.bias": "blocks.13.conv1.bias",
        "decoder.up.1.block.2.conv1.weight": "blocks.13.conv1.weight",
        "decoder.up.1.block.2.conv2.bias": "blocks.13.conv2.bias",
        "decoder.up.1.block.2.conv2.weight": "blocks.13.conv2.weight",
        "decoder.up.1.block.2.norm1.bias": "blocks.13.norm1.bias",
        "decoder.up.1.block.2.norm1.weight": "blocks.13.norm1.weight",
        "decoder.up.1.block.2.norm2.bias": "blocks.13.norm2.bias",
        "decoder.up.1.block.2.norm2.weight": "blocks.13.norm2.weight",
        "decoder.up.1.upsample.conv.bias": "blocks.14.conv.bias",
        "decoder.up.1.upsample.conv.weight": "blocks.14.conv.weight",
        "decoder.up.2.block.0.conv1.bias": "blocks.7.conv1.bias",
        "decoder.up.2.block.0.conv1.weight": "blocks.7.conv1.weight",
        "decoder.up.2.block.0.conv2.bias": "blocks.7.conv2.bias",
        "decoder.up.2.block.0.conv2.weight": "blocks.7.conv2.weight",
        "decoder.up.2.block.0.norm1.bias": "blocks.7.norm1.bias",
        "decoder.up.2.block.0.norm1.weight": "blocks.7.norm1.weight",
        "decoder.up.2.block.0.norm2.bias": "blocks.7.norm2.bias",
        "decoder.up.2.block.0.norm2.weight": "blocks.7.norm2.weight",
        "decoder.up.2.block.1.conv1.bias": "blocks.8.conv1.bias",
        "decoder.up.2.block.1.conv1.weight": "blocks.8.conv1.weight",
        "decoder.up.2.block.1.conv2.bias": "blocks.8.conv2.bias",
        "decoder.up.2.block.1.conv2.weight": "blocks.8.conv2.weight",
        "decoder.up.2.block.1.norm1.bias": "blocks.8.norm1.bias",
        "decoder.up.2.block.1.norm1.weight": "blocks.8.norm1.weight",
        "decoder.up.2.block.1.norm2.bias": "blocks.8.norm2.bias",
        "decoder.up.2.block.1.norm2.weight": "blocks.8.norm2.weight",
        "decoder.up.2.block.2.conv1.bias": "blocks.9.conv1.bias",
        "decoder.up.2.block.2.conv1.weight": "blocks.9.conv1.weight",
        "decoder.up.2.block.2.conv2.bias": "blocks.9.conv2.bias",
        "decoder.up.2.block.2.conv2.weight": "blocks.9.conv2.weight",
        "decoder.up.2.block.2.norm1.bias": "blocks.9.norm1.bias",
        "decoder.up.2.block.2.norm1.weight": "blocks.9.norm1.weight",
        "decoder.up.2.block.2.norm2.bias": "blocks.9.norm2.bias",
        "decoder.up.2.block.2.norm2.weight": "blocks.9.norm2.weight",
        "decoder.up.2.upsample.conv.bias": "blocks.10.conv.bias",
        "decoder.up.2.upsample.conv.weight": "blocks.10.conv.weight",
        "decoder.up.3.block.0.conv1.bias": "blocks.3.conv1.bias",
        "decoder.up.3.block.0.conv1.weight": "blocks.3.conv1.weight",
        "decoder.up.3.block.0.conv2.bias": "blocks.3.conv2.bias",
        "decoder.up.3.block.0.conv2.weight": "blocks.3.conv2.weight",
        "decoder.up.3.block.0.norm1.bias": "blocks.3.norm1.bias",
        "decoder.up.3.block.0.norm1.weight": "blocks.3.norm1.weight",
        "decoder.up.3.block.0.norm2.bias": "blocks.3.norm2.bias",
        "decoder.up.3.block.0.norm2.weight": "blocks.3.norm2.weight",
        "decoder.up.3.block.1.conv1.bias": "blocks.4.conv1.bias",
        "decoder.up.3.block.1.conv1.weight": "blocks.4.conv1.weight",
        "decoder.up.3.block.1.conv2.bias": "blocks.4.conv2.bias",
        "decoder.up.3.block.1.conv2.weight": "blocks.4.conv2.weight",
        "decoder.up.3.block.1.norm1.bias": "blocks.4.norm1.bias",
        "decoder.up.3.block.1.norm1.weight": "blocks.4.norm1.weight",
        "decoder.up.3.block.1.norm2.bias": "blocks.4.norm2.bias",
        "decoder.up.3.block.1.norm2.weight": "blocks.4.norm2.weight",
        "decoder.up.3.block.2.conv1.bias": "blocks.5.conv1.bias",
        "decoder.up.3.block.2.conv1.weight": "blocks.5.conv1.weight",
        "decoder.up.3.block.2.conv2.bias": "blocks.5.conv2.bias",
        "decoder.up.3.block.2.conv2.weight": "blocks.5.conv2.weight",
        "decoder.up.3.block.2.norm1.bias": "blocks.5.norm1.bias",
        "decoder.up.3.block.2.norm1.weight": "blocks.5.norm1.weight",
        "decoder.up.3.block.2.norm2.bias": "blocks.5.norm2.bias",
        "decoder.up.3.block.2.norm2.weight": "blocks.5.norm2.weight",
        "decoder.up.3.upsample.conv.bias": "blocks.6.conv.bias",
        "decoder.up.3.upsample.conv.weight": "blocks.6.conv.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            param = state_dict[name]
            state_dict_[rename_dict[name]] = param
    return state_dict_


def FluxVAEEncoderStateDictConverterDiffusers(state_dict):
    # architecture
    block_types = [
        'ResnetBlock', 'ResnetBlock', 'DownSampler',
        'ResnetBlock', 'ResnetBlock', 'DownSampler',
        'ResnetBlock', 'ResnetBlock', 'DownSampler',
        'ResnetBlock', 'ResnetBlock',
        'ResnetBlock', 'VAEAttentionBlock', 'ResnetBlock'
    ]

    # Rename each parameter
    local_rename_dict = {
        "quant_conv": "quant_conv",
        "encoder.conv_in": "conv_in",
        "encoder.mid_block.attentions.0.group_norm": "blocks.12.norm",
        "encoder.mid_block.attentions.0.to_q": "blocks.12.transformer_blocks.0.to_q",
        "encoder.mid_block.attentions.0.to_k": "blocks.12.transformer_blocks.0.to_k",
        "encoder.mid_block.attentions.0.to_v": "blocks.12.transformer_blocks.0.to_v",
        "encoder.mid_block.attentions.0.to_out.0": "blocks.12.transformer_blocks.0.to_out",
        "encoder.mid_block.resnets.0.norm1": "blocks.11.norm1",
        "encoder.mid_block.resnets.0.conv1": "blocks.11.conv1",
        "encoder.mid_block.resnets.0.norm2": "blocks.11.norm2",
        "encoder.mid_block.resnets.0.conv2": "blocks.11.conv2",
        "encoder.mid_block.resnets.1.norm1": "blocks.13.norm1",
        "encoder.mid_block.resnets.1.conv1": "blocks.13.conv1",
        "encoder.mid_block.resnets.1.norm2": "blocks.13.norm2",
        "encoder.mid_block.resnets.1.conv2": "blocks.13.conv2",
        "encoder.conv_norm_out": "conv_norm_out",
        "encoder.conv_out": "conv_out",
    }
    name_list = sorted([name for name in state_dict])
    rename_dict = {}
    block_id = {"ResnetBlock": -1, "DownSampler": -1, "UpSampler": -1}
    last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
    for name in name_list:
        names = name.split(".")
        name_prefix = ".".join(names[:-1])
        if name_prefix in local_rename_dict:
            rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
        elif name.startswith("encoder.down_blocks"):
            block_type = {"resnets": "ResnetBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[3]]
            block_type_with_id = ".".join(names[:5])
            if block_type_with_id != last_block_type_with_id[block_type]:
                block_id[block_type] += 1
            last_block_type_with_id[block_type] = block_type_with_id
            while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                block_id[block_type] += 1
            block_type_with_id = ".".join(names[:5])
            names = ["blocks", str(block_id[block_type])] + names[5:]
            rename_dict[name] = ".".join(names)

    # Convert state_dict
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            state_dict_[rename_dict[name]] = state_dict[name]
    return state_dict_


def FluxVAEDecoderStateDictConverterDiffusers(state_dict):
    # architecture
        block_types = [
            'ResnetBlock', 'VAEAttentionBlock', 'ResnetBlock',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock', 'UpSampler',
            'ResnetBlock', 'ResnetBlock', 'ResnetBlock'
        ]

        # Rename each parameter
        local_rename_dict = {
            "post_quant_conv": "post_quant_conv",
            "decoder.conv_in": "conv_in",
            "decoder.mid_block.attentions.0.group_norm": "blocks.1.norm",
            "decoder.mid_block.attentions.0.to_q": "blocks.1.transformer_blocks.0.to_q",
            "decoder.mid_block.attentions.0.to_k": "blocks.1.transformer_blocks.0.to_k",
            "decoder.mid_block.attentions.0.to_v": "blocks.1.transformer_blocks.0.to_v",
            "decoder.mid_block.attentions.0.to_out.0": "blocks.1.transformer_blocks.0.to_out",
            "decoder.mid_block.resnets.0.norm1": "blocks.0.norm1",
            "decoder.mid_block.resnets.0.conv1": "blocks.0.conv1",
            "decoder.mid_block.resnets.0.norm2": "blocks.0.norm2",
            "decoder.mid_block.resnets.0.conv2": "blocks.0.conv2",
            "decoder.mid_block.resnets.1.norm1": "blocks.2.norm1",
            "decoder.mid_block.resnets.1.conv1": "blocks.2.conv1",
            "decoder.mid_block.resnets.1.norm2": "blocks.2.norm2",
            "decoder.mid_block.resnets.1.conv2": "blocks.2.conv2",
            "decoder.conv_norm_out": "conv_norm_out",
            "decoder.conv_out": "conv_out",
        }
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": 2, "DownSampler": 2, "UpSampler": 2}
        last_block_type_with_id = {"ResnetBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            name_prefix = ".".join(names[:-1])
            if name_prefix in local_rename_dict:
                rename_dict[name] = local_rename_dict[name_prefix] + "." + names[-1]
            elif name.startswith("decoder.up_blocks"):
                block_type = {"resnets": "ResnetBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[3]]
                block_type_with_id = ".".join(names[:5])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:5])
                names = ["blocks", str(block_id[block_type])] + names[5:]
                rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                state_dict_[rename_dict[name]] = state_dict[name]
        return state_dict_