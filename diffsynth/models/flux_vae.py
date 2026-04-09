import torch
from einops import rearrange, repeat


class TileWorker:
    def __init__(self):
        pass


    def mask(self, height, width, border_width):
        # Create a mask with shape (height, width).
        # The centre area is filled with 1, and the border line is filled with values in range (0, 1].
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        return mask


    def tile(self, model_input, tile_size, tile_stride, tile_device, tile_dtype):
        # Convert a tensor (b, c, h, w) to (b, c, tile_size, tile_size, tile_num)
        batch_size, channel, _, _ = model_input.shape
        model_input = model_input.to(device=tile_device, dtype=tile_dtype)
        unfold_operator = torch.nn.Unfold(
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        model_input = unfold_operator(model_input)
        model_input = model_input.view((batch_size, channel, tile_size, tile_size, -1))

        return model_input


    def tiled_inference(self, forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype):
        # Call y=forward_fn(x) for each tile
        tile_num = model_input.shape[-1]
        model_output_stack = []

        for tile_id in range(0, tile_num, tile_batch_size):

            # process input
            tile_id_ = min(tile_id + tile_batch_size, tile_num)
            x = model_input[:, :, :, :, tile_id: tile_id_]
            x = x.to(device=inference_device, dtype=inference_dtype)
            x = rearrange(x, "b c h w n -> (n b) c h w")

            # process output
            y = forward_fn(x)
            y = rearrange(y, "(n b) c h w -> b c h w n", n=tile_id_-tile_id)
            y = y.to(device=tile_device, dtype=tile_dtype)
            model_output_stack.append(y)

        model_output = torch.concat(model_output_stack, dim=-1)
        return model_output


    def io_scale(self, model_output, tile_size):
        # Determine the size modification happened in forward_fn
        # We only consider the same scale on height and width.
        io_scale = model_output.shape[2] / tile_size
        return io_scale
    

    def untile(self, model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype):
        # The reversed function of tile
        mask = self.mask(tile_size, tile_size, border_width)
        mask = mask.to(device=tile_device, dtype=tile_dtype)
        mask = rearrange(mask, "h w -> 1 1 h w 1")
        model_output = model_output * mask

        fold_operator = torch.nn.Fold(
            output_size=(height, width),
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        mask = repeat(mask[0, 0, :, :, 0], "h w -> 1 (h w) n", n=model_output.shape[-1])
        model_output = rearrange(model_output, "b c h w n -> b (c h w) n")
        model_output = fold_operator(model_output) / fold_operator(mask)

        return model_output


    def tiled_forward(self, forward_fn, model_input, tile_size, tile_stride, tile_batch_size=1, tile_device="cpu", tile_dtype=torch.float32, border_width=None):
        # Prepare
        inference_device, inference_dtype = model_input.device, model_input.dtype
        height, width = model_input.shape[2], model_input.shape[3]
        border_width = int(tile_stride*0.5) if border_width is None else border_width

        # tile
        model_input = self.tile(model_input, tile_size, tile_stride, tile_device, tile_dtype)

        # inference
        model_output = self.tiled_inference(forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype)

        # resize
        io_scale = self.io_scale(model_output, tile_size)
        height, width = int(height*io_scale), int(width*io_scale)
        tile_size, tile_stride = int(tile_size*io_scale), int(tile_stride*io_scale)
        border_width = int(border_width*io_scale)

        # untile
        model_output = self.untile(model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype)
        
        # Done!
        model_output = model_output.to(device=inference_device, dtype=inference_dtype)
        return model_output


class ConvAttention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Conv2d(q_dim, dim_inner, kernel_size=(1, 1), bias=bias_q)
        self.to_k = torch.nn.Conv2d(kv_dim, dim_inner, kernel_size=(1, 1), bias=bias_kv)
        self.to_v = torch.nn.Conv2d(kv_dim, dim_inner, kernel_size=(1, 1), bias=bias_kv)
        self.to_out = torch.nn.Conv2d(dim_inner, q_dim, kernel_size=(1, 1), bias=bias_out)

    def forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        conv_input = rearrange(hidden_states, "B L C -> B C L 1")
        q = self.to_q(conv_input)
        q = rearrange(q[:, :, :, 0], "B C L -> B L C")
        conv_input = rearrange(encoder_hidden_states, "B L C -> B C L 1")
        k = self.to_k(conv_input)
        v = self.to_v(conv_input)
        k = rearrange(k[:, :, :, 0], "B C L -> B L C")
        v = rearrange(v[:, :, :, 0], "B C L -> B L C")

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        conv_input = rearrange(hidden_states, "B L C -> B C L 1")
        hidden_states = self.to_out(conv_input)
        hidden_states = rearrange(hidden_states[:, :, :, 0], "B C L -> B L C")

        return hidden_states


class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class VAEAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5, use_conv_attention=True):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)

        if use_conv_attention:
            self.transformer_blocks = torch.nn.ModuleList([
                ConvAttention(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    bias_q=True,
                    bias_kv=True,
                    bias_out=True
                )
                for d in range(num_layers)
            ])
        else:
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


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class DownSampler(torch.nn.Module):
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class FluxVAEDecoder(torch.nn.Module):
    def __init__(self, use_conv_attention=True):
        super().__init__()
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.conv_in = torch.nn.Conv2d(16, 512, kernel_size=3, padding=1) # Different from SD 1.x

        self.blocks = torch.nn.ModuleList([
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, use_conv_attention=use_conv_attention),
            ResnetBlock(512, 512, eps=1e-6),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            UpSampler(512),
            # UpDecoderBlock2D
            ResnetBlock(512, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            UpSampler(256),
            # UpDecoderBlock2D
            ResnetBlock(256, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
    
    def tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)

        # 1. pre-process
        hidden_states = sample / self.scaling_factor + self.shift_factor
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

        return hidden_states


class FluxVAEEncoder(torch.nn.Module):
    def __init__(self, use_conv_attention=True):
        super().__init__()
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6, use_conv_attention=use_conv_attention),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 32, kernel_size=3, padding=1)

    def tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)
        
        # 1. pre-process
        hidden_states = self.conv_in(sample)
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
        hidden_states = hidden_states[:, :16]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor

        return hidden_states
    
    def encode_video(self, sample, batch_size=8):
        B = sample.shape[0]
        hidden_states = []

        for i in range(0, sample.shape[2], batch_size):

            j = min(i + batch_size, sample.shape[2])
            sample_batch = rearrange(sample[:,:,i:j], "B C T H W -> (B T) C H W")

            hidden_states_batch = self(sample_batch)
            hidden_states_batch = rearrange(hidden_states_batch, "(B T) C H W -> B C T H W", B=B)

            hidden_states.append(hidden_states_batch)
        
        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states
