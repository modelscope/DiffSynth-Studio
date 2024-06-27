import torch, math
from einops import rearrange, repeat
from .sd_unet import Timesteps, PushBlock, PopBlock, Attention, GEGLU, ResnetBlock, AttentionBlock, DownSampler, UpSampler


class TemporalResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = rearrange(hidden_states, "f c h w -> 1 c f h w")
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)
            emb = repeat(emb, "b c -> b c f 1 1", f=hidden_states.shape[0])
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        x = rearrange(x[0], "c f h w -> f c h w")
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalTimesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb
    

class TrainableTemporalTimesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, num_frames: int):
        super().__init__()
        timesteps = PositionalID()(num_frames)
        embeddings = get_timestep_embedding(timesteps, num_channels, flip_sin_to_cos, downscale_freq_shift)
        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, timesteps):
        t_emb = self.embeddings[timesteps]
        return t_emb


class PositionalID(torch.nn.Module):
    def __init__(self, max_id=25, repeat_length=20):
        super().__init__()
        self.max_id = max_id
        self.repeat_length = repeat_length

    def frame_id_to_position_id(self, frame_id):
        if frame_id < self.max_id:
            position_id = frame_id
        else:
            position_id = (frame_id - self.max_id) % (self.repeat_length * 2)
            if position_id < self.repeat_length:
                position_id = self.max_id - 2 - position_id
            else:
                position_id = self.max_id - 2 * self.repeat_length + position_id
        return position_id

    def forward(self, num_frames, pivot_frame_id=0):
        position_ids = [self.frame_id_to_position_id(abs(i-pivot_frame_id)) for i in range(num_frames)]
        position_ids = torch.IntTensor(position_ids)
        return position_ids


class TemporalAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, cross_attention_dim=None, add_positional_conv=None):
        super().__init__()

        self.positional_embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(in_channels * 4, in_channels)
        )
        if add_positional_conv is not None:
            self.positional_embedding = TrainableTemporalTimesteps(in_channels, True, 0, add_positional_conv)
            self.positional_conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode="reflect")
        else:
            self.positional_embedding = TemporalTimesteps(in_channels, True, 0)
            self.positional_conv = None

        self.norm_in = torch.nn.LayerNorm(in_channels)
        self.act_fn_in = GEGLU(in_channels, in_channels * 4)
        self.ff_in = torch.nn.Linear(in_channels * 4, in_channels)

        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.attn1 = Attention(
            q_dim=in_channels,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True
        )

        self.norm2 = torch.nn.LayerNorm(in_channels)
        self.attn2 = Attention(
            q_dim=in_channels,
            kv_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True
        )

        self.norm_out = torch.nn.LayerNorm(in_channels)
        self.act_fn_out = GEGLU(in_channels, in_channels * 4)
        self.ff_out = torch.nn.Linear(in_channels * 4, in_channels)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):

        batch, inner_dim, height, width = hidden_states.shape
        pos_emb = torch.arange(batch)
        pos_emb = self.positional_embedding(pos_emb).to(dtype=hidden_states.dtype, device=hidden_states.device)
        pos_emb = self.positional_embedding_proj(pos_emb)
        
        hidden_states = rearrange(hidden_states, "T C H W -> 1 C T H W") + rearrange(pos_emb, "T C -> 1 C T 1 1")
        if self.positional_conv is not None:
            hidden_states = self.positional_conv(hidden_states)
        hidden_states = rearrange(hidden_states[0], "C T H W -> (H W) T C")

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.act_fn_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=text_emb.repeat(height * width, 1))
        hidden_states = attn_output + hidden_states

        residual = hidden_states
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.act_fn_out(hidden_states)
        hidden_states = self.ff_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states.reshape(height, width, batch, inner_dim).permute(2, 3, 0, 1)

        return hidden_states, time_emb, text_emb, res_stack
    

class PopMixBlock(torch.nn.Module):
    def __init__(self, in_channels=None):
        super().__init__()
        self.mix_factor = torch.nn.Parameter(torch.Tensor([0.5]))
        self.need_proj = in_channels is not None
        if self.need_proj:
            self.proj = torch.nn.Linear(in_channels, in_channels)
    
    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_hidden_states = res_stack.pop()
        alpha = torch.sigmoid(self.mix_factor)
        hidden_states = alpha * res_hidden_states + (1 - alpha) * hidden_states
        if self.need_proj:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.proj(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            res_hidden_states = res_stack.pop()
            hidden_states = hidden_states + res_hidden_states
        return hidden_states, time_emb, text_emb, res_stack


class SVDUNet(torch.nn.Module):
    def __init__(self, add_positional_conv=None):
        super().__init__()
        self.time_proj = Timesteps(320)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.add_time_proj = Timesteps(256)
        self.add_time_embedding = torch.nn.Sequential(
            torch.nn.Linear(768, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.conv_in = torch.nn.Conv2d(8, 320, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(320, 320, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024, add_positional_conv),   PopMixBlock(320),  PushBlock(),
            ResnetBlock(320, 320, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024, add_positional_conv),   PopMixBlock(320),  PushBlock(),
            DownSampler(320), PushBlock(),
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(320, 640, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024, add_positional_conv),  PopMixBlock(640),  PushBlock(),
            ResnetBlock(640, 640, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024, add_positional_conv),  PopMixBlock(640),  PushBlock(),
            DownSampler(640), PushBlock(),
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(640, 1280, 1280, eps=1e-6),                     PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280), PushBlock(),
            ResnetBlock(1280, 1280, 1280, eps=1e-6),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280), PushBlock(),
            DownSampler(1280), PushBlock(),
            # DownBlockSpatioTemporal
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),  PushBlock(),
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),  PushBlock(),
            # UNetMidBlockSpatioTemporal
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280),
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            # UpBlockSpatioTemporal
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            UpSampler(1280),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(2560, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280),
            PopBlock(),        ResnetBlock(2560, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280),
            PopBlock(),        ResnetBlock(1920, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),  PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024, add_positional_conv), PopMixBlock(1280),
            UpSampler(1280),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(1920, 640, 1280, eps=1e-6),  PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024, add_positional_conv),  PopMixBlock(640),
            PopBlock(),        ResnetBlock(1280, 640, 1280, eps=1e-6),  PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024, add_positional_conv),  PopMixBlock(640),
            PopBlock(),        ResnetBlock(960, 640, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024, add_positional_conv),  PopMixBlock(640),
            UpSampler(640),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(960, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024, add_positional_conv),   PopMixBlock(320),
            PopBlock(),        ResnetBlock(640, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024, add_positional_conv),   PopMixBlock(320),
            PopBlock(),        ResnetBlock(640, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),  PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024, add_positional_conv),   PopMixBlock(320),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        

    def build_mask(self, data, is_bound):
        T, C, H, W = data.shape
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
        mask = rearrange(mask, "T H W -> T 1 H W")
        return mask
    

    def tiled_forward(
        self, sample, timestep, encoder_hidden_states, add_time_id,
        batch_time=25, batch_height=128, batch_width=128,
        stride_time=5, stride_height=64, stride_width=64,
        progress_bar=lambda x:x
    ):
        data_device = sample.device
        computation_device = self.conv_in.weight.device
        torch_dtype = sample.dtype
        T, C, H, W = sample.shape

        weight = torch.zeros((T, 1, H, W), dtype=torch_dtype, device=data_device)
        values = torch.zeros((T, 4, H, W), dtype=torch_dtype, device=data_device)

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
            sample_batch = sample[tl:tr, :, hl:hr, wl:wr].to(computation_device)
            sample_batch = self.forward(sample_batch, timestep, encoder_hidden_states, add_time_id).to(data_device)
            mask = self.build_mask(sample_batch, is_bound=(tl==0, tr>=T, hl==0, hr>=H, wl==0, wr>=W))
            values[tl:tr, :, hl:hr, wl:wr] += sample_batch * mask
            weight[tl:tr, :, hl:hr, wl:wr] += mask
        values /= weight
        return values
    

    def forward(self, sample, timestep, encoder_hidden_states, add_time_id, use_gradient_checkpointing=False, **kwargs):
        # 1. time
        timestep = torch.tensor((timestep,)).to(sample.device)
        t_emb = self.time_proj(timestep).to(sample.dtype)
        t_emb = self.time_embedding(t_emb)

        add_embeds = self.add_time_proj(add_time_id.flatten()).to(sample.dtype)
        add_embeds = add_embeds.reshape((-1, 768))
        add_embeds = self.add_time_embedding(add_embeds)

        time_emb = t_emb + add_embeds

        # 2. pre-process
        height, width = sample.shape[2], sample.shape[3]
        hidden_states = self.conv_in(sample)
        text_emb = encoder_hidden_states
        res_stack = [hidden_states]

        # 3. blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        for i, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing and not (isinstance(block, PushBlock) or isinstance(block, PopBlock) or isinstance(block, PopMixBlock)):
                hidden_states, time_emb, text_emb, res_stack = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, time_emb, text_emb, res_stack,
                    use_reentrant=False,
                )
            else:
                hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 4. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states
    
    def state_dict_converter(self):
        return SVDUNetStateDictConverter()
    


class SVDUNetStateDictConverter:
    def __init__(self):
        pass

    def get_block_name(self, names):
        if names[0] in ["down_blocks", "mid_block", "up_blocks"]:
            if names[4] in ["norm", "proj_in"]:
                return ".".join(names[:4] + ["transformer_blocks"])
            elif names[4] in ["time_pos_embed"]:
                return ".".join(names[:4] + ["temporal_transformer_blocks"])
            elif names[4] in ["proj_out"]:
                return ".".join(names[:4] + ["time_mixer"])
            else:
                return ".".join(names[:5])
        return ""

    def from_diffusers(self, state_dict):
        rename_dict = {
            "time_embedding.linear_1": "time_embedding.0",
            "time_embedding.linear_2": "time_embedding.2",
            "add_embedding.linear_1": "add_time_embedding.0",
            "add_embedding.linear_2": "add_time_embedding.2",
            "conv_in": "conv_in",
            "conv_norm_out": "conv_norm_out",
            "conv_out": "conv_out",
        }
        blocks_rename_dict = [
            "down_blocks.0.resnets.0.spatial_res_block", None, "down_blocks.0.resnets.0.temporal_res_block", "down_blocks.0.resnets.0.time_mixer", None,
            "down_blocks.0.attentions.0.transformer_blocks", None, "down_blocks.0.attentions.0.temporal_transformer_blocks", "down_blocks.0.attentions.0.time_mixer", None,
            "down_blocks.0.resnets.1.spatial_res_block", None, "down_blocks.0.resnets.1.temporal_res_block", "down_blocks.0.resnets.1.time_mixer", None,
            "down_blocks.0.attentions.1.transformer_blocks", None, "down_blocks.0.attentions.1.temporal_transformer_blocks", "down_blocks.0.attentions.1.time_mixer", None,
            "down_blocks.0.downsamplers.0.conv", None,
            "down_blocks.1.resnets.0.spatial_res_block", None, "down_blocks.1.resnets.0.temporal_res_block", "down_blocks.1.resnets.0.time_mixer", None,
            "down_blocks.1.attentions.0.transformer_blocks", None, "down_blocks.1.attentions.0.temporal_transformer_blocks", "down_blocks.1.attentions.0.time_mixer", None,
            "down_blocks.1.resnets.1.spatial_res_block", None, "down_blocks.1.resnets.1.temporal_res_block", "down_blocks.1.resnets.1.time_mixer", None,
            "down_blocks.1.attentions.1.transformer_blocks", None, "down_blocks.1.attentions.1.temporal_transformer_blocks", "down_blocks.1.attentions.1.time_mixer", None,
            "down_blocks.1.downsamplers.0.conv", None,
            "down_blocks.2.resnets.0.spatial_res_block", None, "down_blocks.2.resnets.0.temporal_res_block", "down_blocks.2.resnets.0.time_mixer", None,
            "down_blocks.2.attentions.0.transformer_blocks", None, "down_blocks.2.attentions.0.temporal_transformer_blocks", "down_blocks.2.attentions.0.time_mixer", None,
            "down_blocks.2.resnets.1.spatial_res_block", None, "down_blocks.2.resnets.1.temporal_res_block", "down_blocks.2.resnets.1.time_mixer", None,
            "down_blocks.2.attentions.1.transformer_blocks", None, "down_blocks.2.attentions.1.temporal_transformer_blocks", "down_blocks.2.attentions.1.time_mixer", None,
            "down_blocks.2.downsamplers.0.conv", None,
            "down_blocks.3.resnets.0.spatial_res_block", None, "down_blocks.3.resnets.0.temporal_res_block", "down_blocks.3.resnets.0.time_mixer", None,
            "down_blocks.3.resnets.1.spatial_res_block", None, "down_blocks.3.resnets.1.temporal_res_block", "down_blocks.3.resnets.1.time_mixer", None,
            "mid_block.mid_block.resnets.0.spatial_res_block", None, "mid_block.mid_block.resnets.0.temporal_res_block", "mid_block.mid_block.resnets.0.time_mixer", None,
            "mid_block.mid_block.attentions.0.transformer_blocks", None, "mid_block.mid_block.attentions.0.temporal_transformer_blocks", "mid_block.mid_block.attentions.0.time_mixer",
            "mid_block.mid_block.resnets.1.spatial_res_block", None, "mid_block.mid_block.resnets.1.temporal_res_block", "mid_block.mid_block.resnets.1.time_mixer",
            None, "up_blocks.0.resnets.0.spatial_res_block", None, "up_blocks.0.resnets.0.temporal_res_block", "up_blocks.0.resnets.0.time_mixer",
            None, "up_blocks.0.resnets.1.spatial_res_block", None, "up_blocks.0.resnets.1.temporal_res_block", "up_blocks.0.resnets.1.time_mixer",
            None, "up_blocks.0.resnets.2.spatial_res_block", None, "up_blocks.0.resnets.2.temporal_res_block", "up_blocks.0.resnets.2.time_mixer",
            "up_blocks.0.upsamplers.0.conv",
            None, "up_blocks.1.resnets.0.spatial_res_block", None, "up_blocks.1.resnets.0.temporal_res_block", "up_blocks.1.resnets.0.time_mixer", None,
            "up_blocks.1.attentions.0.transformer_blocks", None, "up_blocks.1.attentions.0.temporal_transformer_blocks", "up_blocks.1.attentions.0.time_mixer",
            None, "up_blocks.1.resnets.1.spatial_res_block", None, "up_blocks.1.resnets.1.temporal_res_block", "up_blocks.1.resnets.1.time_mixer", None,
            "up_blocks.1.attentions.1.transformer_blocks", None, "up_blocks.1.attentions.1.temporal_transformer_blocks", "up_blocks.1.attentions.1.time_mixer",
            None, "up_blocks.1.resnets.2.spatial_res_block", None, "up_blocks.1.resnets.2.temporal_res_block", "up_blocks.1.resnets.2.time_mixer", None,
            "up_blocks.1.attentions.2.transformer_blocks", None, "up_blocks.1.attentions.2.temporal_transformer_blocks", "up_blocks.1.attentions.2.time_mixer",
            "up_blocks.1.upsamplers.0.conv",
            None, "up_blocks.2.resnets.0.spatial_res_block", None, "up_blocks.2.resnets.0.temporal_res_block", "up_blocks.2.resnets.0.time_mixer", None,
            "up_blocks.2.attentions.0.transformer_blocks", None, "up_blocks.2.attentions.0.temporal_transformer_blocks", "up_blocks.2.attentions.0.time_mixer",
            None, "up_blocks.2.resnets.1.spatial_res_block", None, "up_blocks.2.resnets.1.temporal_res_block", "up_blocks.2.resnets.1.time_mixer", None,
            "up_blocks.2.attentions.1.transformer_blocks", None, "up_blocks.2.attentions.1.temporal_transformer_blocks", "up_blocks.2.attentions.1.time_mixer",
            None, "up_blocks.2.resnets.2.spatial_res_block", None, "up_blocks.2.resnets.2.temporal_res_block", "up_blocks.2.resnets.2.time_mixer", None,
            "up_blocks.2.attentions.2.transformer_blocks", None, "up_blocks.2.attentions.2.temporal_transformer_blocks", "up_blocks.2.attentions.2.time_mixer",
            "up_blocks.2.upsamplers.0.conv",
            None, "up_blocks.3.resnets.0.spatial_res_block", None, "up_blocks.3.resnets.0.temporal_res_block", "up_blocks.3.resnets.0.time_mixer", None,
            "up_blocks.3.attentions.0.transformer_blocks", None, "up_blocks.3.attentions.0.temporal_transformer_blocks", "up_blocks.3.attentions.0.time_mixer",
            None, "up_blocks.3.resnets.1.spatial_res_block", None, "up_blocks.3.resnets.1.temporal_res_block", "up_blocks.3.resnets.1.time_mixer", None,
            "up_blocks.3.attentions.1.transformer_blocks", None, "up_blocks.3.attentions.1.temporal_transformer_blocks", "up_blocks.3.attentions.1.time_mixer",
            None, "up_blocks.3.resnets.2.spatial_res_block", None, "up_blocks.3.resnets.2.temporal_res_block", "up_blocks.3.resnets.2.time_mixer", None,
            "up_blocks.3.attentions.2.transformer_blocks", None, "up_blocks.3.attentions.2.temporal_transformer_blocks", "up_blocks.3.attentions.2.time_mixer",
        ]
        blocks_rename_dict = {i:j for j,i in enumerate(blocks_rename_dict) if i is not None}
        state_dict_ = {}
        for name, param in sorted(state_dict.items()):
            names = name.split(".")
            if names[0] == "mid_block":
                names = ["mid_block"] + names
            if names[-1] in ["weight", "bias"]:
                name_prefix = ".".join(names[:-1])
                if name_prefix in rename_dict:
                    state_dict_[rename_dict[name_prefix] + "." + names[-1]] = param
                else:
                    block_name = self.get_block_name(names)
                    if "resnets" in block_name and block_name in blocks_rename_dict:
                        rename = ".".join(["blocks", str(blocks_rename_dict[block_name])] + names[5:])
                        state_dict_[rename] = param
                    elif ("downsamplers" in block_name or "upsamplers" in block_name) and block_name in blocks_rename_dict:
                        rename = ".".join(["blocks", str(blocks_rename_dict[block_name])] + names[-2:])
                        state_dict_[rename] = param
                    elif "attentions" in block_name and block_name in blocks_rename_dict:
                        attention_id = names[5]
                        if "transformer_blocks" in names:
                            suffix_dict = {
                                "attn1.to_out.0": "attn1.to_out",
                                "attn2.to_out.0": "attn2.to_out",
                                "ff.net.0.proj": "act_fn.proj",
                                "ff.net.2": "ff",
                            }
                            suffix = ".".join(names[6:-1])
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), "transformer_blocks", attention_id, suffix, names[-1]])
                        elif "temporal_transformer_blocks" in names:
                            suffix_dict = {
                                "attn1.to_out.0": "attn1.to_out",
                                "attn2.to_out.0": "attn2.to_out",
                                "ff_in.net.0.proj": "act_fn_in.proj",
                                "ff_in.net.2": "ff_in",
                                "ff.net.0.proj": "act_fn_out.proj",
                                "ff.net.2": "ff_out",
                                "norm3": "norm_out",
                            }
                            suffix = ".".join(names[6:-1])
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), suffix, names[-1]])
                        elif "time_mixer" in block_name:
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), "proj", names[-1]])
                        else:
                            suffix_dict = {
                                "linear_1": "positional_embedding_proj.0",
                                "linear_2": "positional_embedding_proj.2",
                            }
                            suffix = names[-2]
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), suffix, names[-1]])
                        state_dict_[rename] = param
                    else:
                        print(name)
            else:
                block_name = self.get_block_name(names)
                if len(block_name)>0 and block_name in blocks_rename_dict:
                    rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), names[-1]])
                    state_dict_[rename] = param
        return state_dict_
    

    def from_civitai(self, state_dict, add_positional_conv=None):
        rename_dict = {
            "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
            "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
            "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias": "blocks.0.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": "blocks.0.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.bias": "blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight": "blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.2.bias": "blocks.0.conv1.bias",
            "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": "blocks.0.conv1.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.0.bias": "blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.1.0.out_layers.0.weight": "blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.3.bias": "blocks.0.conv2.bias",
            "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": "blocks.0.conv2.weight",
            "model.diffusion_model.input_blocks.1.0.time_mixer.mix_factor": "blocks.3.mix_factor",
            "model.diffusion_model.input_blocks.1.0.time_stack.emb_layers.1.bias": "blocks.2.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.1.0.time_stack.emb_layers.1.weight": "blocks.2.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.1.0.time_stack.in_layers.0.bias": "blocks.2.norm1.bias",
            "model.diffusion_model.input_blocks.1.0.time_stack.in_layers.0.weight": "blocks.2.norm1.weight",
            "model.diffusion_model.input_blocks.1.0.time_stack.in_layers.2.bias": "blocks.2.conv1.bias",
            "model.diffusion_model.input_blocks.1.0.time_stack.in_layers.2.weight": "blocks.2.conv1.weight",
            "model.diffusion_model.input_blocks.1.0.time_stack.out_layers.0.bias": "blocks.2.norm2.bias",
            "model.diffusion_model.input_blocks.1.0.time_stack.out_layers.0.weight": "blocks.2.norm2.weight",
            "model.diffusion_model.input_blocks.1.0.time_stack.out_layers.3.bias": "blocks.2.conv2.bias",
            "model.diffusion_model.input_blocks.1.0.time_stack.out_layers.3.weight": "blocks.2.conv2.weight",
            "model.diffusion_model.input_blocks.1.1.norm.bias": "blocks.5.norm.bias",
            "model.diffusion_model.input_blocks.1.1.norm.weight": "blocks.5.norm.weight",
            "model.diffusion_model.input_blocks.1.1.proj_in.bias": "blocks.5.proj_in.bias",
            "model.diffusion_model.input_blocks.1.1.proj_in.weight": "blocks.5.proj_in.weight",
            "model.diffusion_model.input_blocks.1.1.proj_out.bias": "blocks.8.proj.bias",
            "model.diffusion_model.input_blocks.1.1.proj_out.weight": "blocks.8.proj.weight",
            "model.diffusion_model.input_blocks.1.1.time_mixer.mix_factor": "blocks.8.mix_factor",
            "model.diffusion_model.input_blocks.1.1.time_pos_embed.0.bias": "blocks.7.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.1.1.time_pos_embed.0.weight": "blocks.7.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.1.1.time_pos_embed.2.bias": "blocks.7.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.1.1.time_pos_embed.2.weight": "blocks.7.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_k.weight": "blocks.7.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_out.0.bias": "blocks.7.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_out.0.weight": "blocks.7.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_q.weight": "blocks.7.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn1.to_v.weight": "blocks.7.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_k.weight": "blocks.7.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_out.0.bias": "blocks.7.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_out.0.weight": "blocks.7.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_q.weight": "blocks.7.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.attn2.to_v.weight": "blocks.7.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff.net.0.proj.bias": "blocks.7.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff.net.0.proj.weight": "blocks.7.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff.net.2.bias": "blocks.7.ff_out.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff.net.2.weight": "blocks.7.ff_out.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.7.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.7.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff_in.net.2.bias": "blocks.7.ff_in.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.ff_in.net.2.weight": "blocks.7.ff_in.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm1.bias": "blocks.7.norm1.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm1.weight": "blocks.7.norm1.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm2.bias": "blocks.7.norm2.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm2.weight": "blocks.7.norm2.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm3.bias": "blocks.7.norm_out.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm3.weight": "blocks.7.norm_out.weight",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm_in.bias": "blocks.7.norm_in.bias",
            "model.diffusion_model.input_blocks.1.1.time_stack.0.norm_in.weight": "blocks.7.norm_in.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": "blocks.5.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.5.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.5.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": "blocks.5.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": "blocks.5.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": "blocks.5.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.5.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.5.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": "blocks.5.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": "blocks.5.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.5.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.5.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias": "blocks.5.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight": "blocks.5.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias": "blocks.5.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight": "blocks.5.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias": "blocks.5.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight": "blocks.5.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias": "blocks.5.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight": "blocks.5.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.10.0.emb_layers.1.bias": "blocks.66.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.10.0.emb_layers.1.weight": "blocks.66.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.10.0.in_layers.0.bias": "blocks.66.norm1.bias",
            "model.diffusion_model.input_blocks.10.0.in_layers.0.weight": "blocks.66.norm1.weight",
            "model.diffusion_model.input_blocks.10.0.in_layers.2.bias": "blocks.66.conv1.bias",
            "model.diffusion_model.input_blocks.10.0.in_layers.2.weight": "blocks.66.conv1.weight",
            "model.diffusion_model.input_blocks.10.0.out_layers.0.bias": "blocks.66.norm2.bias",
            "model.diffusion_model.input_blocks.10.0.out_layers.0.weight": "blocks.66.norm2.weight",
            "model.diffusion_model.input_blocks.10.0.out_layers.3.bias": "blocks.66.conv2.bias",
            "model.diffusion_model.input_blocks.10.0.out_layers.3.weight": "blocks.66.conv2.weight",
            "model.diffusion_model.input_blocks.10.0.time_mixer.mix_factor": "blocks.69.mix_factor",
            "model.diffusion_model.input_blocks.10.0.time_stack.emb_layers.1.bias": "blocks.68.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.10.0.time_stack.emb_layers.1.weight": "blocks.68.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.10.0.time_stack.in_layers.0.bias": "blocks.68.norm1.bias",
            "model.diffusion_model.input_blocks.10.0.time_stack.in_layers.0.weight": "blocks.68.norm1.weight",
            "model.diffusion_model.input_blocks.10.0.time_stack.in_layers.2.bias": "blocks.68.conv1.bias",
            "model.diffusion_model.input_blocks.10.0.time_stack.in_layers.2.weight": "blocks.68.conv1.weight",
            "model.diffusion_model.input_blocks.10.0.time_stack.out_layers.0.bias": "blocks.68.norm2.bias",
            "model.diffusion_model.input_blocks.10.0.time_stack.out_layers.0.weight": "blocks.68.norm2.weight",
            "model.diffusion_model.input_blocks.10.0.time_stack.out_layers.3.bias": "blocks.68.conv2.bias",
            "model.diffusion_model.input_blocks.10.0.time_stack.out_layers.3.weight": "blocks.68.conv2.weight",
            "model.diffusion_model.input_blocks.11.0.emb_layers.1.bias": "blocks.71.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.11.0.emb_layers.1.weight": "blocks.71.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.11.0.in_layers.0.bias": "blocks.71.norm1.bias",
            "model.diffusion_model.input_blocks.11.0.in_layers.0.weight": "blocks.71.norm1.weight",
            "model.diffusion_model.input_blocks.11.0.in_layers.2.bias": "blocks.71.conv1.bias",
            "model.diffusion_model.input_blocks.11.0.in_layers.2.weight": "blocks.71.conv1.weight",
            "model.diffusion_model.input_blocks.11.0.out_layers.0.bias": "blocks.71.norm2.bias",
            "model.diffusion_model.input_blocks.11.0.out_layers.0.weight": "blocks.71.norm2.weight",
            "model.diffusion_model.input_blocks.11.0.out_layers.3.bias": "blocks.71.conv2.bias",
            "model.diffusion_model.input_blocks.11.0.out_layers.3.weight": "blocks.71.conv2.weight",
            "model.diffusion_model.input_blocks.11.0.time_mixer.mix_factor": "blocks.74.mix_factor",
            "model.diffusion_model.input_blocks.11.0.time_stack.emb_layers.1.bias": "blocks.73.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.11.0.time_stack.emb_layers.1.weight": "blocks.73.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.11.0.time_stack.in_layers.0.bias": "blocks.73.norm1.bias",
            "model.diffusion_model.input_blocks.11.0.time_stack.in_layers.0.weight": "blocks.73.norm1.weight",
            "model.diffusion_model.input_blocks.11.0.time_stack.in_layers.2.bias": "blocks.73.conv1.bias",
            "model.diffusion_model.input_blocks.11.0.time_stack.in_layers.2.weight": "blocks.73.conv1.weight",
            "model.diffusion_model.input_blocks.11.0.time_stack.out_layers.0.bias": "blocks.73.norm2.bias",
            "model.diffusion_model.input_blocks.11.0.time_stack.out_layers.0.weight": "blocks.73.norm2.weight",
            "model.diffusion_model.input_blocks.11.0.time_stack.out_layers.3.bias": "blocks.73.conv2.bias",
            "model.diffusion_model.input_blocks.11.0.time_stack.out_layers.3.weight": "blocks.73.conv2.weight",
            "model.diffusion_model.input_blocks.2.0.emb_layers.1.bias": "blocks.10.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.2.0.emb_layers.1.weight": "blocks.10.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.2.0.in_layers.0.bias": "blocks.10.norm1.bias",
            "model.diffusion_model.input_blocks.2.0.in_layers.0.weight": "blocks.10.norm1.weight",
            "model.diffusion_model.input_blocks.2.0.in_layers.2.bias": "blocks.10.conv1.bias",
            "model.diffusion_model.input_blocks.2.0.in_layers.2.weight": "blocks.10.conv1.weight",
            "model.diffusion_model.input_blocks.2.0.out_layers.0.bias": "blocks.10.norm2.bias",
            "model.diffusion_model.input_blocks.2.0.out_layers.0.weight": "blocks.10.norm2.weight",
            "model.diffusion_model.input_blocks.2.0.out_layers.3.bias": "blocks.10.conv2.bias",
            "model.diffusion_model.input_blocks.2.0.out_layers.3.weight": "blocks.10.conv2.weight",
            "model.diffusion_model.input_blocks.2.0.time_mixer.mix_factor": "blocks.13.mix_factor",
            "model.diffusion_model.input_blocks.2.0.time_stack.emb_layers.1.bias": "blocks.12.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.2.0.time_stack.emb_layers.1.weight": "blocks.12.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.2.0.time_stack.in_layers.0.bias": "blocks.12.norm1.bias",
            "model.diffusion_model.input_blocks.2.0.time_stack.in_layers.0.weight": "blocks.12.norm1.weight",
            "model.diffusion_model.input_blocks.2.0.time_stack.in_layers.2.bias": "blocks.12.conv1.bias",
            "model.diffusion_model.input_blocks.2.0.time_stack.in_layers.2.weight": "blocks.12.conv1.weight",
            "model.diffusion_model.input_blocks.2.0.time_stack.out_layers.0.bias": "blocks.12.norm2.bias",
            "model.diffusion_model.input_blocks.2.0.time_stack.out_layers.0.weight": "blocks.12.norm2.weight",
            "model.diffusion_model.input_blocks.2.0.time_stack.out_layers.3.bias": "blocks.12.conv2.bias",
            "model.diffusion_model.input_blocks.2.0.time_stack.out_layers.3.weight": "blocks.12.conv2.weight",
            "model.diffusion_model.input_blocks.2.1.norm.bias": "blocks.15.norm.bias",
            "model.diffusion_model.input_blocks.2.1.norm.weight": "blocks.15.norm.weight",
            "model.diffusion_model.input_blocks.2.1.proj_in.bias": "blocks.15.proj_in.bias",
            "model.diffusion_model.input_blocks.2.1.proj_in.weight": "blocks.15.proj_in.weight",
            "model.diffusion_model.input_blocks.2.1.proj_out.bias": "blocks.18.proj.bias",
            "model.diffusion_model.input_blocks.2.1.proj_out.weight": "blocks.18.proj.weight",
            "model.diffusion_model.input_blocks.2.1.time_mixer.mix_factor": "blocks.18.mix_factor",
            "model.diffusion_model.input_blocks.2.1.time_pos_embed.0.bias": "blocks.17.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.2.1.time_pos_embed.0.weight": "blocks.17.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.2.1.time_pos_embed.2.bias": "blocks.17.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.2.1.time_pos_embed.2.weight": "blocks.17.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn1.to_k.weight": "blocks.17.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn1.to_out.0.bias": "blocks.17.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn1.to_out.0.weight": "blocks.17.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn1.to_q.weight": "blocks.17.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn1.to_v.weight": "blocks.17.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn2.to_k.weight": "blocks.17.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn2.to_out.0.bias": "blocks.17.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn2.to_out.0.weight": "blocks.17.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn2.to_q.weight": "blocks.17.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.attn2.to_v.weight": "blocks.17.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff.net.0.proj.bias": "blocks.17.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff.net.0.proj.weight": "blocks.17.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff.net.2.bias": "blocks.17.ff_out.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff.net.2.weight": "blocks.17.ff_out.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.17.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.17.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff_in.net.2.bias": "blocks.17.ff_in.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.ff_in.net.2.weight": "blocks.17.ff_in.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm1.bias": "blocks.17.norm1.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm1.weight": "blocks.17.norm1.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm2.bias": "blocks.17.norm2.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm2.weight": "blocks.17.norm2.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm3.bias": "blocks.17.norm_out.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm3.weight": "blocks.17.norm_out.weight",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm_in.bias": "blocks.17.norm_in.bias",
            "model.diffusion_model.input_blocks.2.1.time_stack.0.norm_in.weight": "blocks.17.norm_in.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight": "blocks.15.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.15.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.15.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight": "blocks.15.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight": "blocks.15.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": "blocks.15.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.15.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.15.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight": "blocks.15.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": "blocks.15.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.15.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.15.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias": "blocks.15.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight": "blocks.15.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias": "blocks.15.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight": "blocks.15.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias": "blocks.15.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight": "blocks.15.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias": "blocks.15.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight": "blocks.15.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.3.0.op.bias": "blocks.20.conv.bias",
            "model.diffusion_model.input_blocks.3.0.op.weight": "blocks.20.conv.weight",
            "model.diffusion_model.input_blocks.4.0.emb_layers.1.bias": "blocks.22.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.4.0.emb_layers.1.weight": "blocks.22.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.4.0.in_layers.0.bias": "blocks.22.norm1.bias",
            "model.diffusion_model.input_blocks.4.0.in_layers.0.weight": "blocks.22.norm1.weight",
            "model.diffusion_model.input_blocks.4.0.in_layers.2.bias": "blocks.22.conv1.bias",
            "model.diffusion_model.input_blocks.4.0.in_layers.2.weight": "blocks.22.conv1.weight",
            "model.diffusion_model.input_blocks.4.0.out_layers.0.bias": "blocks.22.norm2.bias",
            "model.diffusion_model.input_blocks.4.0.out_layers.0.weight": "blocks.22.norm2.weight",
            "model.diffusion_model.input_blocks.4.0.out_layers.3.bias": "blocks.22.conv2.bias",
            "model.diffusion_model.input_blocks.4.0.out_layers.3.weight": "blocks.22.conv2.weight",
            "model.diffusion_model.input_blocks.4.0.skip_connection.bias": "blocks.22.conv_shortcut.bias",
            "model.diffusion_model.input_blocks.4.0.skip_connection.weight": "blocks.22.conv_shortcut.weight",
            "model.diffusion_model.input_blocks.4.0.time_mixer.mix_factor": "blocks.25.mix_factor",
            "model.diffusion_model.input_blocks.4.0.time_stack.emb_layers.1.bias": "blocks.24.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.4.0.time_stack.emb_layers.1.weight": "blocks.24.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.4.0.time_stack.in_layers.0.bias": "blocks.24.norm1.bias",
            "model.diffusion_model.input_blocks.4.0.time_stack.in_layers.0.weight": "blocks.24.norm1.weight",
            "model.diffusion_model.input_blocks.4.0.time_stack.in_layers.2.bias": "blocks.24.conv1.bias",
            "model.diffusion_model.input_blocks.4.0.time_stack.in_layers.2.weight": "blocks.24.conv1.weight",
            "model.diffusion_model.input_blocks.4.0.time_stack.out_layers.0.bias": "blocks.24.norm2.bias",
            "model.diffusion_model.input_blocks.4.0.time_stack.out_layers.0.weight": "blocks.24.norm2.weight",
            "model.diffusion_model.input_blocks.4.0.time_stack.out_layers.3.bias": "blocks.24.conv2.bias",
            "model.diffusion_model.input_blocks.4.0.time_stack.out_layers.3.weight": "blocks.24.conv2.weight",
            "model.diffusion_model.input_blocks.4.1.norm.bias": "blocks.27.norm.bias",
            "model.diffusion_model.input_blocks.4.1.norm.weight": "blocks.27.norm.weight",
            "model.diffusion_model.input_blocks.4.1.proj_in.bias": "blocks.27.proj_in.bias",
            "model.diffusion_model.input_blocks.4.1.proj_in.weight": "blocks.27.proj_in.weight",
            "model.diffusion_model.input_blocks.4.1.proj_out.bias": "blocks.30.proj.bias",
            "model.diffusion_model.input_blocks.4.1.proj_out.weight": "blocks.30.proj.weight",
            "model.diffusion_model.input_blocks.4.1.time_mixer.mix_factor": "blocks.30.mix_factor",
            "model.diffusion_model.input_blocks.4.1.time_pos_embed.0.bias": "blocks.29.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.4.1.time_pos_embed.0.weight": "blocks.29.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.4.1.time_pos_embed.2.bias": "blocks.29.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.4.1.time_pos_embed.2.weight": "blocks.29.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn1.to_k.weight": "blocks.29.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn1.to_out.0.bias": "blocks.29.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn1.to_out.0.weight": "blocks.29.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn1.to_q.weight": "blocks.29.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn1.to_v.weight": "blocks.29.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn2.to_k.weight": "blocks.29.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn2.to_out.0.bias": "blocks.29.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn2.to_out.0.weight": "blocks.29.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn2.to_q.weight": "blocks.29.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.attn2.to_v.weight": "blocks.29.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff.net.0.proj.bias": "blocks.29.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff.net.0.proj.weight": "blocks.29.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff.net.2.bias": "blocks.29.ff_out.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff.net.2.weight": "blocks.29.ff_out.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.29.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.29.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff_in.net.2.bias": "blocks.29.ff_in.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.ff_in.net.2.weight": "blocks.29.ff_in.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm1.bias": "blocks.29.norm1.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm1.weight": "blocks.29.norm1.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm2.bias": "blocks.29.norm2.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm2.weight": "blocks.29.norm2.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm3.bias": "blocks.29.norm_out.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm3.weight": "blocks.29.norm_out.weight",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm_in.bias": "blocks.29.norm_in.bias",
            "model.diffusion_model.input_blocks.4.1.time_stack.0.norm_in.weight": "blocks.29.norm_in.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "blocks.27.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.27.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.27.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "blocks.27.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "blocks.27.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "blocks.27.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.27.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.27.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "blocks.27.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "blocks.27.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.27.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.27.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "blocks.27.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "blocks.27.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias": "blocks.27.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight": "blocks.27.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias": "blocks.27.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight": "blocks.27.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias": "blocks.27.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight": "blocks.27.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.5.0.emb_layers.1.bias": "blocks.32.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.5.0.emb_layers.1.weight": "blocks.32.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.5.0.in_layers.0.bias": "blocks.32.norm1.bias",
            "model.diffusion_model.input_blocks.5.0.in_layers.0.weight": "blocks.32.norm1.weight",
            "model.diffusion_model.input_blocks.5.0.in_layers.2.bias": "blocks.32.conv1.bias",
            "model.diffusion_model.input_blocks.5.0.in_layers.2.weight": "blocks.32.conv1.weight",
            "model.diffusion_model.input_blocks.5.0.out_layers.0.bias": "blocks.32.norm2.bias",
            "model.diffusion_model.input_blocks.5.0.out_layers.0.weight": "blocks.32.norm2.weight",
            "model.diffusion_model.input_blocks.5.0.out_layers.3.bias": "blocks.32.conv2.bias",
            "model.diffusion_model.input_blocks.5.0.out_layers.3.weight": "blocks.32.conv2.weight",
            "model.diffusion_model.input_blocks.5.0.time_mixer.mix_factor": "blocks.35.mix_factor",
            "model.diffusion_model.input_blocks.5.0.time_stack.emb_layers.1.bias": "blocks.34.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.5.0.time_stack.emb_layers.1.weight": "blocks.34.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.5.0.time_stack.in_layers.0.bias": "blocks.34.norm1.bias",
            "model.diffusion_model.input_blocks.5.0.time_stack.in_layers.0.weight": "blocks.34.norm1.weight",
            "model.diffusion_model.input_blocks.5.0.time_stack.in_layers.2.bias": "blocks.34.conv1.bias",
            "model.diffusion_model.input_blocks.5.0.time_stack.in_layers.2.weight": "blocks.34.conv1.weight",
            "model.diffusion_model.input_blocks.5.0.time_stack.out_layers.0.bias": "blocks.34.norm2.bias",
            "model.diffusion_model.input_blocks.5.0.time_stack.out_layers.0.weight": "blocks.34.norm2.weight",
            "model.diffusion_model.input_blocks.5.0.time_stack.out_layers.3.bias": "blocks.34.conv2.bias",
            "model.diffusion_model.input_blocks.5.0.time_stack.out_layers.3.weight": "blocks.34.conv2.weight",
            "model.diffusion_model.input_blocks.5.1.norm.bias": "blocks.37.norm.bias",
            "model.diffusion_model.input_blocks.5.1.norm.weight": "blocks.37.norm.weight",
            "model.diffusion_model.input_blocks.5.1.proj_in.bias": "blocks.37.proj_in.bias",
            "model.diffusion_model.input_blocks.5.1.proj_in.weight": "blocks.37.proj_in.weight",
            "model.diffusion_model.input_blocks.5.1.proj_out.bias": "blocks.40.proj.bias",
            "model.diffusion_model.input_blocks.5.1.proj_out.weight": "blocks.40.proj.weight",
            "model.diffusion_model.input_blocks.5.1.time_mixer.mix_factor": "blocks.40.mix_factor",
            "model.diffusion_model.input_blocks.5.1.time_pos_embed.0.bias": "blocks.39.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.5.1.time_pos_embed.0.weight": "blocks.39.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.5.1.time_pos_embed.2.bias": "blocks.39.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.5.1.time_pos_embed.2.weight": "blocks.39.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn1.to_k.weight": "blocks.39.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn1.to_out.0.bias": "blocks.39.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn1.to_out.0.weight": "blocks.39.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn1.to_q.weight": "blocks.39.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn1.to_v.weight": "blocks.39.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn2.to_k.weight": "blocks.39.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn2.to_out.0.bias": "blocks.39.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn2.to_out.0.weight": "blocks.39.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn2.to_q.weight": "blocks.39.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.attn2.to_v.weight": "blocks.39.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff.net.0.proj.bias": "blocks.39.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff.net.0.proj.weight": "blocks.39.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff.net.2.bias": "blocks.39.ff_out.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff.net.2.weight": "blocks.39.ff_out.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.39.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.39.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff_in.net.2.bias": "blocks.39.ff_in.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.ff_in.net.2.weight": "blocks.39.ff_in.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm1.bias": "blocks.39.norm1.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm1.weight": "blocks.39.norm1.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm2.bias": "blocks.39.norm2.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm2.weight": "blocks.39.norm2.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm3.bias": "blocks.39.norm_out.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm3.weight": "blocks.39.norm_out.weight",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm_in.bias": "blocks.39.norm_in.bias",
            "model.diffusion_model.input_blocks.5.1.time_stack.0.norm_in.weight": "blocks.39.norm_in.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "blocks.37.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.37.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.37.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "blocks.37.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "blocks.37.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "blocks.37.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.37.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.37.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "blocks.37.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "blocks.37.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.37.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.37.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "blocks.37.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "blocks.37.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias": "blocks.37.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight": "blocks.37.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias": "blocks.37.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight": "blocks.37.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias": "blocks.37.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight": "blocks.37.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.6.0.op.bias": "blocks.42.conv.bias",
            "model.diffusion_model.input_blocks.6.0.op.weight": "blocks.42.conv.weight",
            "model.diffusion_model.input_blocks.7.0.emb_layers.1.bias": "blocks.44.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.7.0.emb_layers.1.weight": "blocks.44.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.7.0.in_layers.0.bias": "blocks.44.norm1.bias",
            "model.diffusion_model.input_blocks.7.0.in_layers.0.weight": "blocks.44.norm1.weight",
            "model.diffusion_model.input_blocks.7.0.in_layers.2.bias": "blocks.44.conv1.bias",
            "model.diffusion_model.input_blocks.7.0.in_layers.2.weight": "blocks.44.conv1.weight",
            "model.diffusion_model.input_blocks.7.0.out_layers.0.bias": "blocks.44.norm2.bias",
            "model.diffusion_model.input_blocks.7.0.out_layers.0.weight": "blocks.44.norm2.weight",
            "model.diffusion_model.input_blocks.7.0.out_layers.3.bias": "blocks.44.conv2.bias",
            "model.diffusion_model.input_blocks.7.0.out_layers.3.weight": "blocks.44.conv2.weight",
            "model.diffusion_model.input_blocks.7.0.skip_connection.bias": "blocks.44.conv_shortcut.bias",
            "model.diffusion_model.input_blocks.7.0.skip_connection.weight": "blocks.44.conv_shortcut.weight",
            "model.diffusion_model.input_blocks.7.0.time_mixer.mix_factor": "blocks.47.mix_factor",
            "model.diffusion_model.input_blocks.7.0.time_stack.emb_layers.1.bias": "blocks.46.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.7.0.time_stack.emb_layers.1.weight": "blocks.46.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.7.0.time_stack.in_layers.0.bias": "blocks.46.norm1.bias",
            "model.diffusion_model.input_blocks.7.0.time_stack.in_layers.0.weight": "blocks.46.norm1.weight",
            "model.diffusion_model.input_blocks.7.0.time_stack.in_layers.2.bias": "blocks.46.conv1.bias",
            "model.diffusion_model.input_blocks.7.0.time_stack.in_layers.2.weight": "blocks.46.conv1.weight",
            "model.diffusion_model.input_blocks.7.0.time_stack.out_layers.0.bias": "blocks.46.norm2.bias",
            "model.diffusion_model.input_blocks.7.0.time_stack.out_layers.0.weight": "blocks.46.norm2.weight",
            "model.diffusion_model.input_blocks.7.0.time_stack.out_layers.3.bias": "blocks.46.conv2.bias",
            "model.diffusion_model.input_blocks.7.0.time_stack.out_layers.3.weight": "blocks.46.conv2.weight",
            "model.diffusion_model.input_blocks.7.1.norm.bias": "blocks.49.norm.bias",
            "model.diffusion_model.input_blocks.7.1.norm.weight": "blocks.49.norm.weight",
            "model.diffusion_model.input_blocks.7.1.proj_in.bias": "blocks.49.proj_in.bias",
            "model.diffusion_model.input_blocks.7.1.proj_in.weight": "blocks.49.proj_in.weight",
            "model.diffusion_model.input_blocks.7.1.proj_out.bias": "blocks.52.proj.bias",
            "model.diffusion_model.input_blocks.7.1.proj_out.weight": "blocks.52.proj.weight",
            "model.diffusion_model.input_blocks.7.1.time_mixer.mix_factor": "blocks.52.mix_factor",
            "model.diffusion_model.input_blocks.7.1.time_pos_embed.0.bias": "blocks.51.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.7.1.time_pos_embed.0.weight": "blocks.51.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.7.1.time_pos_embed.2.bias": "blocks.51.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.7.1.time_pos_embed.2.weight": "blocks.51.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn1.to_k.weight": "blocks.51.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn1.to_out.0.bias": "blocks.51.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn1.to_out.0.weight": "blocks.51.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn1.to_q.weight": "blocks.51.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn1.to_v.weight": "blocks.51.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn2.to_k.weight": "blocks.51.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn2.to_out.0.bias": "blocks.51.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn2.to_out.0.weight": "blocks.51.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn2.to_q.weight": "blocks.51.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.attn2.to_v.weight": "blocks.51.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff.net.0.proj.bias": "blocks.51.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff.net.0.proj.weight": "blocks.51.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff.net.2.bias": "blocks.51.ff_out.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff.net.2.weight": "blocks.51.ff_out.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.51.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.51.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff_in.net.2.bias": "blocks.51.ff_in.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.ff_in.net.2.weight": "blocks.51.ff_in.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm1.bias": "blocks.51.norm1.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm1.weight": "blocks.51.norm1.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm2.bias": "blocks.51.norm2.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm2.weight": "blocks.51.norm2.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm3.bias": "blocks.51.norm_out.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm3.weight": "blocks.51.norm_out.weight",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm_in.bias": "blocks.51.norm_in.bias",
            "model.diffusion_model.input_blocks.7.1.time_stack.0.norm_in.weight": "blocks.51.norm_in.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "blocks.49.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.49.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.49.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "blocks.49.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "blocks.49.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "blocks.49.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.49.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.49.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "blocks.49.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "blocks.49.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.49.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.49.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "blocks.49.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "blocks.49.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias": "blocks.49.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight": "blocks.49.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias": "blocks.49.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight": "blocks.49.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias": "blocks.49.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight": "blocks.49.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.8.0.emb_layers.1.bias": "blocks.54.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.8.0.emb_layers.1.weight": "blocks.54.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.8.0.in_layers.0.bias": "blocks.54.norm1.bias",
            "model.diffusion_model.input_blocks.8.0.in_layers.0.weight": "blocks.54.norm1.weight",
            "model.diffusion_model.input_blocks.8.0.in_layers.2.bias": "blocks.54.conv1.bias",
            "model.diffusion_model.input_blocks.8.0.in_layers.2.weight": "blocks.54.conv1.weight",
            "model.diffusion_model.input_blocks.8.0.out_layers.0.bias": "blocks.54.norm2.bias",
            "model.diffusion_model.input_blocks.8.0.out_layers.0.weight": "blocks.54.norm2.weight",
            "model.diffusion_model.input_blocks.8.0.out_layers.3.bias": "blocks.54.conv2.bias",
            "model.diffusion_model.input_blocks.8.0.out_layers.3.weight": "blocks.54.conv2.weight",
            "model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor": "blocks.57.mix_factor",
            "model.diffusion_model.input_blocks.8.0.time_stack.emb_layers.1.bias": "blocks.56.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.8.0.time_stack.emb_layers.1.weight": "blocks.56.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.8.0.time_stack.in_layers.0.bias": "blocks.56.norm1.bias",
            "model.diffusion_model.input_blocks.8.0.time_stack.in_layers.0.weight": "blocks.56.norm1.weight",
            "model.diffusion_model.input_blocks.8.0.time_stack.in_layers.2.bias": "blocks.56.conv1.bias",
            "model.diffusion_model.input_blocks.8.0.time_stack.in_layers.2.weight": "blocks.56.conv1.weight",
            "model.diffusion_model.input_blocks.8.0.time_stack.out_layers.0.bias": "blocks.56.norm2.bias",
            "model.diffusion_model.input_blocks.8.0.time_stack.out_layers.0.weight": "blocks.56.norm2.weight",
            "model.diffusion_model.input_blocks.8.0.time_stack.out_layers.3.bias": "blocks.56.conv2.bias",
            "model.diffusion_model.input_blocks.8.0.time_stack.out_layers.3.weight": "blocks.56.conv2.weight",
            "model.diffusion_model.input_blocks.8.1.norm.bias": "blocks.59.norm.bias",
            "model.diffusion_model.input_blocks.8.1.norm.weight": "blocks.59.norm.weight",
            "model.diffusion_model.input_blocks.8.1.proj_in.bias": "blocks.59.proj_in.bias",
            "model.diffusion_model.input_blocks.8.1.proj_in.weight": "blocks.59.proj_in.weight",
            "model.diffusion_model.input_blocks.8.1.proj_out.bias": "blocks.62.proj.bias",
            "model.diffusion_model.input_blocks.8.1.proj_out.weight": "blocks.62.proj.weight",
            "model.diffusion_model.input_blocks.8.1.time_mixer.mix_factor": "blocks.62.mix_factor",
            "model.diffusion_model.input_blocks.8.1.time_pos_embed.0.bias": "blocks.61.positional_embedding_proj.0.bias",
            "model.diffusion_model.input_blocks.8.1.time_pos_embed.0.weight": "blocks.61.positional_embedding_proj.0.weight",
            "model.diffusion_model.input_blocks.8.1.time_pos_embed.2.bias": "blocks.61.positional_embedding_proj.2.bias",
            "model.diffusion_model.input_blocks.8.1.time_pos_embed.2.weight": "blocks.61.positional_embedding_proj.2.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn1.to_k.weight": "blocks.61.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn1.to_out.0.bias": "blocks.61.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn1.to_out.0.weight": "blocks.61.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn1.to_q.weight": "blocks.61.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn1.to_v.weight": "blocks.61.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn2.to_k.weight": "blocks.61.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn2.to_out.0.bias": "blocks.61.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn2.to_out.0.weight": "blocks.61.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn2.to_q.weight": "blocks.61.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.attn2.to_v.weight": "blocks.61.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff.net.0.proj.bias": "blocks.61.act_fn_out.proj.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff.net.0.proj.weight": "blocks.61.act_fn_out.proj.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff.net.2.bias": "blocks.61.ff_out.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff.net.2.weight": "blocks.61.ff_out.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.61.act_fn_in.proj.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.61.act_fn_in.proj.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff_in.net.2.bias": "blocks.61.ff_in.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.ff_in.net.2.weight": "blocks.61.ff_in.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm1.bias": "blocks.61.norm1.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm1.weight": "blocks.61.norm1.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm2.bias": "blocks.61.norm2.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm2.weight": "blocks.61.norm2.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm3.bias": "blocks.61.norm_out.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm3.weight": "blocks.61.norm_out.weight",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm_in.bias": "blocks.61.norm_in.bias",
            "model.diffusion_model.input_blocks.8.1.time_stack.0.norm_in.weight": "blocks.61.norm_in.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "blocks.59.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.59.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.59.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "blocks.59.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "blocks.59.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "blocks.59.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.59.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.59.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "blocks.59.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "blocks.59.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.59.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.59.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "blocks.59.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "blocks.59.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias": "blocks.59.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight": "blocks.59.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias": "blocks.59.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight": "blocks.59.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias": "blocks.59.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight": "blocks.59.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.9.0.op.bias": "blocks.64.conv.bias",
            "model.diffusion_model.input_blocks.9.0.op.weight": "blocks.64.conv.weight",
            "model.diffusion_model.label_emb.0.0.bias": "add_time_embedding.0.bias",
            "model.diffusion_model.label_emb.0.0.weight": "add_time_embedding.0.weight",
            "model.diffusion_model.label_emb.0.2.bias": "add_time_embedding.2.bias",
            "model.diffusion_model.label_emb.0.2.weight": "add_time_embedding.2.weight",
            "model.diffusion_model.middle_block.0.emb_layers.1.bias": "blocks.76.time_emb_proj.bias",
            "model.diffusion_model.middle_block.0.emb_layers.1.weight": "blocks.76.time_emb_proj.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.bias": "blocks.76.norm1.bias",
            "model.diffusion_model.middle_block.0.in_layers.0.weight": "blocks.76.norm1.weight",
            "model.diffusion_model.middle_block.0.in_layers.2.bias": "blocks.76.conv1.bias",
            "model.diffusion_model.middle_block.0.in_layers.2.weight": "blocks.76.conv1.weight",
            "model.diffusion_model.middle_block.0.out_layers.0.bias": "blocks.76.norm2.bias",
            "model.diffusion_model.middle_block.0.out_layers.0.weight": "blocks.76.norm2.weight",
            "model.diffusion_model.middle_block.0.out_layers.3.bias": "blocks.76.conv2.bias",
            "model.diffusion_model.middle_block.0.out_layers.3.weight": "blocks.76.conv2.weight",
            "model.diffusion_model.middle_block.0.time_mixer.mix_factor": "blocks.79.mix_factor",
            "model.diffusion_model.middle_block.0.time_stack.emb_layers.1.bias": "blocks.78.time_emb_proj.bias",
            "model.diffusion_model.middle_block.0.time_stack.emb_layers.1.weight": "blocks.78.time_emb_proj.weight",
            "model.diffusion_model.middle_block.0.time_stack.in_layers.0.bias": "blocks.78.norm1.bias",
            "model.diffusion_model.middle_block.0.time_stack.in_layers.0.weight": "blocks.78.norm1.weight",
            "model.diffusion_model.middle_block.0.time_stack.in_layers.2.bias": "blocks.78.conv1.bias",
            "model.diffusion_model.middle_block.0.time_stack.in_layers.2.weight": "blocks.78.conv1.weight",
            "model.diffusion_model.middle_block.0.time_stack.out_layers.0.bias": "blocks.78.norm2.bias",
            "model.diffusion_model.middle_block.0.time_stack.out_layers.0.weight": "blocks.78.norm2.weight",
            "model.diffusion_model.middle_block.0.time_stack.out_layers.3.bias": "blocks.78.conv2.bias",
            "model.diffusion_model.middle_block.0.time_stack.out_layers.3.weight": "blocks.78.conv2.weight",
            "model.diffusion_model.middle_block.1.norm.bias": "blocks.81.norm.bias",
            "model.diffusion_model.middle_block.1.norm.weight": "blocks.81.norm.weight",
            "model.diffusion_model.middle_block.1.proj_in.bias": "blocks.81.proj_in.bias",
            "model.diffusion_model.middle_block.1.proj_in.weight": "blocks.81.proj_in.weight",
            "model.diffusion_model.middle_block.1.proj_out.bias": "blocks.84.proj.bias",
            "model.diffusion_model.middle_block.1.proj_out.weight": "blocks.84.proj.weight",
            "model.diffusion_model.middle_block.1.time_mixer.mix_factor": "blocks.84.mix_factor",
            "model.diffusion_model.middle_block.1.time_pos_embed.0.bias": "blocks.83.positional_embedding_proj.0.bias",
            "model.diffusion_model.middle_block.1.time_pos_embed.0.weight": "blocks.83.positional_embedding_proj.0.weight",
            "model.diffusion_model.middle_block.1.time_pos_embed.2.bias": "blocks.83.positional_embedding_proj.2.bias",
            "model.diffusion_model.middle_block.1.time_pos_embed.2.weight": "blocks.83.positional_embedding_proj.2.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn1.to_k.weight": "blocks.83.attn1.to_k.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn1.to_out.0.bias": "blocks.83.attn1.to_out.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.attn1.to_out.0.weight": "blocks.83.attn1.to_out.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn1.to_q.weight": "blocks.83.attn1.to_q.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn1.to_v.weight": "blocks.83.attn1.to_v.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn2.to_k.weight": "blocks.83.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn2.to_out.0.bias": "blocks.83.attn2.to_out.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.attn2.to_out.0.weight": "blocks.83.attn2.to_out.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn2.to_q.weight": "blocks.83.attn2.to_q.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.attn2.to_v.weight": "blocks.83.attn2.to_v.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.ff.net.0.proj.bias": "blocks.83.act_fn_out.proj.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.ff.net.0.proj.weight": "blocks.83.act_fn_out.proj.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.ff.net.2.bias": "blocks.83.ff_out.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.ff.net.2.weight": "blocks.83.ff_out.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.83.act_fn_in.proj.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.83.act_fn_in.proj.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.ff_in.net.2.bias": "blocks.83.ff_in.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.ff_in.net.2.weight": "blocks.83.ff_in.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.norm1.bias": "blocks.83.norm1.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.norm1.weight": "blocks.83.norm1.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.norm2.bias": "blocks.83.norm2.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.norm2.weight": "blocks.83.norm2.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.norm3.bias": "blocks.83.norm_out.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.norm3.weight": "blocks.83.norm_out.weight",
            "model.diffusion_model.middle_block.1.time_stack.0.norm_in.bias": "blocks.83.norm_in.bias",
            "model.diffusion_model.middle_block.1.time_stack.0.norm_in.weight": "blocks.83.norm_in.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight": "blocks.81.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.81.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.81.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": "blocks.81.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight": "blocks.81.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": "blocks.81.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.81.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.81.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight": "blocks.81.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": "blocks.81.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.81.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.81.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias": "blocks.81.transformer_blocks.0.ff.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight": "blocks.81.transformer_blocks.0.ff.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias": "blocks.81.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight": "blocks.81.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias": "blocks.81.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight": "blocks.81.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias": "blocks.81.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight": "blocks.81.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.middle_block.2.emb_layers.1.bias": "blocks.85.time_emb_proj.bias",
            "model.diffusion_model.middle_block.2.emb_layers.1.weight": "blocks.85.time_emb_proj.weight",
            "model.diffusion_model.middle_block.2.in_layers.0.bias": "blocks.85.norm1.bias",
            "model.diffusion_model.middle_block.2.in_layers.0.weight": "blocks.85.norm1.weight",
            "model.diffusion_model.middle_block.2.in_layers.2.bias": "blocks.85.conv1.bias",
            "model.diffusion_model.middle_block.2.in_layers.2.weight": "blocks.85.conv1.weight",
            "model.diffusion_model.middle_block.2.out_layers.0.bias": "blocks.85.norm2.bias",
            "model.diffusion_model.middle_block.2.out_layers.0.weight": "blocks.85.norm2.weight",
            "model.diffusion_model.middle_block.2.out_layers.3.bias": "blocks.85.conv2.bias",
            "model.diffusion_model.middle_block.2.out_layers.3.weight": "blocks.85.conv2.weight",
            "model.diffusion_model.middle_block.2.time_mixer.mix_factor": "blocks.88.mix_factor",
            "model.diffusion_model.middle_block.2.time_stack.emb_layers.1.bias": "blocks.87.time_emb_proj.bias",
            "model.diffusion_model.middle_block.2.time_stack.emb_layers.1.weight": "blocks.87.time_emb_proj.weight",
            "model.diffusion_model.middle_block.2.time_stack.in_layers.0.bias": "blocks.87.norm1.bias",
            "model.diffusion_model.middle_block.2.time_stack.in_layers.0.weight": "blocks.87.norm1.weight",
            "model.diffusion_model.middle_block.2.time_stack.in_layers.2.bias": "blocks.87.conv1.bias",
            "model.diffusion_model.middle_block.2.time_stack.in_layers.2.weight": "blocks.87.conv1.weight",
            "model.diffusion_model.middle_block.2.time_stack.out_layers.0.bias": "blocks.87.norm2.bias",
            "model.diffusion_model.middle_block.2.time_stack.out_layers.0.weight": "blocks.87.norm2.weight",
            "model.diffusion_model.middle_block.2.time_stack.out_layers.3.bias": "blocks.87.conv2.bias",
            "model.diffusion_model.middle_block.2.time_stack.out_layers.3.weight": "blocks.87.conv2.weight",
            "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
            "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
            "model.diffusion_model.out.2.bias": "conv_out.bias",
            "model.diffusion_model.out.2.weight": "conv_out.weight",
            "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias": "blocks.90.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight": "blocks.90.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.0.0.in_layers.0.bias": "blocks.90.norm1.bias",
            "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": "blocks.90.norm1.weight",
            "model.diffusion_model.output_blocks.0.0.in_layers.2.bias": "blocks.90.conv1.bias",
            "model.diffusion_model.output_blocks.0.0.in_layers.2.weight": "blocks.90.conv1.weight",
            "model.diffusion_model.output_blocks.0.0.out_layers.0.bias": "blocks.90.norm2.bias",
            "model.diffusion_model.output_blocks.0.0.out_layers.0.weight": "blocks.90.norm2.weight",
            "model.diffusion_model.output_blocks.0.0.out_layers.3.bias": "blocks.90.conv2.bias",
            "model.diffusion_model.output_blocks.0.0.out_layers.3.weight": "blocks.90.conv2.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.bias": "blocks.90.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight": "blocks.90.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.0.0.time_mixer.mix_factor": "blocks.93.mix_factor",
            "model.diffusion_model.output_blocks.0.0.time_stack.emb_layers.1.bias": "blocks.92.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.0.0.time_stack.emb_layers.1.weight": "blocks.92.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.0.0.time_stack.in_layers.0.bias": "blocks.92.norm1.bias",
            "model.diffusion_model.output_blocks.0.0.time_stack.in_layers.0.weight": "blocks.92.norm1.weight",
            "model.diffusion_model.output_blocks.0.0.time_stack.in_layers.2.bias": "blocks.92.conv1.bias",
            "model.diffusion_model.output_blocks.0.0.time_stack.in_layers.2.weight": "blocks.92.conv1.weight",
            "model.diffusion_model.output_blocks.0.0.time_stack.out_layers.0.bias": "blocks.92.norm2.bias",
            "model.diffusion_model.output_blocks.0.0.time_stack.out_layers.0.weight": "blocks.92.norm2.weight",
            "model.diffusion_model.output_blocks.0.0.time_stack.out_layers.3.bias": "blocks.92.conv2.bias",
            "model.diffusion_model.output_blocks.0.0.time_stack.out_layers.3.weight": "blocks.92.conv2.weight",
            "model.diffusion_model.output_blocks.1.0.emb_layers.1.bias": "blocks.95.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.1.0.emb_layers.1.weight": "blocks.95.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.1.0.in_layers.0.bias": "blocks.95.norm1.bias",
            "model.diffusion_model.output_blocks.1.0.in_layers.0.weight": "blocks.95.norm1.weight",
            "model.diffusion_model.output_blocks.1.0.in_layers.2.bias": "blocks.95.conv1.bias",
            "model.diffusion_model.output_blocks.1.0.in_layers.2.weight": "blocks.95.conv1.weight",
            "model.diffusion_model.output_blocks.1.0.out_layers.0.bias": "blocks.95.norm2.bias",
            "model.diffusion_model.output_blocks.1.0.out_layers.0.weight": "blocks.95.norm2.weight",
            "model.diffusion_model.output_blocks.1.0.out_layers.3.bias": "blocks.95.conv2.bias",
            "model.diffusion_model.output_blocks.1.0.out_layers.3.weight": "blocks.95.conv2.weight",
            "model.diffusion_model.output_blocks.1.0.skip_connection.bias": "blocks.95.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.1.0.skip_connection.weight": "blocks.95.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.1.0.time_mixer.mix_factor": "blocks.98.mix_factor",
            "model.diffusion_model.output_blocks.1.0.time_stack.emb_layers.1.bias": "blocks.97.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.1.0.time_stack.emb_layers.1.weight": "blocks.97.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.1.0.time_stack.in_layers.0.bias": "blocks.97.norm1.bias",
            "model.diffusion_model.output_blocks.1.0.time_stack.in_layers.0.weight": "blocks.97.norm1.weight",
            "model.diffusion_model.output_blocks.1.0.time_stack.in_layers.2.bias": "blocks.97.conv1.bias",
            "model.diffusion_model.output_blocks.1.0.time_stack.in_layers.2.weight": "blocks.97.conv1.weight",
            "model.diffusion_model.output_blocks.1.0.time_stack.out_layers.0.bias": "blocks.97.norm2.bias",
            "model.diffusion_model.output_blocks.1.0.time_stack.out_layers.0.weight": "blocks.97.norm2.weight",
            "model.diffusion_model.output_blocks.1.0.time_stack.out_layers.3.bias": "blocks.97.conv2.bias",
            "model.diffusion_model.output_blocks.1.0.time_stack.out_layers.3.weight": "blocks.97.conv2.weight",
            "model.diffusion_model.output_blocks.10.0.emb_layers.1.bias": "blocks.178.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.10.0.emb_layers.1.weight": "blocks.178.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.10.0.in_layers.0.bias": "blocks.178.norm1.bias",
            "model.diffusion_model.output_blocks.10.0.in_layers.0.weight": "blocks.178.norm1.weight",
            "model.diffusion_model.output_blocks.10.0.in_layers.2.bias": "blocks.178.conv1.bias",
            "model.diffusion_model.output_blocks.10.0.in_layers.2.weight": "blocks.178.conv1.weight",
            "model.diffusion_model.output_blocks.10.0.out_layers.0.bias": "blocks.178.norm2.bias",
            "model.diffusion_model.output_blocks.10.0.out_layers.0.weight": "blocks.178.norm2.weight",
            "model.diffusion_model.output_blocks.10.0.out_layers.3.bias": "blocks.178.conv2.bias",
            "model.diffusion_model.output_blocks.10.0.out_layers.3.weight": "blocks.178.conv2.weight",
            "model.diffusion_model.output_blocks.10.0.skip_connection.bias": "blocks.178.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.10.0.skip_connection.weight": "blocks.178.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.10.0.time_mixer.mix_factor": "blocks.181.mix_factor",
            "model.diffusion_model.output_blocks.10.0.time_stack.emb_layers.1.bias": "blocks.180.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.10.0.time_stack.emb_layers.1.weight": "blocks.180.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.10.0.time_stack.in_layers.0.bias": "blocks.180.norm1.bias",
            "model.diffusion_model.output_blocks.10.0.time_stack.in_layers.0.weight": "blocks.180.norm1.weight",
            "model.diffusion_model.output_blocks.10.0.time_stack.in_layers.2.bias": "blocks.180.conv1.bias",
            "model.diffusion_model.output_blocks.10.0.time_stack.in_layers.2.weight": "blocks.180.conv1.weight",
            "model.diffusion_model.output_blocks.10.0.time_stack.out_layers.0.bias": "blocks.180.norm2.bias",
            "model.diffusion_model.output_blocks.10.0.time_stack.out_layers.0.weight": "blocks.180.norm2.weight",
            "model.diffusion_model.output_blocks.10.0.time_stack.out_layers.3.bias": "blocks.180.conv2.bias",
            "model.diffusion_model.output_blocks.10.0.time_stack.out_layers.3.weight": "blocks.180.conv2.weight",
            "model.diffusion_model.output_blocks.10.1.norm.bias": "blocks.183.norm.bias",
            "model.diffusion_model.output_blocks.10.1.norm.weight": "blocks.183.norm.weight",
            "model.diffusion_model.output_blocks.10.1.proj_in.bias": "blocks.183.proj_in.bias",
            "model.diffusion_model.output_blocks.10.1.proj_in.weight": "blocks.183.proj_in.weight",
            "model.diffusion_model.output_blocks.10.1.proj_out.bias": "blocks.186.proj.bias",
            "model.diffusion_model.output_blocks.10.1.proj_out.weight": "blocks.186.proj.weight",
            "model.diffusion_model.output_blocks.10.1.time_mixer.mix_factor": "blocks.186.mix_factor",
            "model.diffusion_model.output_blocks.10.1.time_pos_embed.0.bias": "blocks.185.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.10.1.time_pos_embed.0.weight": "blocks.185.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.10.1.time_pos_embed.2.bias": "blocks.185.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.10.1.time_pos_embed.2.weight": "blocks.185.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn1.to_k.weight": "blocks.185.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn1.to_out.0.bias": "blocks.185.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn1.to_out.0.weight": "blocks.185.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn1.to_q.weight": "blocks.185.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn1.to_v.weight": "blocks.185.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn2.to_k.weight": "blocks.185.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn2.to_out.0.bias": "blocks.185.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn2.to_out.0.weight": "blocks.185.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn2.to_q.weight": "blocks.185.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.attn2.to_v.weight": "blocks.185.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff.net.0.proj.bias": "blocks.185.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff.net.0.proj.weight": "blocks.185.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff.net.2.bias": "blocks.185.ff_out.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff.net.2.weight": "blocks.185.ff_out.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.185.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.185.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff_in.net.2.bias": "blocks.185.ff_in.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.ff_in.net.2.weight": "blocks.185.ff_in.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm1.bias": "blocks.185.norm1.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm1.weight": "blocks.185.norm1.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm2.bias": "blocks.185.norm2.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm2.weight": "blocks.185.norm2.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm3.bias": "blocks.185.norm_out.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm3.weight": "blocks.185.norm_out.weight",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm_in.bias": "blocks.185.norm_in.bias",
            "model.diffusion_model.output_blocks.10.1.time_stack.0.norm_in.weight": "blocks.185.norm_in.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight": "blocks.183.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.183.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.183.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight": "blocks.183.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight": "blocks.183.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight": "blocks.183.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.183.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.183.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight": "blocks.183.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight": "blocks.183.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.183.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.183.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias": "blocks.183.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight": "blocks.183.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias": "blocks.183.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight": "blocks.183.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias": "blocks.183.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight": "blocks.183.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias": "blocks.183.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight": "blocks.183.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.11.0.emb_layers.1.bias": "blocks.188.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.11.0.emb_layers.1.weight": "blocks.188.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.11.0.in_layers.0.bias": "blocks.188.norm1.bias",
            "model.diffusion_model.output_blocks.11.0.in_layers.0.weight": "blocks.188.norm1.weight",
            "model.diffusion_model.output_blocks.11.0.in_layers.2.bias": "blocks.188.conv1.bias",
            "model.diffusion_model.output_blocks.11.0.in_layers.2.weight": "blocks.188.conv1.weight",
            "model.diffusion_model.output_blocks.11.0.out_layers.0.bias": "blocks.188.norm2.bias",
            "model.diffusion_model.output_blocks.11.0.out_layers.0.weight": "blocks.188.norm2.weight",
            "model.diffusion_model.output_blocks.11.0.out_layers.3.bias": "blocks.188.conv2.bias",
            "model.diffusion_model.output_blocks.11.0.out_layers.3.weight": "blocks.188.conv2.weight",
            "model.diffusion_model.output_blocks.11.0.skip_connection.bias": "blocks.188.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.11.0.skip_connection.weight": "blocks.188.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.11.0.time_mixer.mix_factor": "blocks.191.mix_factor",
            "model.diffusion_model.output_blocks.11.0.time_stack.emb_layers.1.bias": "blocks.190.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.11.0.time_stack.emb_layers.1.weight": "blocks.190.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.11.0.time_stack.in_layers.0.bias": "blocks.190.norm1.bias",
            "model.diffusion_model.output_blocks.11.0.time_stack.in_layers.0.weight": "blocks.190.norm1.weight",
            "model.diffusion_model.output_blocks.11.0.time_stack.in_layers.2.bias": "blocks.190.conv1.bias",
            "model.diffusion_model.output_blocks.11.0.time_stack.in_layers.2.weight": "blocks.190.conv1.weight",
            "model.diffusion_model.output_blocks.11.0.time_stack.out_layers.0.bias": "blocks.190.norm2.bias",
            "model.diffusion_model.output_blocks.11.0.time_stack.out_layers.0.weight": "blocks.190.norm2.weight",
            "model.diffusion_model.output_blocks.11.0.time_stack.out_layers.3.bias": "blocks.190.conv2.bias",
            "model.diffusion_model.output_blocks.11.0.time_stack.out_layers.3.weight": "blocks.190.conv2.weight",
            "model.diffusion_model.output_blocks.11.1.norm.bias": "blocks.193.norm.bias",
            "model.diffusion_model.output_blocks.11.1.norm.weight": "blocks.193.norm.weight",
            "model.diffusion_model.output_blocks.11.1.proj_in.bias": "blocks.193.proj_in.bias",
            "model.diffusion_model.output_blocks.11.1.proj_in.weight": "blocks.193.proj_in.weight",
            "model.diffusion_model.output_blocks.11.1.proj_out.bias": "blocks.196.proj.bias",
            "model.diffusion_model.output_blocks.11.1.proj_out.weight": "blocks.196.proj.weight",
            "model.diffusion_model.output_blocks.11.1.time_mixer.mix_factor": "blocks.196.mix_factor",
            "model.diffusion_model.output_blocks.11.1.time_pos_embed.0.bias": "blocks.195.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.11.1.time_pos_embed.0.weight": "blocks.195.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.11.1.time_pos_embed.2.bias": "blocks.195.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.11.1.time_pos_embed.2.weight": "blocks.195.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn1.to_k.weight": "blocks.195.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn1.to_out.0.bias": "blocks.195.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn1.to_out.0.weight": "blocks.195.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn1.to_q.weight": "blocks.195.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn1.to_v.weight": "blocks.195.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn2.to_k.weight": "blocks.195.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn2.to_out.0.bias": "blocks.195.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn2.to_out.0.weight": "blocks.195.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn2.to_q.weight": "blocks.195.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.attn2.to_v.weight": "blocks.195.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff.net.0.proj.bias": "blocks.195.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff.net.0.proj.weight": "blocks.195.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff.net.2.bias": "blocks.195.ff_out.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff.net.2.weight": "blocks.195.ff_out.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.195.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.195.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff_in.net.2.bias": "blocks.195.ff_in.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.ff_in.net.2.weight": "blocks.195.ff_in.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm1.bias": "blocks.195.norm1.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm1.weight": "blocks.195.norm1.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm2.bias": "blocks.195.norm2.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm2.weight": "blocks.195.norm2.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm3.bias": "blocks.195.norm_out.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm3.weight": "blocks.195.norm_out.weight",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm_in.bias": "blocks.195.norm_in.bias",
            "model.diffusion_model.output_blocks.11.1.time_stack.0.norm_in.weight": "blocks.195.norm_in.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight": "blocks.193.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.193.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.193.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight": "blocks.193.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight": "blocks.193.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight": "blocks.193.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.193.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.193.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight": "blocks.193.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight": "blocks.193.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.193.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.193.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias": "blocks.193.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight": "blocks.193.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias": "blocks.193.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight": "blocks.193.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias": "blocks.193.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight": "blocks.193.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias": "blocks.193.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight": "blocks.193.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.2.0.emb_layers.1.bias": "blocks.100.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.2.0.emb_layers.1.weight": "blocks.100.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.2.0.in_layers.0.bias": "blocks.100.norm1.bias",
            "model.diffusion_model.output_blocks.2.0.in_layers.0.weight": "blocks.100.norm1.weight",
            "model.diffusion_model.output_blocks.2.0.in_layers.2.bias": "blocks.100.conv1.bias",
            "model.diffusion_model.output_blocks.2.0.in_layers.2.weight": "blocks.100.conv1.weight",
            "model.diffusion_model.output_blocks.2.0.out_layers.0.bias": "blocks.100.norm2.bias",
            "model.diffusion_model.output_blocks.2.0.out_layers.0.weight": "blocks.100.norm2.weight",
            "model.diffusion_model.output_blocks.2.0.out_layers.3.bias": "blocks.100.conv2.bias",
            "model.diffusion_model.output_blocks.2.0.out_layers.3.weight": "blocks.100.conv2.weight",
            "model.diffusion_model.output_blocks.2.0.skip_connection.bias": "blocks.100.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.2.0.skip_connection.weight": "blocks.100.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.2.0.time_mixer.mix_factor": "blocks.103.mix_factor",
            "model.diffusion_model.output_blocks.2.0.time_stack.emb_layers.1.bias": "blocks.102.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.2.0.time_stack.emb_layers.1.weight": "blocks.102.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.2.0.time_stack.in_layers.0.bias": "blocks.102.norm1.bias",
            "model.diffusion_model.output_blocks.2.0.time_stack.in_layers.0.weight": "blocks.102.norm1.weight",
            "model.diffusion_model.output_blocks.2.0.time_stack.in_layers.2.bias": "blocks.102.conv1.bias",
            "model.diffusion_model.output_blocks.2.0.time_stack.in_layers.2.weight": "blocks.102.conv1.weight",
            "model.diffusion_model.output_blocks.2.0.time_stack.out_layers.0.bias": "blocks.102.norm2.bias",
            "model.diffusion_model.output_blocks.2.0.time_stack.out_layers.0.weight": "blocks.102.norm2.weight",
            "model.diffusion_model.output_blocks.2.0.time_stack.out_layers.3.bias": "blocks.102.conv2.bias",
            "model.diffusion_model.output_blocks.2.0.time_stack.out_layers.3.weight": "blocks.102.conv2.weight",
            "model.diffusion_model.output_blocks.2.1.conv.bias": "blocks.104.conv.bias",
            "model.diffusion_model.output_blocks.2.1.conv.weight": "blocks.104.conv.weight",
            "model.diffusion_model.output_blocks.3.0.emb_layers.1.bias": "blocks.106.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.3.0.emb_layers.1.weight": "blocks.106.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.3.0.in_layers.0.bias": "blocks.106.norm1.bias",
            "model.diffusion_model.output_blocks.3.0.in_layers.0.weight": "blocks.106.norm1.weight",
            "model.diffusion_model.output_blocks.3.0.in_layers.2.bias": "blocks.106.conv1.bias",
            "model.diffusion_model.output_blocks.3.0.in_layers.2.weight": "blocks.106.conv1.weight",
            "model.diffusion_model.output_blocks.3.0.out_layers.0.bias": "blocks.106.norm2.bias",
            "model.diffusion_model.output_blocks.3.0.out_layers.0.weight": "blocks.106.norm2.weight",
            "model.diffusion_model.output_blocks.3.0.out_layers.3.bias": "blocks.106.conv2.bias",
            "model.diffusion_model.output_blocks.3.0.out_layers.3.weight": "blocks.106.conv2.weight",
            "model.diffusion_model.output_blocks.3.0.skip_connection.bias": "blocks.106.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.3.0.skip_connection.weight": "blocks.106.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.3.0.time_mixer.mix_factor": "blocks.109.mix_factor",
            "model.diffusion_model.output_blocks.3.0.time_stack.emb_layers.1.bias": "blocks.108.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.3.0.time_stack.emb_layers.1.weight": "blocks.108.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.3.0.time_stack.in_layers.0.bias": "blocks.108.norm1.bias",
            "model.diffusion_model.output_blocks.3.0.time_stack.in_layers.0.weight": "blocks.108.norm1.weight",
            "model.diffusion_model.output_blocks.3.0.time_stack.in_layers.2.bias": "blocks.108.conv1.bias",
            "model.diffusion_model.output_blocks.3.0.time_stack.in_layers.2.weight": "blocks.108.conv1.weight",
            "model.diffusion_model.output_blocks.3.0.time_stack.out_layers.0.bias": "blocks.108.norm2.bias",
            "model.diffusion_model.output_blocks.3.0.time_stack.out_layers.0.weight": "blocks.108.norm2.weight",
            "model.diffusion_model.output_blocks.3.0.time_stack.out_layers.3.bias": "blocks.108.conv2.bias",
            "model.diffusion_model.output_blocks.3.0.time_stack.out_layers.3.weight": "blocks.108.conv2.weight",
            "model.diffusion_model.output_blocks.3.1.norm.bias": "blocks.111.norm.bias",
            "model.diffusion_model.output_blocks.3.1.norm.weight": "blocks.111.norm.weight",
            "model.diffusion_model.output_blocks.3.1.proj_in.bias": "blocks.111.proj_in.bias",
            "model.diffusion_model.output_blocks.3.1.proj_in.weight": "blocks.111.proj_in.weight",
            "model.diffusion_model.output_blocks.3.1.proj_out.bias": "blocks.114.proj.bias",
            "model.diffusion_model.output_blocks.3.1.proj_out.weight": "blocks.114.proj.weight",
            "model.diffusion_model.output_blocks.3.1.time_mixer.mix_factor": "blocks.114.mix_factor",
            "model.diffusion_model.output_blocks.3.1.time_pos_embed.0.bias": "blocks.113.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.3.1.time_pos_embed.0.weight": "blocks.113.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.3.1.time_pos_embed.2.bias": "blocks.113.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.3.1.time_pos_embed.2.weight": "blocks.113.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn1.to_k.weight": "blocks.113.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn1.to_out.0.bias": "blocks.113.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn1.to_out.0.weight": "blocks.113.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn1.to_q.weight": "blocks.113.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn1.to_v.weight": "blocks.113.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn2.to_k.weight": "blocks.113.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn2.to_out.0.bias": "blocks.113.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn2.to_out.0.weight": "blocks.113.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn2.to_q.weight": "blocks.113.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.attn2.to_v.weight": "blocks.113.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff.net.0.proj.bias": "blocks.113.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff.net.0.proj.weight": "blocks.113.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff.net.2.bias": "blocks.113.ff_out.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff.net.2.weight": "blocks.113.ff_out.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.113.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.113.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff_in.net.2.bias": "blocks.113.ff_in.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.ff_in.net.2.weight": "blocks.113.ff_in.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm1.bias": "blocks.113.norm1.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm1.weight": "blocks.113.norm1.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm2.bias": "blocks.113.norm2.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm2.weight": "blocks.113.norm2.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm3.bias": "blocks.113.norm_out.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm3.weight": "blocks.113.norm_out.weight",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm_in.bias": "blocks.113.norm_in.bias",
            "model.diffusion_model.output_blocks.3.1.time_stack.0.norm_in.weight": "blocks.113.norm_in.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight": "blocks.111.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.111.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.111.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight": "blocks.111.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight": "blocks.111.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight": "blocks.111.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.111.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.111.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight": "blocks.111.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight": "blocks.111.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.111.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.111.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias": "blocks.111.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight": "blocks.111.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias": "blocks.111.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight": "blocks.111.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias": "blocks.111.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight": "blocks.111.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias": "blocks.111.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight": "blocks.111.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.4.0.emb_layers.1.bias": "blocks.116.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.4.0.emb_layers.1.weight": "blocks.116.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.4.0.in_layers.0.bias": "blocks.116.norm1.bias",
            "model.diffusion_model.output_blocks.4.0.in_layers.0.weight": "blocks.116.norm1.weight",
            "model.diffusion_model.output_blocks.4.0.in_layers.2.bias": "blocks.116.conv1.bias",
            "model.diffusion_model.output_blocks.4.0.in_layers.2.weight": "blocks.116.conv1.weight",
            "model.diffusion_model.output_blocks.4.0.out_layers.0.bias": "blocks.116.norm2.bias",
            "model.diffusion_model.output_blocks.4.0.out_layers.0.weight": "blocks.116.norm2.weight",
            "model.diffusion_model.output_blocks.4.0.out_layers.3.bias": "blocks.116.conv2.bias",
            "model.diffusion_model.output_blocks.4.0.out_layers.3.weight": "blocks.116.conv2.weight",
            "model.diffusion_model.output_blocks.4.0.skip_connection.bias": "blocks.116.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.4.0.skip_connection.weight": "blocks.116.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.4.0.time_mixer.mix_factor": "blocks.119.mix_factor",
            "model.diffusion_model.output_blocks.4.0.time_stack.emb_layers.1.bias": "blocks.118.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.4.0.time_stack.emb_layers.1.weight": "blocks.118.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.4.0.time_stack.in_layers.0.bias": "blocks.118.norm1.bias",
            "model.diffusion_model.output_blocks.4.0.time_stack.in_layers.0.weight": "blocks.118.norm1.weight",
            "model.diffusion_model.output_blocks.4.0.time_stack.in_layers.2.bias": "blocks.118.conv1.bias",
            "model.diffusion_model.output_blocks.4.0.time_stack.in_layers.2.weight": "blocks.118.conv1.weight",
            "model.diffusion_model.output_blocks.4.0.time_stack.out_layers.0.bias": "blocks.118.norm2.bias",
            "model.diffusion_model.output_blocks.4.0.time_stack.out_layers.0.weight": "blocks.118.norm2.weight",
            "model.diffusion_model.output_blocks.4.0.time_stack.out_layers.3.bias": "blocks.118.conv2.bias",
            "model.diffusion_model.output_blocks.4.0.time_stack.out_layers.3.weight": "blocks.118.conv2.weight",
            "model.diffusion_model.output_blocks.4.1.norm.bias": "blocks.121.norm.bias",
            "model.diffusion_model.output_blocks.4.1.norm.weight": "blocks.121.norm.weight",
            "model.diffusion_model.output_blocks.4.1.proj_in.bias": "blocks.121.proj_in.bias",
            "model.diffusion_model.output_blocks.4.1.proj_in.weight": "blocks.121.proj_in.weight",
            "model.diffusion_model.output_blocks.4.1.proj_out.bias": "blocks.124.proj.bias",
            "model.diffusion_model.output_blocks.4.1.proj_out.weight": "blocks.124.proj.weight",
            "model.diffusion_model.output_blocks.4.1.time_mixer.mix_factor": "blocks.124.mix_factor",
            "model.diffusion_model.output_blocks.4.1.time_pos_embed.0.bias": "blocks.123.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.4.1.time_pos_embed.0.weight": "blocks.123.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.4.1.time_pos_embed.2.bias": "blocks.123.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.4.1.time_pos_embed.2.weight": "blocks.123.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn1.to_k.weight": "blocks.123.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn1.to_out.0.bias": "blocks.123.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn1.to_out.0.weight": "blocks.123.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn1.to_q.weight": "blocks.123.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn1.to_v.weight": "blocks.123.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn2.to_k.weight": "blocks.123.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn2.to_out.0.bias": "blocks.123.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn2.to_out.0.weight": "blocks.123.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn2.to_q.weight": "blocks.123.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.attn2.to_v.weight": "blocks.123.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff.net.0.proj.bias": "blocks.123.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff.net.0.proj.weight": "blocks.123.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff.net.2.bias": "blocks.123.ff_out.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff.net.2.weight": "blocks.123.ff_out.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.123.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.123.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff_in.net.2.bias": "blocks.123.ff_in.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.ff_in.net.2.weight": "blocks.123.ff_in.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm1.bias": "blocks.123.norm1.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm1.weight": "blocks.123.norm1.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm2.bias": "blocks.123.norm2.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm2.weight": "blocks.123.norm2.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm3.bias": "blocks.123.norm_out.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm3.weight": "blocks.123.norm_out.weight",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm_in.bias": "blocks.123.norm_in.bias",
            "model.diffusion_model.output_blocks.4.1.time_stack.0.norm_in.weight": "blocks.123.norm_in.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "blocks.121.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.121.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.121.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "blocks.121.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "blocks.121.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "blocks.121.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.121.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.121.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "blocks.121.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "blocks.121.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.121.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.121.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "blocks.121.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "blocks.121.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias": "blocks.121.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight": "blocks.121.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias": "blocks.121.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight": "blocks.121.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias": "blocks.121.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight": "blocks.121.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.5.0.emb_layers.1.bias": "blocks.126.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.5.0.emb_layers.1.weight": "blocks.126.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.5.0.in_layers.0.bias": "blocks.126.norm1.bias",
            "model.diffusion_model.output_blocks.5.0.in_layers.0.weight": "blocks.126.norm1.weight",
            "model.diffusion_model.output_blocks.5.0.in_layers.2.bias": "blocks.126.conv1.bias",
            "model.diffusion_model.output_blocks.5.0.in_layers.2.weight": "blocks.126.conv1.weight",
            "model.diffusion_model.output_blocks.5.0.out_layers.0.bias": "blocks.126.norm2.bias",
            "model.diffusion_model.output_blocks.5.0.out_layers.0.weight": "blocks.126.norm2.weight",
            "model.diffusion_model.output_blocks.5.0.out_layers.3.bias": "blocks.126.conv2.bias",
            "model.diffusion_model.output_blocks.5.0.out_layers.3.weight": "blocks.126.conv2.weight",
            "model.diffusion_model.output_blocks.5.0.skip_connection.bias": "blocks.126.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.5.0.skip_connection.weight": "blocks.126.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.5.0.time_mixer.mix_factor": "blocks.129.mix_factor",
            "model.diffusion_model.output_blocks.5.0.time_stack.emb_layers.1.bias": "blocks.128.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.5.0.time_stack.emb_layers.1.weight": "blocks.128.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.5.0.time_stack.in_layers.0.bias": "blocks.128.norm1.bias",
            "model.diffusion_model.output_blocks.5.0.time_stack.in_layers.0.weight": "blocks.128.norm1.weight",
            "model.diffusion_model.output_blocks.5.0.time_stack.in_layers.2.bias": "blocks.128.conv1.bias",
            "model.diffusion_model.output_blocks.5.0.time_stack.in_layers.2.weight": "blocks.128.conv1.weight",
            "model.diffusion_model.output_blocks.5.0.time_stack.out_layers.0.bias": "blocks.128.norm2.bias",
            "model.diffusion_model.output_blocks.5.0.time_stack.out_layers.0.weight": "blocks.128.norm2.weight",
            "model.diffusion_model.output_blocks.5.0.time_stack.out_layers.3.bias": "blocks.128.conv2.bias",
            "model.diffusion_model.output_blocks.5.0.time_stack.out_layers.3.weight": "blocks.128.conv2.weight",
            "model.diffusion_model.output_blocks.5.1.norm.bias": "blocks.131.norm.bias",
            "model.diffusion_model.output_blocks.5.1.norm.weight": "blocks.131.norm.weight",
            "model.diffusion_model.output_blocks.5.1.proj_in.bias": "blocks.131.proj_in.bias",
            "model.diffusion_model.output_blocks.5.1.proj_in.weight": "blocks.131.proj_in.weight",
            "model.diffusion_model.output_blocks.5.1.proj_out.bias": "blocks.134.proj.bias",
            "model.diffusion_model.output_blocks.5.1.proj_out.weight": "blocks.134.proj.weight",
            "model.diffusion_model.output_blocks.5.1.time_mixer.mix_factor": "blocks.134.mix_factor",
            "model.diffusion_model.output_blocks.5.1.time_pos_embed.0.bias": "blocks.133.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.5.1.time_pos_embed.0.weight": "blocks.133.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.5.1.time_pos_embed.2.bias": "blocks.133.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.5.1.time_pos_embed.2.weight": "blocks.133.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn1.to_k.weight": "blocks.133.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn1.to_out.0.bias": "blocks.133.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn1.to_out.0.weight": "blocks.133.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn1.to_q.weight": "blocks.133.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn1.to_v.weight": "blocks.133.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn2.to_k.weight": "blocks.133.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn2.to_out.0.bias": "blocks.133.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn2.to_out.0.weight": "blocks.133.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn2.to_q.weight": "blocks.133.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.attn2.to_v.weight": "blocks.133.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff.net.0.proj.bias": "blocks.133.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff.net.0.proj.weight": "blocks.133.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff.net.2.bias": "blocks.133.ff_out.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff.net.2.weight": "blocks.133.ff_out.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.133.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.133.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff_in.net.2.bias": "blocks.133.ff_in.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.ff_in.net.2.weight": "blocks.133.ff_in.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm1.bias": "blocks.133.norm1.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm1.weight": "blocks.133.norm1.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm2.bias": "blocks.133.norm2.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm2.weight": "blocks.133.norm2.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm3.bias": "blocks.133.norm_out.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm3.weight": "blocks.133.norm_out.weight",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm_in.bias": "blocks.133.norm_in.bias",
            "model.diffusion_model.output_blocks.5.1.time_stack.0.norm_in.weight": "blocks.133.norm_in.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "blocks.131.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.131.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.131.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "blocks.131.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "blocks.131.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "blocks.131.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.131.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.131.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "blocks.131.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "blocks.131.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.131.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.131.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "blocks.131.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "blocks.131.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias": "blocks.131.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight": "blocks.131.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias": "blocks.131.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight": "blocks.131.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias": "blocks.131.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight": "blocks.131.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.5.2.conv.bias": "blocks.135.conv.bias",
            "model.diffusion_model.output_blocks.5.2.conv.weight": "blocks.135.conv.weight",
            "model.diffusion_model.output_blocks.6.0.emb_layers.1.bias": "blocks.137.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.6.0.emb_layers.1.weight": "blocks.137.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.6.0.in_layers.0.bias": "blocks.137.norm1.bias",
            "model.diffusion_model.output_blocks.6.0.in_layers.0.weight": "blocks.137.norm1.weight",
            "model.diffusion_model.output_blocks.6.0.in_layers.2.bias": "blocks.137.conv1.bias",
            "model.diffusion_model.output_blocks.6.0.in_layers.2.weight": "blocks.137.conv1.weight",
            "model.diffusion_model.output_blocks.6.0.out_layers.0.bias": "blocks.137.norm2.bias",
            "model.diffusion_model.output_blocks.6.0.out_layers.0.weight": "blocks.137.norm2.weight",
            "model.diffusion_model.output_blocks.6.0.out_layers.3.bias": "blocks.137.conv2.bias",
            "model.diffusion_model.output_blocks.6.0.out_layers.3.weight": "blocks.137.conv2.weight",
            "model.diffusion_model.output_blocks.6.0.skip_connection.bias": "blocks.137.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.6.0.skip_connection.weight": "blocks.137.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.6.0.time_mixer.mix_factor": "blocks.140.mix_factor",
            "model.diffusion_model.output_blocks.6.0.time_stack.emb_layers.1.bias": "blocks.139.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.6.0.time_stack.emb_layers.1.weight": "blocks.139.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.6.0.time_stack.in_layers.0.bias": "blocks.139.norm1.bias",
            "model.diffusion_model.output_blocks.6.0.time_stack.in_layers.0.weight": "blocks.139.norm1.weight",
            "model.diffusion_model.output_blocks.6.0.time_stack.in_layers.2.bias": "blocks.139.conv1.bias",
            "model.diffusion_model.output_blocks.6.0.time_stack.in_layers.2.weight": "blocks.139.conv1.weight",
            "model.diffusion_model.output_blocks.6.0.time_stack.out_layers.0.bias": "blocks.139.norm2.bias",
            "model.diffusion_model.output_blocks.6.0.time_stack.out_layers.0.weight": "blocks.139.norm2.weight",
            "model.diffusion_model.output_blocks.6.0.time_stack.out_layers.3.bias": "blocks.139.conv2.bias",
            "model.diffusion_model.output_blocks.6.0.time_stack.out_layers.3.weight": "blocks.139.conv2.weight",
            "model.diffusion_model.output_blocks.6.1.norm.bias": "blocks.142.norm.bias",
            "model.diffusion_model.output_blocks.6.1.norm.weight": "blocks.142.norm.weight",
            "model.diffusion_model.output_blocks.6.1.proj_in.bias": "blocks.142.proj_in.bias",
            "model.diffusion_model.output_blocks.6.1.proj_in.weight": "blocks.142.proj_in.weight",
            "model.diffusion_model.output_blocks.6.1.proj_out.bias": "blocks.145.proj.bias",
            "model.diffusion_model.output_blocks.6.1.proj_out.weight": "blocks.145.proj.weight",
            "model.diffusion_model.output_blocks.6.1.time_mixer.mix_factor": "blocks.145.mix_factor",
            "model.diffusion_model.output_blocks.6.1.time_pos_embed.0.bias": "blocks.144.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.6.1.time_pos_embed.0.weight": "blocks.144.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.6.1.time_pos_embed.2.bias": "blocks.144.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.6.1.time_pos_embed.2.weight": "blocks.144.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn1.to_k.weight": "blocks.144.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn1.to_out.0.bias": "blocks.144.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn1.to_out.0.weight": "blocks.144.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn1.to_q.weight": "blocks.144.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn1.to_v.weight": "blocks.144.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn2.to_k.weight": "blocks.144.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn2.to_out.0.bias": "blocks.144.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn2.to_out.0.weight": "blocks.144.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn2.to_q.weight": "blocks.144.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.attn2.to_v.weight": "blocks.144.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff.net.0.proj.bias": "blocks.144.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff.net.0.proj.weight": "blocks.144.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff.net.2.bias": "blocks.144.ff_out.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff.net.2.weight": "blocks.144.ff_out.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.144.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.144.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff_in.net.2.bias": "blocks.144.ff_in.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.ff_in.net.2.weight": "blocks.144.ff_in.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm1.bias": "blocks.144.norm1.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm1.weight": "blocks.144.norm1.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm2.bias": "blocks.144.norm2.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm2.weight": "blocks.144.norm2.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm3.bias": "blocks.144.norm_out.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm3.weight": "blocks.144.norm_out.weight",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm_in.bias": "blocks.144.norm_in.bias",
            "model.diffusion_model.output_blocks.6.1.time_stack.0.norm_in.weight": "blocks.144.norm_in.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight": "blocks.142.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.142.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.142.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight": "blocks.142.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight": "blocks.142.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight": "blocks.142.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.142.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.142.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight": "blocks.142.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight": "blocks.142.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.142.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.142.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias": "blocks.142.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight": "blocks.142.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias": "blocks.142.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight": "blocks.142.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias": "blocks.142.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight": "blocks.142.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias": "blocks.142.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight": "blocks.142.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.7.0.emb_layers.1.bias": "blocks.147.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.7.0.emb_layers.1.weight": "blocks.147.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.7.0.in_layers.0.bias": "blocks.147.norm1.bias",
            "model.diffusion_model.output_blocks.7.0.in_layers.0.weight": "blocks.147.norm1.weight",
            "model.diffusion_model.output_blocks.7.0.in_layers.2.bias": "blocks.147.conv1.bias",
            "model.diffusion_model.output_blocks.7.0.in_layers.2.weight": "blocks.147.conv1.weight",
            "model.diffusion_model.output_blocks.7.0.out_layers.0.bias": "blocks.147.norm2.bias",
            "model.diffusion_model.output_blocks.7.0.out_layers.0.weight": "blocks.147.norm2.weight",
            "model.diffusion_model.output_blocks.7.0.out_layers.3.bias": "blocks.147.conv2.bias",
            "model.diffusion_model.output_blocks.7.0.out_layers.3.weight": "blocks.147.conv2.weight",
            "model.diffusion_model.output_blocks.7.0.skip_connection.bias": "blocks.147.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.7.0.skip_connection.weight": "blocks.147.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.7.0.time_mixer.mix_factor": "blocks.150.mix_factor",
            "model.diffusion_model.output_blocks.7.0.time_stack.emb_layers.1.bias": "blocks.149.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.7.0.time_stack.emb_layers.1.weight": "blocks.149.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.7.0.time_stack.in_layers.0.bias": "blocks.149.norm1.bias",
            "model.diffusion_model.output_blocks.7.0.time_stack.in_layers.0.weight": "blocks.149.norm1.weight",
            "model.diffusion_model.output_blocks.7.0.time_stack.in_layers.2.bias": "blocks.149.conv1.bias",
            "model.diffusion_model.output_blocks.7.0.time_stack.in_layers.2.weight": "blocks.149.conv1.weight",
            "model.diffusion_model.output_blocks.7.0.time_stack.out_layers.0.bias": "blocks.149.norm2.bias",
            "model.diffusion_model.output_blocks.7.0.time_stack.out_layers.0.weight": "blocks.149.norm2.weight",
            "model.diffusion_model.output_blocks.7.0.time_stack.out_layers.3.bias": "blocks.149.conv2.bias",
            "model.diffusion_model.output_blocks.7.0.time_stack.out_layers.3.weight": "blocks.149.conv2.weight",
            "model.diffusion_model.output_blocks.7.1.norm.bias": "blocks.152.norm.bias",
            "model.diffusion_model.output_blocks.7.1.norm.weight": "blocks.152.norm.weight",
            "model.diffusion_model.output_blocks.7.1.proj_in.bias": "blocks.152.proj_in.bias",
            "model.diffusion_model.output_blocks.7.1.proj_in.weight": "blocks.152.proj_in.weight",
            "model.diffusion_model.output_blocks.7.1.proj_out.bias": "blocks.155.proj.bias",
            "model.diffusion_model.output_blocks.7.1.proj_out.weight": "blocks.155.proj.weight",
            "model.diffusion_model.output_blocks.7.1.time_mixer.mix_factor": "blocks.155.mix_factor",
            "model.diffusion_model.output_blocks.7.1.time_pos_embed.0.bias": "blocks.154.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.7.1.time_pos_embed.0.weight": "blocks.154.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.7.1.time_pos_embed.2.bias": "blocks.154.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.7.1.time_pos_embed.2.weight": "blocks.154.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn1.to_k.weight": "blocks.154.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn1.to_out.0.bias": "blocks.154.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn1.to_out.0.weight": "blocks.154.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn1.to_q.weight": "blocks.154.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn1.to_v.weight": "blocks.154.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn2.to_k.weight": "blocks.154.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn2.to_out.0.bias": "blocks.154.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn2.to_out.0.weight": "blocks.154.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn2.to_q.weight": "blocks.154.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.attn2.to_v.weight": "blocks.154.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff.net.0.proj.bias": "blocks.154.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff.net.0.proj.weight": "blocks.154.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff.net.2.bias": "blocks.154.ff_out.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff.net.2.weight": "blocks.154.ff_out.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.154.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.154.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff_in.net.2.bias": "blocks.154.ff_in.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.ff_in.net.2.weight": "blocks.154.ff_in.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm1.bias": "blocks.154.norm1.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm1.weight": "blocks.154.norm1.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm2.bias": "blocks.154.norm2.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm2.weight": "blocks.154.norm2.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm3.bias": "blocks.154.norm_out.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm3.weight": "blocks.154.norm_out.weight",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm_in.bias": "blocks.154.norm_in.bias",
            "model.diffusion_model.output_blocks.7.1.time_stack.0.norm_in.weight": "blocks.154.norm_in.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "blocks.152.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.152.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.152.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "blocks.152.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "blocks.152.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "blocks.152.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.152.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.152.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "blocks.152.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "blocks.152.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.152.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.152.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "blocks.152.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "blocks.152.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias": "blocks.152.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight": "blocks.152.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias": "blocks.152.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight": "blocks.152.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias": "blocks.152.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight": "blocks.152.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.8.0.emb_layers.1.bias": "blocks.157.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.8.0.emb_layers.1.weight": "blocks.157.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.8.0.in_layers.0.bias": "blocks.157.norm1.bias",
            "model.diffusion_model.output_blocks.8.0.in_layers.0.weight": "blocks.157.norm1.weight",
            "model.diffusion_model.output_blocks.8.0.in_layers.2.bias": "blocks.157.conv1.bias",
            "model.diffusion_model.output_blocks.8.0.in_layers.2.weight": "blocks.157.conv1.weight",
            "model.diffusion_model.output_blocks.8.0.out_layers.0.bias": "blocks.157.norm2.bias",
            "model.diffusion_model.output_blocks.8.0.out_layers.0.weight": "blocks.157.norm2.weight",
            "model.diffusion_model.output_blocks.8.0.out_layers.3.bias": "blocks.157.conv2.bias",
            "model.diffusion_model.output_blocks.8.0.out_layers.3.weight": "blocks.157.conv2.weight",
            "model.diffusion_model.output_blocks.8.0.skip_connection.bias": "blocks.157.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.8.0.skip_connection.weight": "blocks.157.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.8.0.time_mixer.mix_factor": "blocks.160.mix_factor",
            "model.diffusion_model.output_blocks.8.0.time_stack.emb_layers.1.bias": "blocks.159.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.8.0.time_stack.emb_layers.1.weight": "blocks.159.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.8.0.time_stack.in_layers.0.bias": "blocks.159.norm1.bias",
            "model.diffusion_model.output_blocks.8.0.time_stack.in_layers.0.weight": "blocks.159.norm1.weight",
            "model.diffusion_model.output_blocks.8.0.time_stack.in_layers.2.bias": "blocks.159.conv1.bias",
            "model.diffusion_model.output_blocks.8.0.time_stack.in_layers.2.weight": "blocks.159.conv1.weight",
            "model.diffusion_model.output_blocks.8.0.time_stack.out_layers.0.bias": "blocks.159.norm2.bias",
            "model.diffusion_model.output_blocks.8.0.time_stack.out_layers.0.weight": "blocks.159.norm2.weight",
            "model.diffusion_model.output_blocks.8.0.time_stack.out_layers.3.bias": "blocks.159.conv2.bias",
            "model.diffusion_model.output_blocks.8.0.time_stack.out_layers.3.weight": "blocks.159.conv2.weight",
            "model.diffusion_model.output_blocks.8.1.norm.bias": "blocks.162.norm.bias",
            "model.diffusion_model.output_blocks.8.1.norm.weight": "blocks.162.norm.weight",
            "model.diffusion_model.output_blocks.8.1.proj_in.bias": "blocks.162.proj_in.bias",
            "model.diffusion_model.output_blocks.8.1.proj_in.weight": "blocks.162.proj_in.weight",
            "model.diffusion_model.output_blocks.8.1.proj_out.bias": "blocks.165.proj.bias",
            "model.diffusion_model.output_blocks.8.1.proj_out.weight": "blocks.165.proj.weight",
            "model.diffusion_model.output_blocks.8.1.time_mixer.mix_factor": "blocks.165.mix_factor",
            "model.diffusion_model.output_blocks.8.1.time_pos_embed.0.bias": "blocks.164.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.8.1.time_pos_embed.0.weight": "blocks.164.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.8.1.time_pos_embed.2.bias": "blocks.164.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.8.1.time_pos_embed.2.weight": "blocks.164.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn1.to_k.weight": "blocks.164.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn1.to_out.0.bias": "blocks.164.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn1.to_out.0.weight": "blocks.164.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn1.to_q.weight": "blocks.164.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn1.to_v.weight": "blocks.164.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn2.to_k.weight": "blocks.164.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn2.to_out.0.bias": "blocks.164.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn2.to_out.0.weight": "blocks.164.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn2.to_q.weight": "blocks.164.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.attn2.to_v.weight": "blocks.164.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff.net.0.proj.bias": "blocks.164.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff.net.0.proj.weight": "blocks.164.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff.net.2.bias": "blocks.164.ff_out.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff.net.2.weight": "blocks.164.ff_out.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.164.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.164.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff_in.net.2.bias": "blocks.164.ff_in.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.ff_in.net.2.weight": "blocks.164.ff_in.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm1.bias": "blocks.164.norm1.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm1.weight": "blocks.164.norm1.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm2.bias": "blocks.164.norm2.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm2.weight": "blocks.164.norm2.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm3.bias": "blocks.164.norm_out.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm3.weight": "blocks.164.norm_out.weight",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm_in.bias": "blocks.164.norm_in.bias",
            "model.diffusion_model.output_blocks.8.1.time_stack.0.norm_in.weight": "blocks.164.norm_in.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "blocks.162.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.162.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.162.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "blocks.162.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "blocks.162.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "blocks.162.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.162.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.162.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "blocks.162.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "blocks.162.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.162.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.162.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "blocks.162.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "blocks.162.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias": "blocks.162.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight": "blocks.162.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias": "blocks.162.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight": "blocks.162.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias": "blocks.162.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight": "blocks.162.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.8.2.conv.bias": "blocks.166.conv.bias",
            "model.diffusion_model.output_blocks.8.2.conv.weight": "blocks.166.conv.weight",
            "model.diffusion_model.output_blocks.9.0.emb_layers.1.bias": "blocks.168.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.9.0.emb_layers.1.weight": "blocks.168.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.9.0.in_layers.0.bias": "blocks.168.norm1.bias",
            "model.diffusion_model.output_blocks.9.0.in_layers.0.weight": "blocks.168.norm1.weight",
            "model.diffusion_model.output_blocks.9.0.in_layers.2.bias": "blocks.168.conv1.bias",
            "model.diffusion_model.output_blocks.9.0.in_layers.2.weight": "blocks.168.conv1.weight",
            "model.diffusion_model.output_blocks.9.0.out_layers.0.bias": "blocks.168.norm2.bias",
            "model.diffusion_model.output_blocks.9.0.out_layers.0.weight": "blocks.168.norm2.weight",
            "model.diffusion_model.output_blocks.9.0.out_layers.3.bias": "blocks.168.conv2.bias",
            "model.diffusion_model.output_blocks.9.0.out_layers.3.weight": "blocks.168.conv2.weight",
            "model.diffusion_model.output_blocks.9.0.skip_connection.bias": "blocks.168.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.9.0.skip_connection.weight": "blocks.168.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.9.0.time_mixer.mix_factor": "blocks.171.mix_factor",
            "model.diffusion_model.output_blocks.9.0.time_stack.emb_layers.1.bias": "blocks.170.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.9.0.time_stack.emb_layers.1.weight": "blocks.170.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.9.0.time_stack.in_layers.0.bias": "blocks.170.norm1.bias",
            "model.diffusion_model.output_blocks.9.0.time_stack.in_layers.0.weight": "blocks.170.norm1.weight",
            "model.diffusion_model.output_blocks.9.0.time_stack.in_layers.2.bias": "blocks.170.conv1.bias",
            "model.diffusion_model.output_blocks.9.0.time_stack.in_layers.2.weight": "blocks.170.conv1.weight",
            "model.diffusion_model.output_blocks.9.0.time_stack.out_layers.0.bias": "blocks.170.norm2.bias",
            "model.diffusion_model.output_blocks.9.0.time_stack.out_layers.0.weight": "blocks.170.norm2.weight",
            "model.diffusion_model.output_blocks.9.0.time_stack.out_layers.3.bias": "blocks.170.conv2.bias",
            "model.diffusion_model.output_blocks.9.0.time_stack.out_layers.3.weight": "blocks.170.conv2.weight",
            "model.diffusion_model.output_blocks.9.1.norm.bias": "blocks.173.norm.bias",
            "model.diffusion_model.output_blocks.9.1.norm.weight": "blocks.173.norm.weight",
            "model.diffusion_model.output_blocks.9.1.proj_in.bias": "blocks.173.proj_in.bias",
            "model.diffusion_model.output_blocks.9.1.proj_in.weight": "blocks.173.proj_in.weight",
            "model.diffusion_model.output_blocks.9.1.proj_out.bias": "blocks.176.proj.bias",
            "model.diffusion_model.output_blocks.9.1.proj_out.weight": "blocks.176.proj.weight",
            "model.diffusion_model.output_blocks.9.1.time_mixer.mix_factor": "blocks.176.mix_factor",
            "model.diffusion_model.output_blocks.9.1.time_pos_embed.0.bias": "blocks.175.positional_embedding_proj.0.bias",
            "model.diffusion_model.output_blocks.9.1.time_pos_embed.0.weight": "blocks.175.positional_embedding_proj.0.weight",
            "model.diffusion_model.output_blocks.9.1.time_pos_embed.2.bias": "blocks.175.positional_embedding_proj.2.bias",
            "model.diffusion_model.output_blocks.9.1.time_pos_embed.2.weight": "blocks.175.positional_embedding_proj.2.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn1.to_k.weight": "blocks.175.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn1.to_out.0.bias": "blocks.175.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn1.to_out.0.weight": "blocks.175.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn1.to_q.weight": "blocks.175.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn1.to_v.weight": "blocks.175.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn2.to_k.weight": "blocks.175.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn2.to_out.0.bias": "blocks.175.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn2.to_out.0.weight": "blocks.175.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn2.to_q.weight": "blocks.175.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.attn2.to_v.weight": "blocks.175.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff.net.0.proj.bias": "blocks.175.act_fn_out.proj.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff.net.0.proj.weight": "blocks.175.act_fn_out.proj.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff.net.2.bias": "blocks.175.ff_out.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff.net.2.weight": "blocks.175.ff_out.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff_in.net.0.proj.bias": "blocks.175.act_fn_in.proj.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff_in.net.0.proj.weight": "blocks.175.act_fn_in.proj.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff_in.net.2.bias": "blocks.175.ff_in.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.ff_in.net.2.weight": "blocks.175.ff_in.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm1.bias": "blocks.175.norm1.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm1.weight": "blocks.175.norm1.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm2.bias": "blocks.175.norm2.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm2.weight": "blocks.175.norm2.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm3.bias": "blocks.175.norm_out.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm3.weight": "blocks.175.norm_out.weight",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm_in.bias": "blocks.175.norm_in.bias",
            "model.diffusion_model.output_blocks.9.1.time_stack.0.norm_in.weight": "blocks.175.norm_in.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight": "blocks.173.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.173.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.173.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight": "blocks.173.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight": "blocks.173.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": "blocks.173.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.173.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.173.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight": "blocks.173.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight": "blocks.173.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.173.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.173.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias": "blocks.173.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight": "blocks.173.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias": "blocks.173.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight": "blocks.173.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias": "blocks.173.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight": "blocks.173.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias": "blocks.173.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight": "blocks.173.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.time_embed.0.bias": "time_embedding.0.bias",
            "model.diffusion_model.time_embed.0.weight": "time_embedding.0.weight",
            "model.diffusion_model.time_embed.2.bias": "time_embedding.2.bias",
            "model.diffusion_model.time_embed.2.weight": "time_embedding.2.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if ".proj_in." in name or ".proj_out." in name:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        if add_positional_conv is not None:
            extra_names = [
                "blocks.7.positional_conv", "blocks.17.positional_conv", "blocks.29.positional_conv", "blocks.39.positional_conv",
                "blocks.51.positional_conv", "blocks.61.positional_conv", "blocks.83.positional_conv", "blocks.113.positional_conv",
                "blocks.123.positional_conv", "blocks.133.positional_conv", "blocks.144.positional_conv", "blocks.154.positional_conv",
                "blocks.164.positional_conv", "blocks.175.positional_conv", "blocks.185.positional_conv", "blocks.195.positional_conv",
            ]
            extra_channels = [320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
            for name, channels in zip(extra_names, extra_channels):
                weight = torch.zeros((channels, channels, 3, 3, 3))
                weight[:,:,1,1,1] = torch.eye(channels, channels)
                bias = torch.zeros((channels,))
                state_dict_[name + ".weight"] = weight
                state_dict_[name + ".bias"] = bias
        return state_dict_
