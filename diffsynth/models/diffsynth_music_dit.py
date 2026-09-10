import math, torch
from ..core import gradient_checkpoint_forward
from ..core.attention.attention import attention_forward
from einops import rearrange
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


class TimestepEmbedding(torch.nn.Module):
    def __init__(self, in_channels, time_embed_dim, scale=1):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels
        self.act2 = torch.nn.SiLU()
        self.time_proj = torch.nn.Linear(time_embed_dim, time_embed_dim * 6)
        self.scale = scale

    def timestep_embedding(self, t, dim, max_period=10000):
        t = t * self.scale
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


class DiffSynthMusicTimestepEmbedding(torch.nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.time_embed = TimestepEmbedding(in_channels, time_embed_dim)
        self.time_embed_r = TimestepEmbedding(in_channels, time_embed_dim)

    def forward(self, timestep):
        timestep_r = timestep
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r
        return temb, timestep_proj


class DiffSynthMusicAttention(torch.nn.Module):
    def __init__(self, dim, num_heads_q, num_heads_kv, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(dim, num_heads_q * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, num_heads_kv * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, num_heads_kv * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads_q * self.head_dim, dim, bias=False)
        self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=1e-6)

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, x, y=None, window_size=None, pos_emb=None, return_kv=False, kv_cache=None) -> torch.Tensor:
        q = self.q_proj(x)
        q = rearrange(q, "b s (n d) -> b s n d", d=self.head_dim)
        q = self.q_norm(q)

        if y is None: y = x
        k, v = self.k_proj(y), self.v_proj(y)
        k, v = rearrange(k, "b s (n d) -> b s n d", d=self.head_dim), rearrange(v, "b s (n d) -> b s n d", d=self.head_dim)
        k = self.k_norm(k)

        if pos_emb is not None: q, k = self.apply_rotary_pos_emb(q, k, *pos_emb)
        if kv_cache is not None:
            k = torch.concat([k, kv_cache[0]], dim=1)
            v = torch.concat([v, kv_cache[1]], dim=1)
        attn_output = attention_forward(
            q, k, v,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s (n d)",
            window_size=window_size,
        )
        attn_output = self.o_proj(attn_output)
        if return_kv:
            return attn_output, (k, v)
        else:
            return attn_output


class MLP(torch.nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim_hidden, bias=False)
        self.up_proj = torch.nn.Linear(dim, dim_hidden, bias=False)
        self.down_proj = torch.nn.Linear(dim_hidden, dim, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DiffSynthMusicDiTLayer(torch.nn.Module):
    def __init__(self, dim=2560, num_heads_q=32, num_heads_kv=8, head_dim=128, dim_mlp=9728, window_size=None):
        super().__init__()
        self.self_attn_norm = torch.nn.RMSNorm(dim, eps=1e-6)
        self.self_attn = DiffSynthMusicAttention(dim=dim, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim)
        self.cross_attn_norm = torch.nn.RMSNorm(dim, eps=1e-6)
        self.cross_attn = DiffSynthMusicAttention(dim=dim, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim)
        self.mlp_norm = torch.nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, dim_mlp)
        self.scale_shift_table = torch.nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.window_size = window_size

    def forward(self, x, y, pos_emb, temb, return_kv=False, kv_cache=None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table.to(dtype=x.dtype, device=x.device) + temb).chunk(6, dim=1)
        x_hidden = self.self_attn_norm(x) * (1 + scale_msa) + shift_msa
        x_hidden = self.self_attn(x=x_hidden, pos_emb=pos_emb, window_size=self.window_size, kv_cache=kv_cache, return_kv=return_kv)
        if return_kv: x_hidden, kv = x_hidden
        x = x + x_hidden * gate_msa
        x_hidden = self.cross_attn_norm(x)
        x_hidden = self.cross_attn(x=x_hidden, y=y)
        x = x + x_hidden
        x_hidden = self.mlp_norm(x) * (1 + c_scale_msa) + c_shift_msa
        x_hidden = self.mlp(x_hidden)
        x = x + x_hidden * c_gate_msa
        if return_kv:
            return x, kv
        else:
            return x


class DiffSynthMusicChannelProj(torch.nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, bias=False, transposed=False):
        super().__init__()
        if transposed:
            self.conv = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        else:
            self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class DiffSynthMusicDiTModel(torch.nn.Module):
    def __init__(self, dim=2560, dim_mlp=9728, dim_condition=2048, num_blocks=32, num_heads_q=32, num_heads_kv=8, head_dim=128, window_size=128, patch_size=2):
        super().__init__()
        self.rotary_emb = Qwen3RotaryEmbedding(type('RopeConfig', (), {'head_dim': head_dim, 'max_position_embeddings': 32768, 'rope_theta': 1000000, 'rope_parameters': {'rope_type': 'default', 'rope_theta': 1000000}})())
        self.x_emb = DiffSynthMusicChannelProj(in_channels=64*3, out_channels=dim, patch_size=patch_size, bias=True)
        self.timestep_emb = DiffSynthMusicTimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.condition_emb = torch.nn.Linear(dim_condition, dim, bias=True)
        self.layers = torch.nn.ModuleList([
            DiffSynthMusicDiTLayer(dim=dim, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, dim_mlp=dim_mlp, window_size=window_size if block_id % 2 == 0 else None)
            for block_id in range(num_blocks)
        ])
        self.norm_out = torch.nn.RMSNorm(dim, eps=1e-6)
        self.proj_out = DiffSynthMusicChannelProj(in_channels=dim, out_channels=64, patch_size=patch_size, bias=True, transposed=True)
        self.scale_shift_table = torch.nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.placeholder_audio = torch.nn.Parameter(torch.zeros((1, 15000, 64)))

    def forward_kv_cache(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        x = torch.concat([self.placeholder_audio[:, :x.shape[1]], torch.ones_like(x), x], dim=-1)
        x = self.x_emb(x)
        y = self.condition_emb(y)
        timestep = torch.zeros((1,), dtype=x.dtype, device=x.device)
        final_timestep_emb, timestep_emb = self.timestep_emb(timestep)
        pos_emb = self.rotary_emb(x, torch.arange(0, x.shape[1], device=x.device).unsqueeze(0))

        kv_cache = {}
        for block_id, block in enumerate(self.layers):
            return_kv = block.window_size is None
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, y, pos_emb, timestep_emb,
                return_kv=return_kv,
                kv_cache=None,
            )
            if return_kv:
                x, kv = x
                kv_cache[f"{block_id}"] = kv

        return kv_cache

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        x = torch.concat([self.placeholder_audio[:, :x.shape[1]], torch.ones_like(x), x], dim=-1)
        x = self.x_emb(x)
        y = self.condition_emb(y)
        final_timestep_emb, timestep_emb = self.timestep_emb(timestep)
        pos_emb = self.rotary_emb(x, torch.arange(0, x.shape[1], device=x.device).unsqueeze(0))

        for block_id, block in enumerate(self.layers):
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, y, pos_emb, timestep_emb,
                return_kv=False,
                kv_cache=None if kv_cache is None else kv_cache.get(f"{block_id}"),
            )

        shift, scale = (self.scale_shift_table.to(dtype=x.dtype, device=x.device) + final_timestep_emb.unsqueeze(1)).chunk(2, dim=1)
        x = self.proj_out(self.norm_out(x) * (1 + scale) + shift)
        return x
