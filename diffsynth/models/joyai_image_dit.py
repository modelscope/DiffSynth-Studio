import math
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        self.act = nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        self.post_act = nn.SiLU() if post_act_fn == "silu" else None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        if activation_fn == "gelu-approximate":
            self.proj = nn.Linear(dim, inner_dim, bias=bias)
            self.act = lambda x: F.gelu(x, approximate="tanh")
        elif activation_fn == "gelu":
            self.proj = nn.Linear(dim, inner_dim, bias=bias)
            self.act = F.gelu
        else:
            self.proj = nn.Linear(dim, inner_dim, bias=bias)
            self.act = F.gelu
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, dim_out, bias=bias)
        if final_dropout:
            self.final_drop = nn.Dropout(dropout)
        else:
            self.final_drop = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        if self.final_drop is not None:
            hidden_states = self.final_drop(hidden_states)
        return hidden_states



def get_cu_seqlens(text_mask, img_len):
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2
    return cu_seqlens


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)
    return grid


def reshape_for_broadcast(freqs_cis, x, head_first=False):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (x.shape[-2], x.shape[-1])
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(xq, xk, freqs_cis, head_first=False):
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    cos, sin = cos.to(xq.device), sin.to(xq.device)
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, use_real=False, theta_rescale_factor=1.0, interpolation_factor=1.0):
    if isinstance(pos, int):
        pos = torch.arange(pos).float()
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


def get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000.0, use_real=False,
                            txt_rope_size=None, theta_rescale_factor=1.0, interpolation_factor=1.0):
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i], grid[i].reshape(-1), theta,
            use_real=use_real, theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)
    if use_real:
        vis_emb = (torch.cat([emb[0] for emb in embs], dim=1), torch.cat([emb[1] for emb in embs], dim=1))
    else:
        vis_emb = torch.cat(embs, dim=1)
    if txt_rope_size is not None:
        embs_txt = []
        vis_max_ids = grid.view(-1).max().item()
        grid_txt = torch.arange(txt_rope_size) + vis_max_ids + 1
        for i in range(len(rope_dim_list)):
            emb = get_1d_rotary_pos_embed(
                rope_dim_list[i], grid_txt, theta,
                use_real=use_real, theta_rescale_factor=theta_rescale_factor[i],
                interpolation_factor=interpolation_factor[i],
            )
            embs_txt.append(emb)
        if use_real:
            txt_emb = (torch.cat([emb[0] for emb in embs_txt], dim=1), torch.cat([emb[1] for emb in embs_txt], dim=1))
        else:
            txt_emb = torch.cat(embs_txt, dim=1)
    else:
        txt_emb = None
    return vis_emb, txt_emb


class ModulateWan(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        factory_kwargs = {"dtype": dtype, "device": device}
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size, **factory_kwargs) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]


def modulate(x, shift=None, scale=None):
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


def load_modulation(modulate_type: str, hidden_size: int, factor: int, dtype=None, device=None):
    factory_kwargs = {"dtype": dtype, "device": device}
    if modulate_type == 'wanx':
        return ModulateWan(hidden_size, factor, **factory_kwargs)
    raise ValueError(f"Unknown modulation type: {modulate_type}. Only 'wanx' is supported.")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with separate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: Optional[str] = "wanx",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dit_modulation_type = dit_modulation_type
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size, factor=6, **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        self.img_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.img_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

        self.txt_mod = load_modulation(
            modulate_type=self.dit_modulation_type,
            hidden_size=hidden_size, factor=6, **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        self.txt_attn_q_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.txt_attn_k_norm = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim, activation_fn="gelu-approximate")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple = None,
        txt_freqs_cis: tuple = None,
        attn_kwargs: Optional[dict] = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift, img_mod1_scale, img_mod1_gate,
            img_mod2_shift, img_mod2_scale, img_mod2_gate,
        ) = self.img_mod(vec)
        (
            txt_mod1_shift, txt_mod1_scale, txt_mod1_gate,
            txt_mod2_shift, txt_mod2_scale, txt_mod2_gate,
        ) = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if vis_freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, vis_freqs_cis, head_first=False)
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        if txt_freqs_cis is not None:
            raise NotImplementedError("RoPE text is not supported for inference")

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        # Use DiffSynth unified attention
        attn_out = attention_forward(
            q, k, v,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )

        attn_out = attn_out.flatten(2, 3)
        img_attn, txt_attn = attn_out[:, : img.shape[1]], attn_out[:, img.shape[1]:]

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class Transformer3DModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 16,
        out_channels: int = None,
        hidden_size: int = 4096,
        heads_num: int = 32,
        text_states_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        mm_double_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        rope_type: str = 'rope',
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        dit_modulation_type: str = "wanx",
        theta: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.rope_dim_list = rope_dim_list
        self.dit_modulation_type = dit_modulation_type
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.rope_type = rope_type
        self.theta = theta

        factory_kwargs = {"device": device, "dtype": dtype}

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")

        self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_states_dim,
        )

        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size, self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                dit_modulation_type=self.dit_modulation_type,
                **factory_kwargs,
            )
            for _ in range(mm_double_blocks_depth)
        ])

        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(patch_size), **factory_kwargs)

    def get_rotary_pos_embed(self, vis_rope_size, txt_rope_size=None):
        target_ndim = 3
        if len(vis_rope_size) != target_ndim:
            vis_rope_size = [1] * (target_ndim - len(vis_rope_size)) + vis_rope_size
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim
        vis_freqs, txt_freqs = get_nd_rotary_pos_embed(
            rope_dim_list, vis_rope_size,
            txt_rope_size=txt_rope_size if self.rope_type == 'mrope' else None,
            theta=self.theta, use_real=True, theta_rescale_factor=1,
        )
        return vis_freqs, txt_freqs

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        return_dict: bool = True,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        is_multi_item = (len(hidden_states.shape) == 6)
        num_items = 0
        if is_multi_item:
            num_items = hidden_states.shape[1]
            if num_items > 1:
                assert self.patch_size[0] == 1, "For multi-item input, patch_size[0] must be 1"
                hidden_states = torch.cat([hidden_states[:, -1:], hidden_states[:, :-1]], dim=1)
            hidden_states = rearrange(hidden_states, 'b n c t h w -> b c (n t) h w')

        batch_size, _, ot, oh, ow = hidden_states.shape
        tt, th, tw = ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2]

        if encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = torch.ones(
                (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                dtype=torch.bool,
            ).to(encoder_hidden_states.device)

        img = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temb, vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        if vec.shape[-1] > self.hidden_size:
            vec = vec.unflatten(1, (6, -1))

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        vis_freqs_cis, txt_freqs_cis = self.get_rotary_pos_embed(
            vis_rope_size=(tt, th, tw),
            txt_rope_size=txt_seq_len if self.rope_type == 'mrope' else None,
        )

        for block in self.double_blocks:
            img, txt = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                img=img, txt=txt, vec=vec,
                vis_freqs_cis=vis_freqs_cis, txt_freqs_cis=txt_freqs_cis,
                attn_kwargs={},
            )

        img_len = img.shape[1]
        x = torch.cat((img, txt), 1)
        img = x[:, :img_len, ...]

        img = self.proj_out(self.norm_out(img))
        img = self.unpatchify(img, tt, th, tw)

        if is_multi_item:
            img = rearrange(img, 'b c (n t) h w -> b n c t h w', n=num_items)
            if num_items > 1:
                img = torch.cat([img[:, 1:], img[:, :1]], dim=1)

        return img

    def unpatchify(self, x, t, h, w):
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nthwopqc->nctohpwq", x)
        return x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
