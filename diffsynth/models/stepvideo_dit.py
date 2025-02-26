# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Dict, Optional, Tuple, Union, List
import torch, math
from torch import nn
from einops import rearrange, repeat
from tqdm import tqdm


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output
    

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


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


class Timesteps(nn.Module):
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


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True
    ):
        super().__init__()
        linear_cls = nn.Linear

        self.linear_1 = linear_cls(
                in_channels, 
                time_embed_dim, 
                bias=sample_proj_bias,
            )

        if cond_proj_dim is not None:
            self.cond_proj = linear_cls(
                    cond_proj_dim, 
                    in_channels, 
                    bias=False,
                )
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
            
        self.linear_2 = linear_cls(
                time_embed_dim, 
                time_embed_dim_out, 
                bias=sample_proj_bias, 
            )

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

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


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.nframe_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
            self.fps_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, resolution=None, nframe=None, fps=None):
        hidden_dtype = timestep.dtype

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            batch_size = timestep.shape[0]
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            nframe_emb = self.additional_condition_proj(nframe.flatten()).to(hidden_dtype)
            nframe_emb = self.nframe_embedder(nframe_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + resolution_emb + nframe_emb

            if fps is not None:
                fps_emb = self.additional_condition_proj(fps.flatten()).to(hidden_dtype)
                fps_emb = self.fps_embedder(fps_emb).reshape(batch_size, -1)
                conditioning = conditioning + fps_emb
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Module):
    r"""
        Norm layer adaptive layer norm single (adaLN-single).

        As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

        Parameters:
            embedding_dim (`int`): The size of each embedding vector.
            use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """
    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, time_step_rescale=1000):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 2, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

        self.time_step_rescale = time_step_rescale  ## timestep usually in [0, 1], we rescale it to [0,1000] for stability

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep*self.time_step_rescale, **added_cond_kwargs)

        out = self.linear(self.silu(embedded_timestep))

        return out, embedded_timestep
    

class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(
                in_features, 
                hidden_size, 
                bias=True, 
            )        
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(
                hidden_size, 
                hidden_size, 
                bias=True, 
            )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def attn_processor(self, attn_type):
        if attn_type == 'torch':
            return self.torch_attn_func
        elif attn_type == 'parallel':
            return self.parallel_attn_func
        else:
            raise Exception('Not supported attention type...')

    def torch_attn_func(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        **kwargs
    ):

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
            
        if attn_mask is not None and attn_mask.ndim == 3:   ## no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        x = rearrange(x, 'b h s d -> b s h d')
        return x        


class RoPE1D:
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        self.base = freq
        self.F0 = F0
        self.scaling_factor = scaling_factor
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def __call__(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * positions: batch_size x ntokens (t position of each token)
        output:
            * tokens after applying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        D = tokens.size(3)
        assert positions.ndim == 2  # Batch, Seq
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        tokens = self.apply_rope1d(tokens, positions, cos, sin)
        return tokens


class RoPE3D(RoPE1D):
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        super(RoPE3D, self).__init__(freq, F0, scaling_factor)
        self.position_cache = {}

    def get_mesh_3d(self, rope_positions, bsz):
        f, h, w = rope_positions

        if f"{f}-{h}-{w}" not in self.position_cache:
            x = torch.arange(f, device='cpu')
            y = torch.arange(h, device='cpu')
            z = torch.arange(w, device='cpu')
            self.position_cache[f"{f}-{h}-{w}"] = torch.cartesian_prod(x, y, z).view(1, f*h*w, 3).expand(bsz, -1, 3)
        return self.position_cache[f"{f}-{h}-{w}"]
     
    def __call__(self, tokens, rope_positions, ch_split, parallel=False):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * rope_positions: list of (f, h, w)
        output:
            * tokens after applying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        assert sum(ch_split) == tokens.size(-1); 

        mesh_grid = self.get_mesh_3d(rope_positions, bsz=tokens.shape[0])
        out = []
        for i, (D, x) in enumerate(zip(ch_split, torch.split(tokens, ch_split, dim=-1))):
            cos, sin = self.get_cos_sin(D, int(mesh_grid.max()) + 1, tokens.device, tokens.dtype)
            
            if parallel:
                pass
            else:
                mesh = mesh_grid[:, :, i].clone()
            x = self.apply_rope1d(x, mesh.to(tokens.device), cos, sin)
            out.append(x)
            
        tokens = torch.cat(out, dim=-1)
        return tokens


class SelfAttention(Attention):
    def __init__(self, hidden_dim, head_dim, bias=False, with_rope=True, with_qk_norm=True, attn_type='torch'):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim
        
        self.wqkv = nn.Linear(hidden_dim, hidden_dim*3, bias=bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.with_rope = with_rope
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)
        
        if self.with_rope:
            self.rope_3d = RoPE3D(freq=1e4, F0=1.0, scaling_factor=1.0)
            self.rope_ch_split = [64, 32, 32]
        
        self.core_attention = self.attn_processor(attn_type=attn_type)
        self.parallel = attn_type=='parallel'
        
    def apply_rope3d(self, x, fhw_positions, rope_ch_split, parallel=True):
        x = self.rope_3d(x, fhw_positions, rope_ch_split, parallel)
        return x
        
    def forward(
        self, 
        x,
        cu_seqlens=None,
        max_seqlen=None,
        rope_positions=None,
        attn_mask=None
    ):
        xqkv = self.wqkv(x) 
        xqkv = xqkv.view(*x.shape[:-1], self.n_heads, 3*self.head_dim)

        xq, xk, xv = torch.split(xqkv, [self.head_dim]*3, dim=-1)  ## seq_len, n, dim
    
        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
    
        if self.with_rope:
            xq = self.apply_rope3d(xq, rope_positions, self.rope_ch_split, parallel=self.parallel)
            xk = self.apply_rope3d(xk, rope_positions, self.rope_ch_split, parallel=self.parallel)
            
        output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attn_mask=attn_mask
                )
        output = rearrange(output, 'b s h d -> b s (h d)')
        output = self.wo(output)
        
        return output
    
    
class CrossAttention(Attention):
    def __init__(self, hidden_dim, head_dim, bias=False, with_qk_norm=True, attn_type='torch'):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim
        
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wkv = nn.Linear(hidden_dim, hidden_dim*2, bias=bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)
        
        self.core_attention = self.attn_processor(attn_type=attn_type)

    def forward(
            self, 
            x: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attn_mask=None
        ):
        xq = self.wq(x) 
        xq = xq.view(*xq.shape[:-1], self.n_heads, self.head_dim)
        
        xkv = self.wkv(encoder_hidden_states)
        xkv = xkv.view(*xkv.shape[:-1], self.n_heads, 2*self.head_dim)

        xk, xv = torch.split(xkv, [self.head_dim]*2, dim=-1)  ## seq_len, n, dim
    
        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=attn_mask
                )
        
        output = rearrange(output, 'b s h d -> b s (h d)')
        output = self.wo(output)
        
        return output

    
class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
    
class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int,
        inner_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim*mult if inner_dim is None else inner_dim
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.ModuleList([
            GELU(dim, inner_dim, approximate="tanh", bias=bias),
            nn.Identity(),
            nn.Linear(inner_dim, dim_out, bias=bias)
        ])
        
        
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
    

def modulate(x, scale, shift):
    x = x * (1 + scale) + shift
    return x


def gate(x, gate):
    x = gate * x
    return x


class StepVideoTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        attention_head_dim: int,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = False,
        attention_type: str = 'parallel'
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = SelfAttention(dim, attention_head_dim, bias=False, with_rope=True, with_qk_norm=True, attn_type=attention_type)
        
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = CrossAttention(dim, attention_head_dim, bias=False, with_qk_norm=True, attn_type='torch')

        self.ff = FeedForward(dim=dim, inner_dim=ff_inner_dim, dim_out=dim, bias=ff_bias)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) /dim**0.5)

    @torch.no_grad()
    def forward(
        self,
        q: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] =  None,
        attn_mask = None,
        rope_positions: list = None, 
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            torch.clone(chunk) for chunk in (self.scale_shift_table[None].to(dtype=q.dtype, device=q.device) + timestep.reshape(-1, 6, self.dim)).chunk(6, dim=1)
        )
        
        scale_shift_q = modulate(self.norm1(q), scale_msa, shift_msa)

        attn_q = self.attn1(
            scale_shift_q,
            rope_positions=rope_positions
        )

        q = gate(attn_q, gate_msa) + q
        
        attn_q = self.attn2(
                q,
                kv,
                attn_mask
            )

        q = attn_q + q

        scale_shift_q = modulate(self.norm2(q), scale_mlp, shift_mlp)

        ff_output = self.ff(scale_shift_q)
        
        q = gate(ff_output, gate_mlp) + q
        
        return q
    
    
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=64,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, latent):
        latent = self.proj(latent).to(latent.dtype)   
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        return latent


class StepVideoModel(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        dropout: float = 0.0,
        patch_size: int = 1,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[Union[int, List, Tuple]] = [6144, 1024],
        attention_type: Optional[str] = "torch",
    ):
        super().__init__()

        # Set some common variables used across the board.
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels

        self.use_additional_conditions = use_additional_conditions

        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                StepVideoTransformerBlock(
                    dim=self.inner_dim,
                    attention_head_dim=attention_head_dim,
                    attention_type=attention_type
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.patch_size = patch_size

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )

        if isinstance(caption_channels, int):
            caption_channel = caption_channels
        else:
            caption_channel, clip_channel = caption_channels
            self.clip_projection = nn.Linear(clip_channel, self.inner_dim) 

        self.caption_norm = nn.LayerNorm(caption_channel,  eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channel, hidden_size=self.inner_dim
        )
        
        self.parallel = attention_type=='parallel'

    def patchfy(self, hidden_states):
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.zeros([len(kv_seqlens), q_seqlen, max(kv_seqlens)], dtype=torch.bool, device=encoder_attention_mask.device)
        encoder_hidden_states = encoder_hidden_states[:,: max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask
        
        
    def block_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        rope_positions=None,
        attn_mask=None,
        parallel=True
    ):
        for block in tqdm(self.transformer_blocks, desc="Transformer blocks"):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                attn_mask=attn_mask,
                rope_positions=rope_positions
            )

        return hidden_states
        

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor=None,
        return_dict: bool = False,
    ):
        assert hidden_states.ndim==5; "hidden_states's shape should be (bsz, f, ch, h ,w)"

        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
                
        hidden_states = self.patchfy(hidden_states) 
        len_frame = hidden_states.shape[1]
                
        if self.use_additional_conditions:
            added_cond_kwargs = {
                "resolution": torch.tensor([(height, width)]*bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "nframe": torch.tensor([frame]*bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                "fps": fps
            }    
        else:
            added_cond_kwargs = {}
        
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs
        )

        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))
        
        if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
            clip_embedding = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

        hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()
        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask, encoder_hidden_states, q_seqlen=frame*len_frame)
        
        hidden_states = self.block_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            rope_positions=[frame, height, width],
            attn_mask=attn_mask,
            parallel=self.parallel
        )
        
        hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)
        
        embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()
        
        shift, scale = (self.scale_shift_table[None].to(dtype=embedded_timestep.dtype, device=embedded_timestep.device) + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        
        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        
        hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)

        if return_dict:
            return {'x': output}
        return output
    
    @staticmethod
    def state_dict_converter():
        return StepVideoDiTStateDictConverter()


class StepVideoDiTStateDictConverter:
    def __init__(self):
        super().__init__()

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        return state_dict

    
    