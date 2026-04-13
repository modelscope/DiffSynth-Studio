"""
Ernie-Image DiT for DiffSynth-Studio.

Refactored from diffusers ErnieImageTransformer2DModel to use DiffSynth core modules.
Default parameters from actual checkpoint config.json (baidu/ERNIE-Image transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
from .flux2_dit import Timesteps, TimestepEmbedding


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        emb = emb.unsqueeze(2)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()


class ErnieImageSingleStreamAttnProcessor:
    def __call__(
        self,
        attn: "ErnieImageAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            rot_dim = freqs_cis.shape[-1]
            x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
            cos_ = torch.cos(freqs_cis).to(x.dtype)
            sin_ = torch.sin(freqs_cis).to(x.dtype)
            x1, x2 = x.chunk(2, dim=-1)
            x_rotated = torch.cat((-x2, x1), dim=-1)
            return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            attn_mask=attention_mask,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        output = attn.to_out[0](hidden_states)

        return output


class ErnieImageAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: str = "rms_norm",
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'rms_norm'."
            )

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))

        self.processor = ErnieImageSingleStreamAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb)


class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


class ErnieImageRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * self.weight
        return hidden_states.to(input_dtype)


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.adaLN_sa_ln = ErnieImageRMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            qk_norm="rms_norm" if qk_layernorm else None,
            eps=eps,
            bias=False,
            out_bias=False,
        )
        self.adaLN_mlp_ln = ErnieImageRMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        temb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb
        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
        x_bsh = x.permute(1, 0, 2)
        attn_out = self.self_attention(x_bsh, attention_mask=attention_mask, image_rotary_emb=rotary_pos_emb)
        attn_out = attn_out.permute(1, 0, 2)
        x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)
        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
        return residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class ErnieImageDiT(nn.Module):
    """
    Ernie-Image DiT model for DiffSynth-Studio.

    Architecture: SharedAdaLN + RoPE 3D + Joint Image-Text Attention.
    Internal format: [S, B, H] for transformer blocks, [B, S, H] for attention.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: Tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.layers = nn.ModuleList([
            ErnieImageSharedAdaLNBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm=qk_layernorm)
            for _ in range(num_layers)
        ])
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()

        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()

        x = torch.cat([img_sbh, text_sbh], dim=0)
        S = x.shape[0]

        text_ids = torch.cat([
            torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
            torch.zeros((B, Tmax, 2), device=device)
        ], dim=-1) if Tmax > 0 else torch.zeros((B, 0, 3), device=device)
        grid_yx = torch.stack(
            torch.meshgrid(torch.arange(Hp, device=device, dtype=torch.float32),
                           torch.arange(Wp, device=device, dtype=torch.float32), indexing="ij"),
            dim=-1
        ).reshape(-1, 2)
        image_ids = torch.cat([
            text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),
            grid_yx.view(1, N_img, 2).expand(B, -1, -1)
        ], dim=-1)
        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        valid_text = torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1) if Tmax > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool)
        attention_mask = torch.cat([
            torch.ones((B, N_img), device=device, dtype=torch.bool),
            valid_text
        ], dim=1)[:, None, None, :]

        sample = self.time_proj(timestep.to(dtype))
        sample = sample.to(self.time_embedding.linear_1.weight.dtype)
        c = self.time_embedding(sample)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(0).expand(S, -1, -1).contiguous()
            for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]

        for layer in self.layers:
            temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            if torch.is_grad_enabled() and use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    layer,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    rotary_pos_emb,
                    temb,
                    attention_mask,
                )
            else:
                x = layer(x, rotary_pos_emb, temb, attention_mask)

        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
        output = patches.view(B, Hp, Wp, p, p, self.out_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, self.out_channels, H, W)

        return output
