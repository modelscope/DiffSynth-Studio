import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general_modules import RMSNorm
from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward


ADALN_EMBED_DIM = 256


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(t.dtype))


def select_per_token(value_noisy, value_clean, noise_mask, seq_len):
    noise_mask = noise_mask.unsqueeze(-1)
    return torch.where(
        noise_mask == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        if noise_mask is not None:
            scale = select_per_token(
                1.0 + self.adaLN_modulation(c_noisy),
                1.0 + self.adaLN_modulation(c_clean),
                noise_mask,
                x.shape[1],
            )
        else:
            scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)
        return self.linear(self.norm_final(x) * scale)


class RopeEmbedder:
    def __init__(self, theta: float = 256.0, axes_dims=(32, 48, 48), axes_lens=(2048, 512, 512)):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
                timestep = torch.arange(e, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def __call__(self, ids: torch.Tensor):
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = [x.to(device) for x in self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [x.to(device) for x in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def compute_gist_count(text_lengths, min_gist, max_gist, gist_step, min_text_len, max_text_len):
    span = max_text_len - min_text_len
    counts = []
    for n in text_lengths:
        ratio = min(1.0, max(0.0, (float(n) - min_text_len) / span))
        raw = min_gist + ratio * (max_gist - min_gist)
        counts.append(max(min_gist, min(max_gist, int(math.ceil(raw / gist_step)) * gist_step)))
    return counts


class FiboAttention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, kv_dim=None, eps=1e-5):
        super().__init__()
        self.heads = heads
        kv_dim = kv_dim or query_dim
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    @staticmethod
    def apply_rotary_emb(x, freqs_cis):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(x_ * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
        query = self.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.norm_q(query.unflatten(-1, (self.heads, -1)))
        key = self.norm_k(key.unflatten(-1, (self.heads, -1)))
        value = value.unflatten(-1, (self.heads, -1))
        if freqs_cis is not None:
            query = self.apply_rotary_emb(query, freqs_cis)
            key = self.apply_rotary_emb(key, freqs_cis)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=attention_mask,
        )
        return self.to_out[0](hidden_states.flatten(2, 3))


class FiboTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps, cross_attention_dim=None):
        super().__init__()
        self.attention = FiboAttention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.feed_forward = FeedForward(dim, (dim * 8 // 3) // 64 * 64)
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps=norm_eps)
            self.cross_attention = FiboAttention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attention = None

    def forward(
        self,
        x,
        attn_mask=None,
        freqs_cis=None,
        adaln_input=None,
        noise_mask=None,
        adaln_noisy=None,
        adaln_clean=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if noise_mask is not None:
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)
            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)
            gate_msa = select_per_token(gate_msa_noisy.tanh(), gate_msa_clean.tanh(), noise_mask, x.shape[1])
            gate_mlp = select_per_token(gate_mlp_noisy.tanh(), gate_mlp_clean.tanh(), noise_mask, x.shape[1])
            scale_msa = select_per_token(1 + scale_msa_noisy, 1 + scale_msa_clean, noise_mask, x.shape[1])
            scale_mlp = select_per_token(1 + scale_mlp_noisy, 1 + scale_mlp_clean, noise_mask, x.shape[1])
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)
        if self.cross_attention is not None:
            cross_out = self.cross_attention(
                self.cross_attn_norm1(x),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            x = x + self.cross_attn_gate.tanh() * self.cross_attn_norm2(cross_out)
        return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))


class FiboPerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim, norm_eps):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention = FiboAttention(dim, n_heads, dim // n_heads, kv_dim=cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.self_attention = FiboAttention(dim, n_heads, dim // n_heads, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, (dim * 8 // 3) // 64 * 64)

    def forward(self, latents, encoder_hidden_states, encoder_attention_mask=None, attention_mask=None):
        latents = latents + self.cross_attention(
            self.cross_attn_norm(latents), encoder_hidden_states, encoder_attention_mask
        )
        latents = latents + self.self_attention(self.self_attn_norm(latents), attention_mask=attention_mask)
        return latents + self.ffn(self.ffn_norm(latents))


class FiboPerceiverResampler(nn.Module):
    def __init__(self, dim, n_heads, cross_attention_dim, num_queries, num_layers, norm_eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList(
            [FiboPerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)]
        )
        self.norm_out = RMSNorm(dim, eps=norm_eps)

    def forward(self, encoder_hidden_states, num_gist_tokens, encoder_attention_mask=None, attention_mask=None):
        latents = self.latent_queries[:num_gist_tokens].unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, encoder_hidden_states, encoder_attention_mask, attention_mask)
        return self.norm_out(latents)


class FiboDiT(nn.Module):

    _repeated_blocks = ["FiboTransformerBlock"]

    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: Tuple[int, ...] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        min_num_gist_tokens: int = 256,
        max_num_gist_tokens: int = 512,
        gist_step: int = 64,
        gist_min_text_len: int = 256,
        gist_max_text_len: int = 2048,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: Tuple[int, ...] = (32, 48, 48),
        axes_lens: Tuple[int, ...] = (2048, 512, 512),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.cap_feat_dim = cap_feat_dim
        self.injection_layer_ids = tuple(injection_layer_ids)
        self.min_num_gist_tokens = min_num_gist_tokens
        self.max_num_gist_tokens = max_num_gist_tokens
        self.gist_step = gist_step
        self.gist_min_text_len = gist_min_text_len
        self.gist_max_text_len = gist_max_text_len
        self.t_scale = t_scale

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.rope_embedder = RopeEmbedder(rope_theta, axes_dims, axes_lens)
        self.noise_refiner = nn.ModuleList(
            [FiboTransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)]
        )
        self.perceiver = FiboPerceiverResampler(
            dim, n_heads, cap_feat_dim, max_num_gist_tokens, perceiver_num_layers, norm_eps
        )
        self.layers = nn.ModuleList(
            [
                FiboTransformerBlock(dim, n_heads, norm_eps, cap_feat_dim if i in self.injection_layer_ids else None)
                for i in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size * patch_size * in_channels)

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        start = (0 for _ in size) if start is None else start
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _patchify(self, latents):
        b, c, h, w = latents.shape
        p = self.patch_size
        latents = latents.view(b, c, h // p, p, w // p, p)
        return latents.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), -1)

    def _unpatchify(self, tokens, height, width):
        b, _, c = tokens.shape
        p = self.patch_size
        tokens = tokens.view(b, height // p, width // p, c // p ** 2, p, p)
        return tokens.permute(0, 3, 1, 4, 2, 5).reshape(b, c // p ** 2, height, width)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        context_latents: Optional[List[torch.Tensor]] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        b, _, height, width = hidden_states.shape
        p = self.patch_size
        h_tokens, w_tokens = height // p, width // p
        num_image_tokens = h_tokens * w_tokens
        device = hidden_states.device

        adaln_noisy = self.t_embedder(timestep * self.t_scale)
        adaln_clean = self.t_embedder(torch.zeros_like(timestep))

        image_plane = self.max_num_gist_tokens + 1
        tokens = [self._patchify(hidden_states)]
        ids = [self.create_coordinate_grid((1, h_tokens, w_tokens), (image_plane, 0, 0), device)]
        for k, latents in enumerate(context_latents or []):
            tokens.append(self._patchify(latents))
            size = (1, latents.shape[2] // p, latents.shape[3] // p)
            ids.append(self.create_coordinate_grid(size, (image_plane + k + 1, 0, 0), device))
        hidden_states = self.x_embedder(torch.cat(tokens, dim=1))
        image_freqs = self.rope_embedder(torch.cat([grid.flatten(0, 2) for grid in ids])).unsqueeze(0)

        refiner_mask = None
        if context_latents:
            refiner_mask = torch.zeros(b, hidden_states.shape[1], dtype=torch.long, device=device)
            refiner_mask[:, :num_image_tokens] = 1
        for block in self.noise_refiner:
            hidden_states = gradient_checkpoint_forward(
                block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                hidden_states, None, image_freqs, adaln_noisy, refiner_mask, adaln_noisy, adaln_clean,
            )

        if encoder_attention_mask is not None:
            text_lengths = encoder_attention_mask.sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[2]] * b
        gist_counts = compute_gist_count(
            text_lengths, self.min_num_gist_tokens, self.max_num_gist_tokens,
            self.gist_step, self.gist_min_text_len, self.gist_max_text_len,
        )
        num_gist_tokens = max(gist_counts)
        gist_mask = None
        if min(gist_counts) < num_gist_tokens:
            gist_mask = torch.arange(num_gist_tokens, device=device) < torch.tensor(gist_counts, device=device)[:, None]
        gist = self.perceiver(encoder_hidden_states[:, 0], num_gist_tokens, encoder_attention_mask, gist_mask)
        gist_ids = self.create_coordinate_grid((num_gist_tokens, 1, 1), (1, 0, 0), device).flatten(0, 2)
        gist_freqs = self.rope_embedder(gist_ids).unsqueeze(0)

        attention_mask = None
        if gist_mask is None:
            image_start = num_gist_tokens
            hidden_states = torch.cat([gist, hidden_states], dim=1)
            freqs_cis = torch.cat([gist_freqs, image_freqs], dim=1)
        else:
            image_start = 0
            hidden_states = torch.cat([hidden_states, gist], dim=1)
            freqs_cis = torch.cat([image_freqs, gist_freqs], dim=1)
            image_mask = torch.ones(b, hidden_states.shape[1] - num_gist_tokens, dtype=torch.bool, device=device)
            attention_mask = torch.cat([image_mask, gist_mask], dim=1)
        noise_mask = torch.zeros(b, hidden_states.shape[1], dtype=torch.long, device=device)
        noise_mask[:, image_start:image_start + num_image_tokens] = 1

        bundle_index = {layer_id: index + 1 for index, layer_id in enumerate(self.injection_layer_ids)}
        for layer_id, block in enumerate(self.layers):
            text = encoder_hidden_states[:, bundle_index[layer_id]] if layer_id in bundle_index else None
            hidden_states = gradient_checkpoint_forward(
                block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                hidden_states, attention_mask, freqs_cis, None, noise_mask, adaln_noisy, adaln_clean,
                text, encoder_attention_mask,
            )

        hidden_states = self.final_layer(
            hidden_states[:, image_start:image_start + num_image_tokens], c=adaln_noisy
        )
        return self._unpatchify(hidden_states, height, width)
