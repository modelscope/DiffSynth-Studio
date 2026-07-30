import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
from ..core.device.npu_compatible_device import get_device_type


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    assert timesteps.ndim == 1
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
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Parameter-free sinusoidal timestep projection."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0, scale: int = 1):
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
    """Two-layer timestep MLP (act_fn='silu')."""

    def __init__(self, in_channels: int, time_embed_dim: int, sample_proj_bias: bool = True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to `(B, S, H, D)` attention tensors."""
    with torch.amp.autocast(get_device_type(), enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 RoPE table indexed by 3D position ids. Holds no persistent state."""

    def __init__(self, axes_dims: Tuple[int, ...], axes_lens: Tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: Tuple[int, ...], end: Tuple[int, ...], theta: float):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (S, 3) int -> (S, head_dim/2) complex64
        device = position_ids.device
        max_vals = position_ids.max(dim=0).values.tolist()
        needs_rebuild = self.freqs_cis is None or any(m >= l for m, l in zip(max_vals, self.axes_lens))
        if needs_rebuild:
            for i in range(len(self.axes_lens)):
                if max_vals[i] >= self.axes_lens[i]:
                    self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, tuple(self.axes_lens), theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def make_joint_position_ids(text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    """3D positions in [video; text] order. Text t-axis is 1..text_len; video t-axis starts at text_len+1."""
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack([text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1)
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


def _cat_interleave(a: torch.Tensor, len_a: list, b: torch.Tensor, len_b: list) -> torch.Tensor:
    a_split = torch.split(a, len_a, dim=1)
    b_split = torch.split(b, len_b, dim=1)
    blocks = []
    for x_part, text_part in zip(a_split, b_split):
        blocks.extend([x_part, text_part])
    return torch.cat(blocks, dim=1)


class LingBotVideoTextEmbedder(nn.Module):
    """RMSNorm(text_dim) -> Linear -> SiLU -> Linear."""

    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)

    def forward(self, x, rotary_emb, attention_mask=None):
        q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)
        # q/k/v are (B, S, H, D).
        out = attention_forward(
            q, k, v,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=attention_mask,
        )
        return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, score_func, norm_topk_prob, n_group, topk_group, route_scale):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)

    def _group_limited_topk(self, scores_for_choice):
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(seq_len, self.n_group, experts_per_group).reshape(seq_len, -1)
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, tokens: torch.Tensor):
        with torch.amp.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype)


class LingBotVideoGroupedExperts(nn.Module):
    """Weight layout: w1 [E,I,H], w2 [E,H,I], w3 [E,I,H]."""

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class LingBotVideoSparseMoeBlock(nn.Module):
    """MoE FFN with a grouped_mm expert path and a per-expert for-loop fallback."""

    def __init__(self, hidden_size, intermediate_size, num_experts, top_k, moe_intermediate_size,
                 score_func, norm_topk_prob, n_group, topk_group, routed_scaling_factor, n_shared_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = LingBotVideoRouter(
            hidden_size, num_experts, top_k, score_func, norm_topk_prob, n_group, topk_group, routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_experts = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(hidden_size, moe_intermediate_size * n_shared_experts)

    @staticmethod
    def _reorder_tokens(tokens, top_scores, top_indices, num_experts):
        num_tokens = tokens.shape[0]
        top_k = top_indices.shape[1]
        flat_scores = top_scores.reshape(-1)
        flat_indices = top_indices.reshape(-1)
        active_positions = torch.where(flat_scores != 0)[0]
        active_experts = flat_indices[active_positions]

        counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
        counts.scatter_add_(0, active_experts, torch.ones_like(active_experts, dtype=torch.int64))

        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        sorted_scores = flat_scores[sorted_positions]
        original_token_idx = sorted_positions // top_k
        permuted_tokens = tokens[original_token_idx]
        return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k

    @staticmethod
    def _pad_grouped_tokens(tokens, counts, align: int = 8):
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_len = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts = ((total_per_expert + align - 1) // align * align).to(torch.int32)
        write_offsets = torch.cumsum(aligned_counts, dim=0) - aligned_counts
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

        fill_value = num_tokens
        permuted_indices = torch.full((max_len,), fill_value, dtype=torch.int64, device=tokens.device)
        for expert_idx in range(num_experts):
            length = int(counts_i64[expert_idx].item())
            if length == 0:
                continue
            write_start = int(write_offsets[expert_idx].item())
            start = int(start_indices[expert_idx].item())
            permuted_indices[write_start:write_start + length] = torch.arange(
                start, start + length, device=tokens.device, dtype=torch.int64
            )

        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return input_shape, tokens_with_pad[permuted_indices], permuted_indices, aligned_counts

    @staticmethod
    def _unpad_grouped_tokens(output, input_shape, permuted_indices):
        unpermuted = output.new_empty(input_shape)
        unpermuted[permuted_indices, :] = output
        return unpermuted[:-1]

    def _run_grouped_experts(self, tokens, counts):
        if not hasattr(torch, "_grouped_mm"):
            return self._run_experts_for_loop(tokens, counts)
        input_shape, padded_tokens, permuted_indices, aligned_counts = self._pad_grouped_tokens(tokens, counts)
        offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
        h = F.silu(torch._grouped_mm(padded_tokens.bfloat16(), self.experts.w1.bfloat16().transpose(-2, -1), offs=offsets))
        h = h * torch._grouped_mm(padded_tokens.bfloat16(), self.experts.w3.bfloat16().transpose(-2, -1), offs=offsets)
        out = torch._grouped_mm(h, self.experts.w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(padded_tokens)
        return self._unpad_grouped_tokens(out, input_shape, permuted_indices)

    def _run_experts_for_loop(self, tokens, counts):
        count_list = counts.tolist()
        splits = torch.split(tokens, count_list, dim=0)
        outputs = []
        for expert_idx, expert_tokens in enumerate(splits):
            if expert_tokens.numel() == 0:
                continue
            h = F.silu(expert_tokens @ self.experts.w1[expert_idx].transpose(-2, -1))
            h = h * (expert_tokens @ self.experts.w3[expert_idx].transpose(-2, -1))
            h = h @ self.experts.w2[expert_idx].transpose(-2, -1)
            outputs.append(h)
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _restore_tokens(expert_output, sorted_positions, sorted_scores, num_tokens, top_k):
        dim = expert_output.shape[-1]
        unsorted = torch.zeros((num_tokens * top_k, dim), dtype=expert_output.dtype, device=expert_output.device)
        unsorted[sorted_positions] = expert_output
        unsorted = unsorted.reshape(num_tokens, top_k, dim)

        scores_unsorted = torch.zeros(num_tokens * top_k, dtype=sorted_scores.dtype, device=sorted_scores.device)
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        return (unsorted.float() * scores_unsorted).sum(dim=1).to(expert_output.dtype)

    def _run_selected_experts(self, tokens, top_scores, top_indices):
        permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k = self._reorder_tokens(
            tokens, top_scores, top_indices, self.router.num_experts
        )
        expert_output = self._run_grouped_experts(permuted_tokens, counts)
        return self._restore_tokens(expert_output, sorted_positions, sorted_scores, num_tokens, top_k)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # hidden_states: (B, S, H); padding_mask: (B*S,) with 1=valid (only needed when B>1)
        B = hidden_states.shape[0]
        tokens = hidden_states.view(-1, self.hidden_size)
        top_indices, top_scores = self.router(tokens)
        if padding_mask is not None:
            pm = padding_mask.unsqueeze(-1).to(top_scores.dtype)
            top_scores = top_scores * pm
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)
            top_scores = top_scores * self.router.route_scale

        out = self._run_selected_experts(tokens, top_scores, top_indices)

        out = out.view(B, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class LingBotVideoBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, norm_eps, qkv_bias, out_bias,
                 num_experts, num_experts_per_tok, moe_intermediate_size, decoder_sparse_step, mlp_only_layers,
                 n_shared_experts, score_func, norm_topk_prob, n_group, topk_group, routed_scaling_factor, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(h, num_attention_heads, norm_eps, qkv_bias, out_bias)
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
        # Sparsity decision: mlp_only_layers + decoder_sparse_step + num_experts.
        if layer_idx not in mlp_only_layers and (num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0):
            self.ffn = LingBotVideoSparseMoeBlock(
                h, intermediate_size, num_experts, num_experts_per_tok, moe_intermediate_size,
                score_func, norm_topk_prob, n_group, topk_group, routed_scaling_factor, n_shared_experts,
            )
        else:
            self.ffn = LingBotVideoMLP(h, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

    def forward(self, x, temb6, rotary_emb, attention_mask=None, moe_padding_mask=None):
        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape (B*S, 6D); "
                f"got {tuple(temb6.shape)} for hidden states {tuple(x.shape)}."
            )
        mod = temb6.view(x.shape[0], x.shape[1], -1) + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        bulk_dtype = x.dtype
        attn_in = (self.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(attn_in, rotary_emb, attention_mask)
        x = x + (gate_msa * self.norm_post_attn(attn_out)).to(x.dtype)

        ffn_in = (self.norm2(x) * scale_mlp + shift_mlp).to(bulk_dtype)
        if isinstance(self.ffn, LingBotVideoSparseMoeBlock):
            ffn_out = self.ffn(ffn_in, padding_mask=moe_padding_mask)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + (gate_mlp * self.norm_post_ffn(ffn_out)).to(x.dtype)
        return x


class LingBotVideoDiT(nn.Module):
    """LingBot-Video MoE DiT.

    Supports both the Dense (`num_experts=0`, FFN = MLP) and MoE
    (`num_experts>0`, FFN = sparse MoE) variants from a single class.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["LingBotVideoBlock"]
    _repeated_blocks = ["LingBotVideoBlock"]

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: Tuple[int, int, int] = (32, 48, 48),
        axes_lens: Tuple[int, int, int] = (8192, 1024, 1024),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Tuple[int, ...] = (),
        n_shared_experts: Optional[int] = None,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"
        mlp_only_layers = tuple(mlp_only_layers)

        # Config attributes used by forward / pipeline.
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.depth = depth
        self.gradient_checkpointing = False

        self.patch_embedder = nn.Linear(in_channels * math.prod(patch_size), hidden_size, bias=patch_embed_bias)
        self.time_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(freq_dim, hidden_size, sample_proj_bias=timestep_mlp_bias)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(axes_dims, axes_lens, rope_theta)
        self.blocks = nn.ModuleList([
            LingBotVideoBlock(
                hidden_size=hidden_size, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                norm_eps=norm_eps, qkv_bias=qkv_bias, out_bias=out_bias, num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok, moe_intermediate_size=moe_intermediate_size,
                decoder_sparse_step=decoder_sparse_step, mlp_only_layers=mlp_only_layers,
                n_shared_experts=n_shared_experts, score_func=score_func, norm_topk_prob=norm_topk_prob,
                n_group=n_group, topk_group=topk_group, routed_scaling_factor=routed_scaling_factor, layer_idx=i,
            )
            for i in range(depth)
        ])
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,             # (B, C, T, H, W)
        timestep: torch.Tensor,                  # (B,) in [0, 1000] (= sigma * 1000)
        encoder_hidden_states: torch.Tensor,     # (B, L, text_dim)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, L) 1=valid
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is not None:
            text_lens = encoder_attention_mask.sum(dim=-1).long()
        else:
            text_lens = torch.full((B,), L, dtype=torch.long, device=device)
        text_lens_list = [int(v) for v in text_lens.detach().cpu().tolist()]
        packed_batch = B > 1

        # patchify: token order (f h w), feature order (pf ph pw c)
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, n_video, pF * pH * pW * C)

        if packed_batch:
            x = torch.cat([self.patch_embedder(patch_tokens[i:i + 1]) for i in range(B)], dim=1)
            text_parts = [self.text_embedder(encoder_hidden_states[i:i + 1, :text_lens_list[i], :]) for i in range(B)]
            text = torch.cat(text_parts, dim=1)
            joint = _cat_interleave(x, [n_video] * B, text, text_lens_list)
        else:
            x = self.patch_embedder(patch_tokens)
            text = self.text_embedder(encoder_hidden_states)
            joint = torch.cat([x, text], dim=1)  # [video; text]
        joint_seq_len = joint.shape[1]

        # Per-sample RoPE: video t-axis start = real text length of this sample + 1.
        rotary_parts = [self.rope(make_joint_position_ids(text_lens_list[i], gt, gh, gw, device)) for i in range(B)]
        if packed_batch:
            rotary = torch.cat(rotary_parts, dim=0).unsqueeze(0)
        else:
            rotary = torch.stack(rotary_parts, dim=0)  # (B, S, head_dim/2) complex64

        attention_mask = None
        moe_padding_mask = None
        if packed_batch:
            # Block-diagonal mask so packed samples only attend within their own block.
            sample_seq_lens = [n_video + tl for tl in text_lens_list]
            total = sum(sample_seq_lens)
            block_mask = torch.zeros((total, total), dtype=torch.bool, device=device)
            start = 0
            for slen in sample_seq_lens:
                block_mask[start:start + slen, start:start + slen] = True
                start += slen
            attention_mask = block_mask[None, None, :, :]  # (1,1,total,total)
        else:
            has_padding = encoder_attention_mask is not None and bool((text_lens < L).any())
            if has_padding:
                key_mask = torch.cat(
                    [torch.ones(B, n_video, dtype=torch.bool, device=device), encoder_attention_mask.bool()], dim=1
                )
                attention_mask = key_mask[:, None, None, :]      # (B,1,1,S) -> SDPA broadcast
                moe_padding_mask = key_mask.reshape(-1).float()  # (B*S,)

        # Timestep -> per-token modulation.
        timestep_proj = self.time_proj(timestep.float())
        t_emb = self.time_embedder(timestep_proj.to(joint.dtype))  # (B, D)
        if packed_batch:
            temb_input = torch.cat(
                [t_emb[i:i + 1].unsqueeze(1).expand(1, sample_seq_lens[i], -1) for i in range(B)], dim=1
            )  # (1, total, D)
        else:
            temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)  # (B, S, D)
        b_eff, s_eff = temb_input.shape[0], temb_input.shape[1]
        temb6 = self.time_modulation(temb_input.reshape(b_eff * s_eff, -1))  # (B*S, 6D)

        for block in self.blocks:
            joint = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=joint, temb6=temb6, rotary_emb=rotary,
                attention_mask=attention_mask, moe_padding_mask=moe_padding_mask,
            )

        final_mod = self.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(joint.dtype))

        if packed_batch:
            split_lengths = []
            for tl in text_lens_list:
                split_lengths.extend([n_video, tl])
            parts = torch.split(projected, split_lengths, dim=1)
            x = torch.cat(parts[::2], dim=1).reshape(B, n_video, -1)
        else:
            x = projected[:, :n_video]

        # unpatchify
        Cout = self.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)
        return x
