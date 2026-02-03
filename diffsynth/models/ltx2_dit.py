import math
import functools
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple, Callable
import numpy as np
import torch
from einops import rearrange
from .ltx2_common import rms_norm, Modality
from ..core.attention.attention import attention_forward


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
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


class TimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim

        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb


class PerturbationType(Enum):
    """Types of attention perturbations for STG (Spatio-Temporal Guidance)."""

    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False

        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        return PerturbationConfig([])


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """Perturbation configurations for a batch, with utilities for generating attention masks."""

    perturbations: list[PerturbationConfig]

    def mask(
        self, perturbation_type: PerturbationType, block: int, device, dtype: torch.dtype
    ) -> torch.Tensor:
        mask = torch.ones((len(self.perturbations),), device=device, dtype=dtype)
        for batch_idx, perturbation in enumerate(self.perturbations):
            if perturbation.is_perturbed(perturbation_type, block):
                mask[batch_idx] = 0

        return mask

    def mask_like(self, perturbation_type: PerturbationType, block: int, values: torch.Tensor) -> torch.Tensor:
        mask = self.mask(perturbation_type, block, values.device, values.dtype)
        return mask.view(mask.numel(), *([1] * len(values.shape[1:])))

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return all(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty(batch_size: int) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig([PerturbationConfig.empty() for _ in range(batch_size)])


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")



def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Reshape for attention_forward using unflatten
        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        out = attention_forward(
            q=q,
            k=k,
            v=v,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            attn_mask=mask
        )

        # Reshape back to original format
        out = out.flatten(2, 3)
        return self.to_out(out)


class PixArtAlphaTextProjection(torch.nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool



class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type

    def _prepare_timestep(
        self, timestep: torch.Tensor, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""

        timestep = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )

        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare context for transformer blocks."""
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = context.view(batch_size, -1, x.shape[-1])

        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], modality.latent.dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: PixArtAlphaTextProjection,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            caption_projection=caption_projection,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=modality.timesteps,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(6, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, slice(None, None)
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        batch_size = video.x.shape[0]
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
                norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                vx = vx + self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa * v_mask

            vx = vx + self.attn2(rms_norm(vx, eps=self.norm_eps), context=video.context, mask=video.context_mask)

            del vshift_msa, vscale_msa, vgate_msa

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            if not perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx):
                norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                a_mask = perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings) * agate_msa * a_mask

            ax = ax + self.audio_attn2(rms_norm(ax, eps=self.norm_eps), context=audio.context, mask=audio.context_mask)

            del ashift_msa, ascale_msa, agate_msa

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )

            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )

            del gate_out_a2v, gate_out_v2a
            del (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
            )

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = [20, 2048, 2048],
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = [20],
        av_ca_timestep_scale_multiplier: int = 1000,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
        )

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)

        self.adaln_single = AdaLayerNormSingle(self.inner_dim)

        # Video caption projection
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
        )

        # Audio caption projection
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def _forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None
        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    def forward(self, video_latents, video_positions, video_context, video_timesteps, audio_latents, audio_positions, audio_context, audio_timesteps):
        cross_pe_max_pos = None
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
        self._init_preprocessors(cross_pe_max_pos)
        video = Modality(video_latents, video_timesteps, video_positions, video_context)
        audio = Modality(audio_latents, audio_timesteps, audio_positions, audio_context)
        vx, ax = self._forward(video=video, audio=audio, perturbations=None)
        return vx, ax
