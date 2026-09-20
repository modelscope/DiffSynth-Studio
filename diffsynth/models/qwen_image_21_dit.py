import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.attention import FLEX_ATTN_AVAILABLE, attention_forward
from ..core.gradient import gradient_checkpoint_forward


_FLEX_BLOCK_SIZE = 128
BlockMask, create_block_mask = None, None
if FLEX_ATTN_AVAILABLE:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.max_period = max_period
        self.time_factor = time_factor

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        half = self.timestep_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        args = timestep[:, None] * freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, sample_proj_bias: bool = True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = QwenImage21TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


def build_qwenimage21_block_causal_mask(
    image_ids: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
):
    seq_len = image_ids.shape[0]
    padded_seq_len = int(math.ceil(seq_len / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE)

    image_ids = F.pad(image_ids, (0, padded_seq_len - seq_len), value=-1)
    if encoder_hidden_states_mask is None:
        key_valid = torch.ones(batch_size, padded_seq_len, dtype=torch.bool, device=device)
    else:
        key_valid = F.pad(encoder_hidden_states_mask.bool(), (0, padded_seq_len - seq_len), value=False)

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        is_padding = (q_idx >= seq_len) | (kv_idx >= seq_len)
        q_image_id, kv_image_id = image_ids[q_idx], image_ids[kv_idx]
        same_image_block = (q_image_id == kv_image_id) & (q_image_id >= 0)
        allowed = ((q_idx >= kv_idx) | same_image_block) & key_valid[batch_idx, kv_idx]
        return allowed & ~is_padding

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=padded_seq_len,
        KV_LEN=padded_seq_len,
        device=device,
        _compile=False,
    )


def _qwenimage21_prefix_segments(image_ids: torch.Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _attention(query, key, value, attn_mask=None, use_flex=False):
    return attention_forward(
        query, key, value, q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        attn_mask=attn_mask, use_flex=use_flex,
    )


class QwenImage21AttnProcessor:
    def __call__(
        self,
        attn: "QwenImage21Attention",
        hidden_states: torch.Tensor,
        attention_mask: Any | None = None,
        rotary_emb: torch.Tensor | None = None,
        kv_cache: dict[str, torch.Tensor] | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)

        if rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, rotary_emb, use_real=False)
            key = apply_rotary_emb_qwen(key, rotary_emb, use_real=False)

        decode = kv_cache is not None and "key" in kv_cache
        if kv_cache is not None:
            if decode:
                key = torch.cat([kv_cache["key"], key], dim=1)
                value = torch.cat([kv_cache["value"], value], dim=1)
            else:
                kv_cache["key"] = key[:, cache_write_slice].clone()
                kv_cache["value"] = value[:, cache_write_slice].clone()

        seq_len_q, seq_len_kv = query.shape[1], key.shape[1]
        if decode:
            # Fast decode path with kv cache
            hidden_states = _attention(query, key, value, attn_mask=attention_mask)
        elif FLEX_ATTN_AVAILABLE and isinstance(attention_mask, BlockMask):
            # Flex attention route on the first step or without kv cache
            pad_q = int(math.ceil(seq_len_q / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE) - seq_len_q
            pad_kv = int(math.ceil(seq_len_kv / _FLEX_BLOCK_SIZE) * _FLEX_BLOCK_SIZE) - seq_len_kv
            if pad_q:
                query = F.pad(query.transpose(1, 3), (0, pad_q)).transpose(1, 3)
            if pad_kv:
                key = F.pad(key.transpose(1, 3), (0, pad_kv)).transpose(1, 3)
                value = F.pad(value.transpose(1, 3), (0, pad_kv)).transpose(1, 3)
            hidden_states = _attention(query, key, value, attn_mask=attention_mask, use_flex=True)[:, :seq_len_q]
        else:
            # Splited attention route on the first step or without kv cache
            valid = None if key_valid is None else key_valid[:, None, None, :]
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = valid if valid is None else valid[..., :end]
                if is_text:
                    seg_len = end - start
                    causal = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                        ],
                        dim=1,
                    )[None, None]
                    seg_mask = causal if seg_mask is None else seg_mask & causal
                outputs.append(_attention(query[:, start:end], key[:, :end], value[:, :end], attn_mask=seg_mask))
            outputs.append(_attention(query[:, prefix_len:], key, value, attn_mask=valid))
            hidden_states = torch.cat(outputs, dim=1)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class QwenImage21Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        self.inner_dim = heads * dim_head
        self.use_bias = False

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21RMSNorm(dim_head, eps=eps)
        self.norm_k = QwenImage21RMSNorm(dim_head, eps=eps)
        self.processor = QwenImage21AttnProcessor()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.processor(self, hidden_states, **kwargs)


class QwenImage21TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(hidden_size=dim, mlp_hidden_size=dim * mlp_ratio)

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: Any | None = None,
        target_token_mask: torch.Tensor | None = None,
        kv_cache: dict[str, torch.Tensor] | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            hidden_states=img_modulated,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)
        return hidden_states


class QwenImage21Rope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        self.freqs = [
            torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
            for dim in axes_dim
        ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]

        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        return torch.cat([self.freqs[0][frame_index], self.freqs[1][height_index], self.freqs[2][width_index]], dim=-1)


class QwenImage21DiT(nn.Module):
    _repeated_blocks = ["QwenImage21TransformerBlock"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImage21TransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _skip_keys = ["kv_cache"]

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_condition = causal_condition

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)

        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

    @staticmethod
    def build_token_metadata(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )

        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids

        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        kv_cache: list[dict[str, torch.Tensor]] | None = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        repeats = torch.where(img_mask, 4, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        if kv_cache is not None and not self.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`; otherwise the condition prefix depends on the "
                "changing target latent across denoising steps."
            )
        if kv_cache is not None and len(kv_cache) != len(self.transformer_blocks):
            raise ValueError(
                f"kv_cache must contain one dictionary per transformer block; got {len(kv_cache)} for "
                f"{len(self.transformer_blocks)} blocks."
            )

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        prefix_len = int((~target_token_mask).sum())
        is_decode = kv_cache is not None and len(kv_cache[0]) > 0
        cache_write_slice = None if is_decode or kv_cache is None else slice(0, prefix_len)
        segments = None
        if is_decode:
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
        elif FLEX_ATTN_AVAILABLE:
            attention_mask = build_qwenimage21_block_causal_mask(
                image_ids, joint_key_valid, batch_size, hidden_states.device
            )
        else:
            attention_mask = None
            segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

        for index_block, block in enumerate(self.transformer_blocks):
            block_kv_cache = kv_cache[index_block] if kv_cache is not None else None
            joint_hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states=joint_hidden_states,
                modulation=modulation,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
                target_token_mask=modulation_mask,
                kv_cache=block_kv_cache,
                cache_write_slice=cache_write_slice,
                segments=segments,
                key_valid=joint_key_valid,
            )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        return self.proj_out(joint_hidden_states)
