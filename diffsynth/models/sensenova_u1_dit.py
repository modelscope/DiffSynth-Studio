import torch, math
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
from .sensenova_u1_common import build_abs_positions_from_grid_hw


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(query, key, value, attention_mask, scaling, num_key_value_groups):
    """Attention with the softmax upcast to float32.

    The understanding branch keeps this instead of the shared `attention_forward`: its
    reference implementation upcasts the softmax to float32, whereas torch SDPA softmaxes
    in bfloat16. That difference is ~3e-3 relative per layer and compounds to ~2e-2 across
    the 42 layers, which would break output equivalence. This branch runs once per
    generation to build the conditioning cache, so the cost of staying exact is negligible.
    The image-generation branch does use `attention_forward` -- it is the per-step hot path
    and its reference implementation also softmaxes in bfloat16.
    """
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


class SenseNovaU1RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SenseNovaU1RotaryEmbedding(nn.Module):
    """Rotary embedding for one of the three positional axes (t / h / w).

    The frequency range is derived by computing the inverse frequencies for a doubled head
    dim and then taking every second entry, reproducing the reference implementation.

    `inv_freq` is a plain attribute built lazily in float32 rather than a registered buffer:
    the model is loaded as bfloat16, and `.to(dtype=...)` would otherwise degrade these
    frequencies (e.g. 0.6175287 -> 0.6171875), which shifts every rotary embedding and
    breaks output equivalence. Keeping it outside the buffer registry also keeps it out of
    the state dict, matching the reference where it is a non-persistent buffer.
    """

    def __init__(self, head_dim, rope_theta, max_position_embeddings, device=None):
        super().__init__()
        self.rope_head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.attention_scaling = 1.0
        self.inv_freq = None

    @staticmethod
    def _compute_inv_freq(base, dim, device=None):
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))

    def _ensure_inv_freq(self, device):
        if self.inv_freq is not None and self.inv_freq.device == device:
            return
        inv_freq_full = self._compute_inv_freq(self.rope_theta, self.rope_head_dim * 2, device)
        self.inv_freq = inv_freq_full[::2]

    @torch.no_grad()
    def forward(self, x, position_ids):
        self._ensure_inv_freq(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SenseNovaU1MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SenseNovaU1Attention(nn.Module):
    """Mixture-of-Transformers attention: a separate projection set per token type.

    The understanding branch uses `*_proj` / `*_norm`, the image-generation branch uses
    `*_proj_mot_gen` / `*_norm_mot_gen`. Each head dim is split into a time segment
    (head_dim // 2) and a spatial segment (two halves of head_dim // 4 for h and w),
    each rotated with its own RoPE frequency base.
    """

    def __init__(
        self,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        max_position_embeddings=262144,
        max_position_embeddings_hw=10000,
        layer_idx=0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.q_proj_mot_gen = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.k_proj_mot_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj_mot_gen = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.o_proj_mot_gen = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_hw = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.q_norm_hw_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)

        self.k_norm = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_hw = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)
        self.k_norm_hw_mot_gen = SenseNovaU1RMSNorm(head_dim // 2, eps=rms_norm_eps)

        self.rotary_emb = SenseNovaU1RotaryEmbedding(head_dim // 2, rope_theta, max_position_embeddings)
        self.rotary_emb_hw = SenseNovaU1RotaryEmbedding(head_dim // 4, rope_theta_hw, max_position_embeddings_hw)

    def _project_qkv(self, hidden_states, q_proj, k_proj, v_proj, q_norm, q_norm_hw, k_norm, k_norm_hw, indexes):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = q_proj(hidden_states).view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t = q_norm(query_states_t).transpose(1, 2)
        query_states_hw = q_norm_hw(query_states_hw).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = k_proj(hidden_states).view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t = k_norm(key_states_t).transpose(1, 2)
        key_states_hw = k_norm_hw(key_states_hw).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)
        return query_states, key_states, value_states, input_shape

    def _merge_cache(self, key_states, value_states, past_key_values, update_cache):
        if past_key_values is None:
            return key_states, value_states
        if update_cache:
            return past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
        # Read-only reuse of the prefix cache: the current tokens are never written back.
        layer = past_key_values.layers[self.layer_idx]
        past_k, past_v = layer.keys, layer.values
        if past_k is not None:
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)
        return key_states, value_states

    def forward_und(self, hidden_states, indexes, attention_mask, past_key_values=None, update_cache=True):
        query_states, key_states, value_states, input_shape = self._project_qkv(
            hidden_states, self.q_proj, self.k_proj, self.v_proj,
            self.q_norm, self.q_norm_hw, self.k_norm, self.k_norm_hw, indexes,
        )
        key_states, value_states = self._merge_cache(key_states, value_states, past_key_values, update_cache)

        attn_output = eager_attention_forward(
            query_states, key_states, value_states,
            attention_mask, self.scaling, self.num_key_value_groups,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)

    def forward_gen(self, hidden_states, indexes, attention_mask, past_key_values=None, update_cache=True):
        query_states, key_states, value_states, input_shape = self._project_qkv(
            hidden_states, self.q_proj_mot_gen, self.k_proj_mot_gen, self.v_proj_mot_gen,
            self.q_norm_mot_gen, self.q_norm_hw_mot_gen, self.k_norm_mot_gen, self.k_norm_hw_mot_gen, indexes,
        )
        key_states, value_states = self._merge_cache(key_states, value_states, past_key_values, update_cache)

        if attention_mask is None:
            # Image tokens attend bidirectionally over [prefix + current image block].
            # This is the per-step hot path; the reference also softmaxes in bfloat16 here.
            attn_output = attention_forward(
                query_states, key_states, value_states,
                q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b s n d",
                is_causal=False,
            )
        else:
            attn_output = eager_attention_forward(
                query_states, key_states, value_states,
                attention_mask, self.scaling, self.num_key_value_groups,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj_mot_gen(attn_output)

    def forward(
        self, hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
        indexes, attention_mask, past_key_values=None, update_cache=True,
    ):
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, update_cache)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, indexes, attention_mask, past_key_values, update_cache)
        raise NotImplementedError(
            "Mixed understanding/generation attention is not supported. Split the sequence at "
            "token-type boundaries and run the understanding and generation paths separately."
        )


class SenseNovaU1DecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        max_position_embeddings=262144,
        max_position_embeddings_hw=10000,
        layer_idx=0,
    ):
        super().__init__()
        self.self_attn = SenseNovaU1Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position_embeddings=max_position_embeddings,
            max_position_embeddings_hw=max_position_embeddings_hw,
            layer_idx=layer_idx,
        )
        self.mlp = SenseNovaU1MLP(hidden_size, intermediate_size)
        self.mlp_mot_gen = SenseNovaU1MLP(hidden_size, intermediate_size)
        self.input_layernorm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward_und(
        self, hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
        indexes, attention_mask, past_key_values=None, update_cache=True,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=update_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_gen(
        self, hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
        indexes, attention_mask, past_key_values=None, update_cache=True,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=update_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        return residual + hidden_states

    def forward(
        self, hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
        indexes, attention_mask, past_key_values=None, update_cache=True,
    ):
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
                indexes, attention_mask, past_key_values, update_cache,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states, exist_non_image_gen_tokens, exist_image_gen_tokens,
                indexes, attention_mask, past_key_values, update_cache,
            )
        raise NotImplementedError(
            "Mixed understanding/generation decoder layer is not supported. Split the sequence at "
            "token-type boundaries and run the understanding and generation paths separately."
        )


class SenseNovaU1Model(nn.Module):

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=42,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-06,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        max_position_embeddings=262144,
        max_position_embeddings_hw=10000,
        pad_token_id=151643,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.layers = nn.ModuleList([
            SenseNovaU1DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                rope_theta_hw=rope_theta_hw,
                max_position_embeddings=max_position_embeddings,
                max_position_embeddings_hw=max_position_embeddings_hw,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_hidden_layers)
        ])
        self.norm = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_mot_gen = SenseNovaU1RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        image_gen_indicators=None,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        update_cache=True,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Resolve once before the layer loop: keeping these as device tensors would force a
            # host readback in every layer and serialize the compute stream.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = gradient_checkpoint_forward(
                decoder_layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                update_cache=update_cache,
            )

        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm_mot_gen(hidden_states)
        return hidden_states, past_key_values


class SenseNovaU1ForCausalLM(nn.Module):

    def __init__(self, vocab_size=151936, hidden_size=4096, **kwargs):
        super().__init__()
        self.model = SenseNovaU1Model(vocab_size=vocab_size, hidden_size=hidden_size, **kwargs)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, *args, logits_to_keep=0, **kwargs):
        hidden_states, past_key_values = self.model(*args, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits, hidden_states, past_key_values


def precompute_rope_freqs_sincos(dim: int, max_position: int, base: float = 10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb_1d(x, cos_cached, sin_cached, positions):
    cos = cos_cached[positions]
    sin = sin_cached[positions]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(x, cos_cached_x, sin_cached_x, cos_cached_y, sin_cached_y, abs_positions_x, abs_positions_y):
    """The first half of the embedding dim carries the x axis, the second half carries the y axis."""
    dim_half = x.shape[-1] // 2
    rotated_part_1 = apply_rotary_emb_1d(x[..., :dim_half], cos_cached_x, sin_cached_x, abs_positions_x)
    rotated_part_2 = apply_rotary_emb_1d(x[..., dim_half:], cos_cached_y, sin_cached_y, abs_positions_y)
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class SenseNovaU1VisionEmbeddings(nn.Module):

    def __init__(
        self,
        hidden_size=1024,
        llm_hidden_size=4096,
        patch_size=16,
        num_channels=3,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.llm_embed_dim = llm_hidden_size
        self.downsample_factor = int(1 / downsample_ratio)
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor, stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()

        self.rope_dim_part = self.embed_dim // 2
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.rope_theta_vision = rope_theta_vision

        # Built lazily on the first real device: the model is constructed on the meta device,
        # and these deterministic caches are absent from the checkpoint.
        self.register_buffer("cos_cached_x", None, persistent=False)
        self.register_buffer("sin_cached_x", None, persistent=False)
        self.register_buffer("cos_cached_y", None, persistent=False)
        self.register_buffer("sin_cached_y", None, persistent=False)

    def _ensure_rope_cache(self, device: torch.device) -> None:
        if self.cos_cached_x is not None and self.cos_cached_x.device == device:
            return
        cos, sin = precompute_rope_freqs_sincos(
            self.rope_dim_part, self.max_position_embeddings_vision,
            base=self.rope_theta_vision, device=device,
        )
        self.cos_cached_x = cos
        self.sin_cached_x = sin
        self.cos_cached_y = cos.clone()
        self.sin_cached_y = sin.clone()

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),  # RoPE is more stable in float32
            self.cos_cached_x, self.sin_cached_x,
            self.cos_cached_y, self.sin_cached_y,
            abs_pos_x, abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(self, pixel_values: torch.Tensor, grid_hw=None) -> torch.Tensor:
        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)
        self._ensure_rope_cache(patch_embeds.device)
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        # Each image has its own grid, so the 2x2 downsampling convolution runs per image.
        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position: cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)

        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor ** 2)

        return embeddings


class SenseNovaU1VisionEncoder(nn.Module):
    """Patch embedder for the understanding branch: pixels to LLM-dimension tokens."""

    def __init__(
        self,
        hidden_size=1024,
        llm_hidden_size=4096,
        patch_size=16,
        num_channels=3,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
    ):
        super().__init__()
        self.embeddings = SenseNovaU1VisionEmbeddings(
            hidden_size=hidden_size,
            llm_hidden_size=llm_hidden_size,
            patch_size=patch_size,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            rope_theta_vision=rope_theta_vision,
            max_position_embeddings_vision=max_position_embeddings_vision,
        )

    def forward(self, pixel_values=None, grid_hw=None, pixel_embeds=None):
        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')
        if pixel_embeds is not None:
            return pixel_embeds
        assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
        return self.embeddings(pixel_values, grid_hw=grid_hw)


class SenseNovaU1TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps (or noise scales) into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class SenseNovaU1ConvDecoder(nn.Module):
    """Pixel head: three PixelShuffle stages take H/32 hidden states back to full resolution."""

    def __init__(self, input_dim=4096, hidden_dim=1024):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)

        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1(x)))
        x = self.ps3(self.conv2(self.ps2(x)))
        return x


class SenseNovaU1DiT(nn.Module):
    """Unified Mixture-of-Transformers denoiser for SenseNova-U1.

    The same 42-layer backbone serves two roles: the understanding branch encodes the
    conditioning prefix into a KV cache, and the generation branch denoises image tokens
    against that cache. Flow matching happens directly in pixel space, so there is no VAE.
    """

    _repeated_blocks = ["SenseNovaU1DecoderLayer"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=42,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-06,
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=5000000.0,
        rope_theta_hw=10000.0,
        max_position_embeddings=262144,
        max_position_embeddings_hw=10000,
        pad_token_id=151643,
        vision_hidden_size=1024,
        patch_size=16,
        num_channels=3,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
        downsample_ratio=0.5,
        use_pixel_head=True,
        fm_head_dim=1536,
        fm_head_layers=2,
        fm_head_mlp_ratio=1,
        add_noise_scale_embedding=True,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.merge_size = int(1 / downsample_ratio)
        self.hidden_size = hidden_size
        self.use_pixel_head = use_pixel_head
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.noise_scale_max_value = noise_scale_max_value

        self.language_model = SenseNovaU1ForCausalLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_theta_hw=rope_theta_hw,
            max_position_embeddings=max_position_embeddings,
            max_position_embeddings_hw=max_position_embeddings_hw,
            pad_token_id=pad_token_id,
        )

        # Understanding-branch patch embedder. Separate weights from the generation-branch copy in
        # `fm_modules`: this one encodes the images the user supplies, that one encodes the noisy
        # image at every denoising step. The attribute name mirrors the checkpoint key prefix.
        self.vision_model = SenseNovaU1VisionEncoder(
            hidden_size=vision_hidden_size,
            llm_hidden_size=hidden_size,
            patch_size=patch_size,
            num_channels=num_channels,
            downsample_ratio=downsample_ratio,
            rope_theta_vision=rope_theta_vision,
            max_position_embeddings_vision=max_position_embeddings_vision,
        )

        fm_modules = {
            "vision_model_mot_gen": SenseNovaU1VisionEncoder(
                hidden_size=vision_hidden_size,
                llm_hidden_size=hidden_size,
                patch_size=patch_size,
                num_channels=num_channels,
                downsample_ratio=downsample_ratio,
                rope_theta_vision=rope_theta_vision,
                max_position_embeddings_vision=max_position_embeddings_vision,
            ),
            "timestep_embedder": SenseNovaU1TimestepEmbedder(hidden_size),
        }
        if add_noise_scale_embedding:
            fm_modules["noise_scale_embedder"] = SenseNovaU1TimestepEmbedder(hidden_size)
        fm_modules["fm_head"] = SenseNovaU1ConvDecoder(hidden_size)
        self.fm_modules = nn.ModuleDict(fm_modules)

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def embed_tokens(self, input_ids):
        return self.language_model.model.embed_tokens(input_ids)

    def compute_noise_scale(self, height, width):
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        noise_scale = self.noise_scale
        if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(self.noise_scale_base_image_seq_len)
            scale = math.sqrt((grid_h * grid_w) / (self.merge_size ** 2) / base)
            noise_scale = scale * float(self.noise_scale)
            if self.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        return min(noise_scale, self.noise_scale_max_value)

    def extract_und_feature(self, pixel_values, grid_hw):
        return self.vision_model(pixel_values=pixel_values, grid_hw=grid_hw)

    def extract_gen_feature(self, pixel_values, grid_hw):
        return self.fm_modules["vision_model_mot_gen"](pixel_values=pixel_values, grid_hw=grid_hw)

    def embed_timestep(self, timestep, noise_scale, num_tokens, batch_size=1):
        t_expanded = timestep.expand(batch_size * num_tokens)
        embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(batch_size, num_tokens, -1)
        if self.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(t_expanded, noise_scale / self.noise_scale_max_value)
            noise_embeddings = self.fm_modules["noise_scale_embedder"](noise_scale_tensor)
            embeddings = embeddings + noise_embeddings.view(batch_size, num_tokens, -1)
        return embeddings

    def encode_prefix(
        self,
        input_ids=None,
        inputs_embeds=None,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        """Run the understanding branch over the conditioning prefix and return its KV cache."""
        hidden_states, past_key_values = self.language_model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            update_cache=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return past_key_values, hidden_states

    def _append_tokens_to_cache(self, past_key_values, t_idx, input_ids):
        seq_len = input_ids.shape[1]
        device = input_ids.device
        t_indexes = torch.arange(t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device)
        zeros = torch.zeros(seq_len, dtype=torch.long, device=device)
        indexes = torch.stack([t_indexes, zeros, zeros], dim=0)

        # The appended run attends fully to the existing cache and causally within itself.
        past_len = past_key_values.get_seq_length()
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        attention_mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
        attention_mask[:, :, :, past_len:] = torch.where(
            causal == 1, torch.tensor(0.0, device=device), torch.tensor(float("-inf"), device=device)
        )

        self.language_model.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return t_idx + seq_len

    def _decode_one_token(self, token, t_idx, past_key_values):
        indexes = torch.tensor([[t_idx + 1], [0], [0]], dtype=torch.long, device=token.device)
        return self.language_model(
            input_ids=token.view(1, 1),
            indexes=indexes,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

    @torch.no_grad()
    def generate_think(
        self,
        input_ids=None,
        inputs_embeds=None,
        indexes=None,
        attention_mask=None,
        eos_token_id=None,
        think_end_token_id=None,
        append_ids=None,
        max_think_tokens=1024,
    ):
        """Encode the prefix, then greedily decode a reasoning block into the same KV cache.

        Decoding is greedy with no temperature or nucleus sampling, matching the reference. The
        returned cache already contains the reasoning block plus `append_ids`, so the image tokens
        that follow must start at `t_idx + 1` rather than at the original prefix length.
        """
        logits, _, past_key_values = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        t_idx = int(indexes[0].max().item())
        token_ids = []
        next_token = logits[:, -1, :].argmax(dim=-1)

        for _ in range(max_think_tokens):
            token_id = int(next_token.item())
            if token_id == eos_token_id:
                break
            token_ids.append(token_id)
            logits, _, past_key_values = self._decode_one_token(next_token, t_idx, past_key_values)
            t_idx += 1
            if token_id == think_end_token_id:
                break
            next_token = logits[:, -1, :].argmax(dim=-1)

        if append_ids is not None and append_ids.shape[1] > 0:
            t_idx = self._append_tokens_to_cache(past_key_values, t_idx, append_ids)
        return past_key_values, t_idx, token_ids

    def forward(
        self,
        image_embeds,
        indexes_image,
        past_key_values,
        image_size,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        batch_size, num_tokens = image_embeds.shape[0], image_embeds.shape[1]
        image_gen_indicators = torch.ones(
            (batch_size, num_tokens), dtype=torch.bool, device=image_embeds.device
        )

        hidden_states, _ = self.language_model.model(
            inputs_embeds=image_embeds,
            image_gen_indicators=image_gen_indicators,
            indexes=indexes_image,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            update_cache=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        merged_patch = self.patch_size * self.merge_size
        token_h = image_size[1] // merged_patch
        token_w = image_size[0] // merged_patch

        img_reshaped = hidden_states[:, -num_tokens:].view(batch_size, token_h, token_w, -1)
        img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
        img_2d = img_2d.contiguous().view(batch_size, -1, token_h, token_w)

        smoothed_img_2d = self.fm_modules["fm_head"](img_2d)

        smoothed = smoothed_img_2d.view(batch_size, 3, token_h, merged_patch, token_w, merged_patch)
        smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
        return smoothed.contiguous().view(batch_size, num_tokens, merged_patch * merged_patch * 3)
