import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.gradient import gradient_checkpoint_forward
from ..core.quant import QuantBackend, register_quant_backend, register_quant_method

LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2
IMAGE_POSITION_OFFSET = 65536
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn


class Fp8Linear(torch.nn.Module):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale."""

    # Names of the packed tensors whose dtype must survive `to(dtype)` / `half()` / etc.
    dtype_guarded_tensor_names: tuple = ("weight", "weight_scale")

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def _apply(self, fn, recurse=True):
        # All dtype/device conversions (`to` / `half` / `float` / ...) funnel through
        # `_apply`; wrap `fn` so that a conversion which would change the dtype of a
        # tensor listed in `dtype_guarded_tensor_names` is redone as a device-only move,
        # keeping contract guarantee (b). The bias and everything else convert as usual.
        protected = {id(tensor) for name in self.dtype_guarded_tensor_names
                     if (tensor := getattr(self, name, None)) is not None}

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


# Backend for the released fp8 checkpoints, whose quantized Linears follow the
# compressed-tensors layout: `X.weight` in float8_e4m3 plus one float32 scale per output
# channel in `X.weight_scale`. `Fp8Linear` names its buffers accordingly, so loading such a
# checkpoint is a pure structure replacement.
@register_quant_backend("ideogram4_fp8")
class Ideogram4Fp8QuantBackend(QuantBackend):
    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True}

    def create_quantized_linear_shell(self, linear, compute_dtype):
        with torch.device("meta"):
            return Fp8Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=compute_dtype)

    def quantized_linear_classes(self) -> tuple:
        return (Fp8Linear,)

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = (module.weight.to(compute_dtype) * module.weight_scale.to(compute_dtype).unsqueeze(1))
        linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = nn.Parameter(module.bias.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        if model_device is not None:
            linear = linear.to(device=model_device)
        return linear


register_quant_method(
    "ideogram4_fp8",
    "ideogram4_fp8",
    lambda o: dict(o),
    label="8bit, fp8 (compressed-tensors layout, per-channel scale), weight-only",
)


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, seq_len, _ = position_ids.shape

        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
            3, batch_size, -1, 1
        ).to(pos.device)
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)

        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(
    t: torch.Tensor, dim: int, scale: float = 1e4
) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(
            getattr(self.mlp_in, "compute_dtype", None) or getattr(self.mlp_in, "computation_dtype", None) or self.mlp_in.weight.dtype
        )
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4DiT(nn.Module):
    """Ideogram 4 flow-matching transformer."""

    def __init__(self, config: Ideogram4Config | dict = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = Ideogram4Config()
        elif isinstance(config, dict):
            config = Ideogram4Config(**config)
        self.config = config
        self.patch_size = 2

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=config.emb_dim,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    norm_eps=config.norm_eps,
                    adanln_dim=config.adanln_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Velocity prediction.

        Args:
            llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features.
            x: (B, L, in_channels) noise tokens.
            t: (B,) or (B, L) flow-matching time in [0, 1].
            position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
            segment_ids: (B, L) sample id within a packed batch.
            indicator: (B, L) per-token role: LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR.

        Returns:
            (B, L, in_channels) velocity prediction in float32.
        """
        batch_size, seq_len, in_channels = x.shape
        assert in_channels == self.config.in_channels

        param_dtype = (
            getattr(self.input_proj, "compute_dtype", None) or getattr(self.input_proj, "computation_dtype", None) or self.input_proj.weight.dtype
        )
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
        )
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for layer in self.layers:
            h = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
            )

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
