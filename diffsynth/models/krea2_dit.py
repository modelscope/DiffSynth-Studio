import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from ..core.gradient import gradient_checkpoint_forward


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


def _mask(mask: Tensor) -> Tensor:
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20

    def get(self, *args, **kwargs):
        # For compatibility with low-version peft
        return None


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)

    def forward(
        self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )
        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))
        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(torch.nn.Module):
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=x.contiguous(),
                mask=None
            )
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs, mask
        )
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        config={"features": 6144, "tdim": 256, "txtdim": 2560, "heads": 48, "kvheads": 12, "multiplier": 4, "layers": 28, "patch": 2, "channels": 16, "txtheads": 20, "txtkvheads": 20, "txtlayers": 12},
        **kwargs
    ):
        super().__init__()
        if config is None:
            config = SingleMMDiTConfig(**kwargs)
        elif isinstance(config, dict):
            config = SingleMMDiTConfig(**config)
        self.config = config
        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"
        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
        context_pre_compute: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)
        if not context_pre_compute:
            txtmask = _mask(mask[:, : context.shape[1]])
            context = self.txtfusion(
                context, mask=txtmask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            context = self.txtmlp(context)
        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))
        mask = _mask(mask)
        freqs = self.posemb(pos)
        for block in self.blocks:
            combined = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=combined,
                vec=tvec,
                freqs=freqs,
                mask=mask,
            )
        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen, :]
        return output
