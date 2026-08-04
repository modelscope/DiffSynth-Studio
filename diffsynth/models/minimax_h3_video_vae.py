# SPDX-License-Identifier: Apache-2.0
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Normalize

from ..core.attention import attention_forward


class WarpedTensor(torch.nn.Module):
    def __init__(self, weight=None, shape=None):
        super().__init__()
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.empty(shape))

    def forward(self):
        return self.weight


def create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = 2.0 * (torch.arange(0.5, dim_size, dtype=dtype, device=device) / dim_size) - 1.0
        coords_list.append(coords)
    return torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1).flatten(0, len(patch_dims) - 1).unsqueeze(0)


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, rotary_pos_emb):
    cos, sin = rotary_pos_emb
    cos, sin = cos.to(t.dtype), sin.to(t.dtype)
    rot_dim = cos.shape[-1]
    if rot_dim < t.shape[-1]:
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        return torch.cat(((t_rot * cos) + (_rotate_half(t_rot) * sin), t_pass), dim=-1)
    return (t * cos) + (_rotate_half(t) * sin)


class BaseConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 padding_mode="zeros", padding_mode_t=None, causal=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=bias, padding_mode=padding_mode)
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.pad_mode_t = ("constant" if padding_mode_t == "zeros" else padding_mode_t) or ("constant" if causal else "replicate")
        self.causal = causal

    def _temporal_pad(self, x):
        if x.shape[2] > 1:
            p = self.padding[0]
            return F.pad(x, (0, 0, 0, 0, p * 2 if self.causal else p, 0 if self.causal else p), mode=self.pad_mode_t)
        if self.pad_mode_t == "constant":
            return torch.cat([torch.zeros_like(x[:, :, :1]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1), x], dim=2)
        return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)
        if self.padding[2] or self.padding[1]:
            x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0), mode=self.pad_mode)
        x = self._temporal_pad(x)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


def _merge_time_to_batch(x):
    B, C, D, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, 1, H, W)


def _split_time_from_batch(x, batch):
    BD, C, _, H, W = x.shape
    return x.view(batch, BD // batch, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


class TemporalIsolatedSpatialGroupNorm(nn.GroupNorm):
    def forward(self, input):
        if input.dim() == 5:
            batch = input.shape[0]
            return _split_time_from_batch(super().forward(_merge_time_to_batch(input)), batch)
        return super().forward(input)


def _group_norm(ch, t_isolated=False):
    if t_isolated:
        return TemporalIsolatedSpatialGroupNorm(32, ch, eps=1e-6, affine=True)
    return nn.GroupNorm(32, ch, eps=1e-6, affine=True)


class Attention(nn.Module):
    def __init__(self, heads, dim_head, embed_dim=None, qk_norm_type=None, qk_norm_affine=False, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        dim = embed_dim or inner_dim
        norm_cls = {"layer_norm": nn.LayerNorm, "rms_norm": nn.RMSNorm}.get(qk_norm_type)
        self.norm_q = norm_cls(dim_head, eps=eps, elementwise_affine=qk_norm_affine) if norm_cls else None
        self.norm_k = norm_cls(dim_head, eps=eps, elementwise_affine=qk_norm_affine) if norm_cls else None
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x, rotary_pos_emb=None):
        B, S, _ = x.shape
        q, k, v = torch.chunk(self.to_qkv(x).view(B, S, -1, 3 * self.dim_head), 3, dim=-1)
        if self.norm_q is not None: q = self.norm_q(q)
        if self.norm_k is not None: k = self.norm_k(k)
        if rotary_pos_emb is not None:
            q, k = apply_rotary_pos_emb(q, rotary_pos_emb), apply_rotary_pos_emb(k, rotary_pos_emb)
        return self.to_out(attention_forward(q, k, v, q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d").reshape(B, S, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 8, bias=bias)
        self.w2 = nn.Linear(dim * 4, dim, bias=bias)

    def forward(self, x):
        x = self.w1(x)
        gate, x = x.chunk(2, dim=-1)
        return self.w2(F.silu(gate) * x)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=10000, n_dim=3, use_angle=False):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi if use_angle else 1.0
        self.rotary_base = rotary_base
        self.dim = dim

    def forward(self, img_ids):
        with torch.autocast("cuda", enabled=False):
            inv_freq = 1 / self.rotary_base ** torch.arange(0, 1, 2 * self.n_dim / self.dim, dtype=torch.float32, device=img_ids.device)
            angles = (self.angle_scale * img_ids[:, :, :, None] * inv_freq[None, None, None, :]).flatten(2, 3).tile(2).unsqueeze(2)
            return torch.cos(angles), torch.sin(angles)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, embed_dim=None, norm_affine=True,
                 qk_norm_type="rms_norm", qk_norm_affine=False, bias=True, eps=1e-5):
        super().__init__()
        dim = embed_dim or dim_head * heads
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=norm_affine, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, embed_dim=dim, qk_norm_type=qk_norm_type,
                              qk_norm_affine=qk_norm_affine, bias=bias, eps=eps)
        self.scale1 = WarpedTensor(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=norm_affine, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = WarpedTensor(torch.zeros(dim))

    def forward(self, x, rotary_pos_emb=None):
        h = self.norm1(x)
        h = x + self.attn(h, rotary_pos_emb) * self.scale1()
        h2 = self.norm2(h)
        return h + self.ff(h2) * self.scale2()


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2, **conv_kwargs):
        super().__init__()
        self.conv = BaseConv3d(in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
                               stride=(time_stride, space_stride, space_stride), **conv_kwargs)
        self.space_stride = space_stride
        self.pad_mode = self.conv.pad_mode

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, padding_mode="zeros", padding_mode_t=None, causal=True, use_t_isolated_gn=False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = _group_norm(in_channels, use_t_isolated_gn)
        self.norm2 = _group_norm(out_channels, use_t_isolated_gn)
        self.conv1 = BaseConv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, padding_mode_t=padding_mode_t, causal=causal)
        self.conv2 = BaseConv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode, padding_mode_t=padding_mode_t, causal=causal)
        if in_channels != out_channels:
            self.nin_shortcut = BaseConv3d(in_channels, out_channels, kernel_size=1, padding_mode=padding_mode, padding_mode_t=padding_mode_t, causal=causal)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if hasattr(self, 'nin_shortcut'):
            x = self.nin_shortcut(x)
        return x + h


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels,
                 double_z=False, padding_mode="zeros", padding_mode_t=None, causal=True, use_t_isolated_gn=False):
        super().__init__()
        num_levels = len(ch_mult)
        num_res_blocks = [num_res_blocks] * num_levels if isinstance(num_res_blocks, int) else num_res_blocks
        block_mid = [ch * ch_mult[i] for i in range(num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        conv_kwargs = dict(padding_mode=padding_mode, padding_mode_t=padding_mode_t, causal=causal)

        self.conv_in = BaseConv3d(in_channels, block_in[0], kernel_size=3, padding=1, **conv_kwargs)
        self.down = nn.ModuleList()
        for i in range(num_levels):
            down = nn.Module()
            down.block = nn.ModuleList([
                ResnetBlock3D(in_channels=block_in[i] if j == 0 else block_mid[i], out_channels=block_mid[i],
                              use_t_isolated_gn=use_t_isolated_gn, **conv_kwargs)
                for j in range(num_res_blocks[i])
            ])
            if space_down[i] * time_down[i] > 1:
                down.downsample = Downsample3D(block_mid[i], block_mid[i], time_stride=time_down[i], space_stride=space_down[i], **conv_kwargs)
            self.down.append(down)
        self.norm_out = _group_norm(block_mid[-1], use_t_isolated_gn)
        self.conv_out = BaseConv3d(block_mid[-1], 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1, **conv_kwargs)

    def forward(self, x):
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def _pack_tensors_3d(x, ps, ps_t):
    B, C, T, H, W = x.shape
    return x.view(B, C, T // ps_t, ps_t, H // ps, ps, W // ps, ps).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        B, (T // ps_t) * (H // ps) * (W // ps), C * ps_t * ps * ps)


def _unpack_tensors_3d(x, ps, ps_t, T, H, W):
    B, N, C = x.shape
    c = C // (ps_t * ps * ps)
    return x.view(B, T // ps_t, H // ps, W // ps, c, ps_t, ps, ps).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(B, c, T, H, W)


class ViT3DDecoder(nn.Module):
    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36,
                 heads=32, dim_head=64, norm_affine=True, qk_norm_type="rms_norm",
                 qk_norm_affine=False, rope_theta=100.0,
                 rope_dim_ratio=0.75, bias=True, eps=1e-5, num_register_tokens=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        dim = heads * dim_head
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3, use_angle=True)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.num_register_tokens = num_register_tokens
        self.register_tokens = WarpedTensor(torch.randn(1, num_register_tokens, dim) * 0.02) if num_register_tokens > 0 else None
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(heads=heads, dim_head=dim_head, norm_affine=norm_affine,
                             qk_norm_type=qk_norm_type, qk_norm_affine=qk_norm_affine,
                             bias=bias, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=norm_affine, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init)

    def forward(self, x):
        B, _, T, H, W = x.shape
        num_suffix = 1 + self.num_register_tokens
        with torch.autocast("cuda", enabled=False):
            hidden_states = self.x_embedder(_pack_tensors_3d(x, 1, 1))
        num_patches = hidden_states.shape[1]
        tokens = [hidden_states]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens().expand(B, -1, -1))
        tokens.append(torch.zeros_like(hidden_states[:, 0:1, :]))
        hidden_states = torch.cat(tokens, dim=1)
        img_ids = torch.cat([
            create_token_ids((T, H, W), x.device, torch.float32).expand(B, -1, -1),
            torch.zeros(B, num_suffix, 3, device=x.device, dtype=torch.float32)
        ], dim=1)
        rotary_pos_emb = self.pos_embed(img_ids)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_pos_emb)
        with torch.autocast("cuda", enabled=False):
            output = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]
        return _unpack_tensors_3d(output, self.patch_size, self.patch_size_t, T * self.patch_size_t, H * self.patch_size, W * self.patch_size)


class VAEProcessor:
    def __init__(self, transform, transform_rev):
        self.transform = transform
        self.transform_rev = transform_rev

    def transform_tensor(self, x):
        if x.ndim == 5:
            if x.shape[2] == 3: x = x.transpose(1, 2)
            B, _, T, _, _ = x.shape
            x = rearrange(self.transform(rearrange(x, "b c t h w -> (b t) c h w")), "(b t) c h w -> b c t h w", b=B, t=T)
        else:
            if x.ndim == 4 and x.shape[1] == 3: x = x.transpose(0, 1) if x.shape[0] == 3 else x
            x = self.transform(x)
        return x.contiguous()

    def revert_tensor(self, x):
        if x.ndim == 5:
            B, _, T, _, _ = x.shape
            x = rearrange(self.transform_rev(rearrange(x, "b c t h w -> (b t) c h w")).clamp(0, 1),
                          "(b t) c h w -> b c t h w", b=B, t=T)
        else:
            x = self.transform_rev(x).clamp(0, 1)
        return x.contiguous()


_VIDEO_LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
_VIDEO_LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(self, in_channels=3, out_ch=3, ch=128, embed_dim=24, z_channels=24, num_res_blocks=2,
                 ch_mult=(1, 2, 2, 4, 4, 8), space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1),
                 padding_mode="reflect", use_t_isolated_gn=True, causal_encoder=True,
                 clip_length=17, token_drop=3, tile_size=256, tile_overlap_min=64, **kwargs):
        super().__init__()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        vae_ratio = 1
        for s in space_down: vae_ratio *= s
        self.vae_ratio = vae_ratio
        vae_ratio_t = 1
        for t in time_down: vae_ratio_t *= t
        self.vae_ratio_t = vae_ratio_t
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min
        self.processor = VAEProcessor(Normalize(mean, std), Normalize(tuple(-m / s for m, s in zip(mean, std)), tuple(1.0 / s for s in std)))
        self.encoder = EncoderFCN3D(double_z=True, z_channels=z_channels, in_channels=in_channels, ch=ch,
                                    num_res_blocks=num_res_blocks, ch_mult=ch_mult, space_down=space_down,
                                    time_down=time_down, padding_mode=padding_mode, padding_mode_t=None,
                                    causal=causal_encoder, use_t_isolated_gn=use_t_isolated_gn)
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio, patch_size_t=self.vae_ratio_t, in_channels=z_channels, out_channels=out_ch,
            dim_head=64, heads=32, norm_affine=True, num_layers=36,
            qk_norm_affine=False, qk_norm_type="rms_norm", rope_dim_ratio=0.75, rope_theta=100.0)

    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def split_tiles(self, input_len):
        ts, ov_min = self.tile_size, self.tile_overlap_min
        if ts >= input_len:
            return [0], [input_len], []
        N = math.ceil(input_len / ts)
        while True:
            overlaps = [ov_min] * (N - 1)
            remaining = ts * N - sum(overlaps) - input_len
            if remaining < 0: N += 1
            else: break
        for i in range(remaining // self.vae_ratio):
            overlaps[i % (N - 1)] += self.vae_ratio
        starts = [0]
        for i in range(N - 1):
            starts.append(starts[-1] + ts - overlaps[i])
        return starts, [ts] * N, overlaps

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        w = torch.arange(blend_extent, device=b.device, dtype=b.dtype) / blend_extent
        shape = [1] * a.ndim; shape[dim] = blend_extent
        w = w.view(shape)
        blended = a.narrow(dim, -blend_extent, blend_extent) * (1 - w) + b.narrow(dim, 0, blend_extent) * w
        if blend_extent < b.shape[dim]:
            return torch.cat([blended, b.narrow(dim, blend_extent, b.shape[dim] - blend_extent)], dim=dim)
        return blended

    def tiled_encode(self, x):
        H, W = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_ovlp = self.split_tiles(H)
        x_idx, x_len, x_ovlp = self.split_tiles(W)
        i_max, j_max = len(y_idx), len(x_idx)
        tiles = [x[..., i:i + il, j:j + jl] for i, il in zip(y_idx, y_len) for j, jl in zip(x_idx, x_len)]
        encoded = [self.encode(t) for t in tiles]
        rows = [[encoded[i * j_max + j].to(x.device) for j in range(j_max)] for i in range(i_max)]
        ly = [o // self.vae_ratio for o in y_ovlp]
        lx = [o // self.vae_ratio for o in x_ovlp]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0: tile = self.blend(rows[i - 1][j], tile, ly[i - 1], dim=-2)
                if j > 0: tile = self.blend(row[j - 1], tile, lx[j - 1], dim=-1)
                if i < i_max - 1: tile = tile[..., :-ly[i], :]
                if j < j_max - 1: tile = tile[..., :, :-lx[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def tiled_decode(self, z):
        H, W = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_ovlp = self.split_tiles(H)
        x_idx, x_len, x_ovlp = self.split_tiles(W)
        i_max, j_max = len(y_idx), len(x_idx)
        tiles = [z[..., y_idx[i] // self.vae_ratio:(y_idx[i] + y_len[i]) // self.vae_ratio,
                     x_idx[j] // self.vae_ratio:(x_idx[j] + x_len[j]) // self.vae_ratio]
                 for i in range(i_max) for j in range(j_max)]
        decoded = [self.decode(t) for t in tiles]
        rows = [[decoded[i * j_max + j].to(z.device) for j in range(j_max)] for i in range(i_max)]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0: tile = self.blend(rows[i - 1][j], tile, y_ovlp[i - 1], dim=-2)
                if j > 0: tile = self.blend(row[j - 1], tile, x_ovlp[j - 1], dim=-1)
                if i < i_max - 1: tile = tile[..., :-y_ovlp[i], :]
                if j < j_max - 1: tile = tile[..., :, :-x_ovlp[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def encode_temporal(self, x):
        if x.shape[2] % self.clip_length != 0:
            x = torch.cat([x, x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % self.clip_length, 1, 1)], dim=2)
        z = torch.cat([self.tiled_encode(x[:, :, i * self.clip_length:(i + 1) * self.clip_length]) for i in range(x.shape[2] // self.clip_length)], dim=2)
        return z[:, :, :-self.token_drop] if self.token_drop > 0 else z

    def _decode_temporal_pad_frames(self, z, pad_tokens):
        if pad_tokens <= 0: return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0: return pad_tokens * self.vae_ratio_t
        z_len = z.shape[2] - pad_tokens
        return sum(intra_tail if (z_len + k) % self.tokens_chunk_size == 0 else self.vae_ratio_t for k in range(pad_tokens))

    def _decode_temporal_output_frame_plan(self, z, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = final_overlap = 0
        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            t_end = t_start + self.tokens_chunk_size + self.token_overlap
            clip_frame_len = max(0, min(t_end, z.shape[2]) - min(t_start, z.shape[2])) * self.vae_ratio_t
            for j in range(split_count):
                f_start = j * chunk_dec
                chunk_frames = max(0, min(f_start + chunk_dec, clip_frame_len) - f_start - self.frame_pre_padding)
                if j == 0: total_frames += chunk_frames
                else: final_overlap = chunk_frames
        total_frames += final_overlap
        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        return int(total_frames), int(pad_frames), int(total_frames - pad_frames)

    def _decode_temporal_streaming(self, z, num_chunks, pad_tokens):
        total_frames, pad_frames, output_frames = self._decode_temporal_output_frame_plan(z, num_chunks, pad_tokens)
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = dec_overlap = None
        write_pos = logical_frames = dropped_frames = 0

        def write_part(part):
            nonlocal dec, write_pos, logical_frames, dropped_frames
            pf = int(part.shape[2])
            if pf <= 0: return
            logical_frames += pf
            if dec is None:
                dec = torch.empty(*part.shape[:2], output_frames, *part.shape[3:], dtype=part.dtype, device=part.device)
            copy = min(pf, max(0, int(dec.shape[2]) - write_pos))
            if copy > 0:
                dec[:, :, write_pos:write_pos + copy].copy_(part[:, :, :copy])
                write_pos += copy
            dropped_frames += pf - copy

        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            clip_dec = self.tiled_decode(z[:, :, t_start:t_start + self.tokens_chunk_size + self.token_overlap])
            if clip_dec.device != z.device: clip_dec = clip_dec.to(z.device)
            for j in range(split_count):
                f_start = j * chunk_dec
                chunk = clip_dec[:, :, f_start:min(f_start + chunk_dec, clip_dec.shape[2])][:, :, self.frame_pre_padding:]
                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    write_part(chunk)
                else:
                    dec_overlap = chunk.contiguous()
            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
            del clip_dec

        if logical_frames != total_frames or dropped_frames != pad_frames or write_pos != output_frames:
            raise RuntimeError(f"decode_temporal frame plan mismatch: logical={logical_frames} total={total_frames} dropped={dropped_frames} pad={pad_frames} pos={write_pos} out={output_frames}")
        return dec

    def decode_temporal(self, z):
        pseudo_total = z.shape[2] + self.token_drop
        pad_tokens = (self.tokens_chunk_size - pseudo_total % self.tokens_chunk_size) % self.tokens_chunk_size
        pseudo_total += pad_tokens
        num_chunks = pseudo_total // self.tokens_chunk_size - int(self.token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        return self._decode_temporal_streaming(z, num_chunks, pad_tokens)

    @torch.no_grad()
    def encode_video(self, video, dtype=None, process_image=False, tiled=True, tile_size=256, tile_overlap=64):
        device, out_dtype = video.device, video.dtype
        x = self.processor.transform_tensor(video.to(torch.float32)).to(device=device, dtype=dtype or out_dtype)
        self.tile_size, self.tile_overlap_min = tile_size, tile_overlap
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if process_image:
            moments = self.tiled_encode(x)
        else:
            moments = self.encode_temporal(x)
        z = torch.chunk(moments, 2, dim=1)[0]
        if process_image:
            z = z[:, :, :1]
        mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, device=z.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return ((z.float() - mean) / std).to(out_dtype)

    @torch.no_grad()
    def decode_video(self, latents, dtype=None, process_image=False, tiled=True, tile_size=256, tile_overlap=64):
        device, out_dtype = latents.device, latents.dtype
        mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(_VIDEO_LATENTS_STD, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        z = (latents.to(device, torch.float32) * std + mean).to(dtype or out_dtype)
        self.tile_size, self.tile_overlap_min = tile_size, tile_overlap
        if process_image:
            recon = self.tiled_decode(z)
            recon = recon.squeeze(2)
        else:
            recon = self.decode_temporal(z)
        return self.processor.revert_tensor(recon.float()).to(out_dtype)

__all__ = ["MiniMaxH3VideoVAE"]
