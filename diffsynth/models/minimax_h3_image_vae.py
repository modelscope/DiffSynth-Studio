# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE: the MiniMax-H3 video VAE encoder run over a single frame, plus an
image-only 2D ViT decoder (one token per latent pixel, register tokens, 2D RoPE). Pixel space is
ImageNet-normalised RGB over a [0, 1] base, i.e. (pixel - mean) / std with mean (0.485, 0.456, 0.406)
and std (0.229, 0.224, 0.225). The latent normalisation (latents_mean / latents_std) is applied by the
pipeline, never here. Tiling is on by default: the released images are the linearly blended tile
outputs, so tiling=False changes them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .minimax_h3_video_vae import (
    BaseConv3d,
    Downsample3D,
    FeedForward,
    ResnetBlock3D,
    RotaryEmbeddingND,
    TemporalIsolatedSpatialGroupNorm,
    TransformerBlock,
    VAEProcessor,
    WarpedTensor,
    create_token_ids,
)
from .stable_diffusion_vae import DiagonalGaussianDistribution


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-channel latent statistics of the released checkpoint (`latents_mean` / `latents_std` in its
# config.json). They are stored for convenience only; `encode` / `decode` do not apply them.
_IMAGE_LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
_IMAGE_LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


class MiniMaxH3ImageResnetBlock3d(ResnetBlock3D):
    # Same block as the H3 video VAE encoder (its `forward` and `BaseConv3d` padding are reused), with
    # configurable GroupNorm groups/eps because the image VAE config exposes `norm_num_groups` /
    # `norm_eps`. The parent `__init__` hard-codes 32 groups, so the parameters are built here.
    def __init__(self, in_channels, out_channels=None, norm_num_groups=32, norm_eps=1e-6,
                 spatial_padding_mode="reflect"):
        nn.Module.__init__(self)
        out_channels = out_channels or in_channels
        conv_kwargs = dict(padding_mode=spatial_padding_mode, padding_mode_t=None, causal=True)
        self.norm1 = TemporalIsolatedSpatialGroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = BaseConv3d(in_channels, out_channels, kernel_size=3, padding=1, **conv_kwargs)
        self.norm2 = TemporalIsolatedSpatialGroupNorm(norm_num_groups, out_channels, eps=norm_eps)
        self.conv2 = BaseConv3d(out_channels, out_channels, kernel_size=3, padding=1, **conv_kwargs)
        if in_channels != out_channels:
            self.nin_shortcut = BaseConv3d(in_channels, out_channels, kernel_size=1, **conv_kwargs)


class MiniMaxH3ImageEncoder3d(nn.Module):
    # Causal 3D CNN encoder. Its key layout (`encoder.down.{L}.block.{j}.*`, and
    # `encoder.down.{L}.downsample.conv.*` for the levels that downsample) is the H3 video VAE one.
    def __init__(self, in_channels=3, out_channels=48, block_out_channels=(128, 256, 256, 512, 512, 1024),
                 layers_per_block=2, spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
                 temporal_downsample_factors=(1, 2, 2, 1, 1, 1), norm_num_groups=32, norm_eps=1e-6,
                 spatial_padding_mode="reflect"):
        super().__init__()
        conv_kwargs = dict(padding_mode=spatial_padding_mode, padding_mode_t=None, causal=True)
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])

        self.conv_in = BaseConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1, **conv_kwargs)
        self.down = nn.ModuleList()
        for i in range(len(block_out_channels)):
            level = nn.Module()
            level.block = nn.ModuleList([
                MiniMaxH3ImageResnetBlock3d(
                    in_channels=block_in_channels[i] if j == 0 else block_out_channels[i],
                    out_channels=block_out_channels[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for j in range(layers_per_block)
            ])
            if spatial_downsample_factors[i] * temporal_downsample_factors[i] > 1:
                level.downsample = Downsample3D(
                    block_out_channels[i], block_out_channels[i],
                    time_stride=temporal_downsample_factors[i],
                    space_stride=spatial_downsample_factors[i],
                    **conv_kwargs,
                )
            self.down.append(level)

        self.norm_out = TemporalIsolatedSpatialGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps)
        self.conv_out = BaseConv3d(block_out_channels[-1], out_channels, kernel_size=3, padding=1, **conv_kwargs)

    def forward(self, x):
        # x: (B, C, T, H, W) with T == 1 for images, (B, 2 * latent_channels, T, H / 16, W / 16) out.
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class MiniMaxH3ImageFeedForward(FeedForward):
    # Gated SwiGLU feed-forward of the H3 ViT decoders: `FeedForward.forward` is reused, so `w1` rows
    # are `[gate; value]` as in the checkpoint; only the expansion factor (`decoder_ffn_mult`) differs.
    def __init__(self, dim, mult=4, bias=True):
        nn.Module.__init__(self)
        hidden_dim = int(dim * mult)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)


class MiniMaxH3ImageTransformerBlock(TransformerBlock):
    # Pre-norm block of the H3 image VAE decoder: `TransformerBlock.forward` (Q/K RMSNorm + partial
    # RoPE attention, gated FFN, `scale1` / `scale2` residual multipliers) is reused, only the FFN
    # expansion factor differs.
    def __init__(self, heads, dim_head, embed_dim=None, ffn_mult=4, norm_affine=True,
                 qk_norm_type="rms_norm", qk_norm_affine=False, bias=True, eps=1e-5):
        super().__init__(heads=heads, dim_head=dim_head, embed_dim=embed_dim, norm_affine=norm_affine,
                         qk_norm_type=qk_norm_type, qk_norm_affine=qk_norm_affine, bias=bias, eps=eps)
        dim = embed_dim or dim_head * heads
        self.ff = MiniMaxH3ImageFeedForward(dim=dim, mult=ffn_mult, bias=bias)


class ViT2DDecoder(nn.Module):
    # Non-causal 2D ViT decoder: every latent pixel becomes one token, `num_register_tokens` learned
    # register tokens plus one all-zero token (all at position `(0, 0)`) are appended and dropped
    # again after the blocks, then every token is projected onto one `patch_size` pixel patch.
    def __init__(self, patch_size=16, in_channels=24, out_channels=3, num_layers=24, heads=16,
                 dim_head=64, ffn_mult=4, rope_theta=100.0, rope_dim_ratio=0.5,
                 norm_eps=1e-5, num_register_tokens=4):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        # RotaryEmbeddingND with two axes is exactly the reference 2D RoPE: coordinates normalised to
        # [-1, 1) and scaled by 2 * pi, the two axes flattened and then tiled, so the first
        # `int(dim_head * rope_dim_ratio)` head channels are rotated.
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=2, use_angle=True)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = WarpedTensor(torch.randn(1, num_register_tokens, dim) * 0.02) if num_register_tokens > 0 else None
        self.transformer_blocks = nn.ModuleList([
            MiniMaxH3ImageTransformerBlock(heads=heads, dim_head=dim_head, ffn_mult=ffn_mult,
                                           qk_norm_affine=False, qk_norm_type="rms_norm", eps=norm_eps)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)

        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)

    def forward(self, x):
        # x: (B, in_channels, h, w), (B, out_channels, h * patch_size, w * patch_size) out.
        B, C, h, w = x.shape
        num_suffix = 1 + self.num_register_tokens

        hidden_states = self.x_embedder(x.permute(0, 2, 3, 1).reshape(B, h * w, C))
        num_patches = hidden_states.shape[1]
        tokens = [hidden_states]
        if self.register_tokens is not None:
            tokens.append(self.register_tokens().expand(B, -1, -1).to(hidden_states.dtype))
        tokens.append(torch.zeros_like(hidden_states[:, 0:1]))
        hidden_states = torch.cat(tokens, dim=1)

        img_ids = torch.cat([
            create_token_ids((h, w), x.device, torch.float32).expand(B, -1, -1),
            torch.zeros(B, num_suffix, 2, device=x.device, dtype=torch.float32),
        ], dim=1)
        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_pos_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]
        hidden_states = hidden_states.view(B, h, w, self.out_channels, self.patch_size, self.patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(B, self.out_channels, h * self.patch_size, w * self.patch_size)


class MiniMaxH3ImageVAE(nn.Module):
    """MiniMax-H3-Image VAE: causal 3D CNN encoder (16x) + non-causal 2D ViT decoder.

    `encode` maps `(B, 3, H, W)` pixels in ImageNet-normalised `[0, 1]` space to a
    `DiagonalGaussianDistribution` over `(B, latent_channels, H / 16, W / 16)` latents (`.mean`,
    `.logvar`, `.mode()`, `.sample(generator=...)`); `decode` maps such latents back to
    `(B, 3, 16 h, 16 w)` pixels. Neither applies `latents_mean` / `latents_std`, the pipeline does.

    Tiling is on by default (released geometry: 256 px tiles with at least 64 px overlap, the slack
    widened in 16 px steps so tile boundaries stay latent-aligned) and can be switched off per call
    with `tiling=False`.
    """

    def __init__(self, in_channels=3, out_channels=3, latent_channels=24,
                 block_out_channels=(128, 256, 256, 512, 512, 1024), layers_per_block=2,
                 spatial_downsample_factors=(2, 2, 2, 2, 1, 1), temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
                 norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect",
                 decoder_num_layers=24, decoder_num_attention_heads=16, decoder_attention_head_dim=64,
                 decoder_num_register_tokens=4, decoder_ffn_mult=4, decoder_rope_theta=100.0,
                 decoder_rope_dim_ratio=0.5, decoder_norm_eps=1e-5,
                 latents_mean=_IMAGE_LATENTS_MEAN, latents_std=_IMAGE_LATENTS_STD,
                 tile_size=256, tile_overlap_min=64, **kwargs):
        super().__init__()
        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.vae_ratio = self.spatial_compression_ratio
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = MiniMaxH3ImageEncoder3d(
            in_channels=in_channels, out_channels=2 * latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups, norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.decoder = ViT2DDecoder(
            patch_size=self.spatial_compression_ratio, in_channels=latent_channels, out_channels=out_channels,
            num_layers=decoder_num_layers, heads=decoder_num_attention_heads,
            dim_head=decoder_attention_head_dim, ffn_mult=decoder_ffn_mult, rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio, norm_eps=decoder_norm_eps,
            num_register_tokens=decoder_num_register_tokens,
        )

        # Convenience only: `encode` does not normalise the pixels. The video VAE ships the same helper.
        self.processor = VAEProcessor(
            Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            Normalize(tuple(-m / s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)), tuple(1.0 / s for s in _IMAGENET_STD)),
        )
        # Non-persistent buffers: the released checkpoint stores these values in its config.json, so
        # keeping them out of `state_dict()` leaves the converted checkpoint exactly matched.
        self.register_buffer("latents_mean", torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    def _split_tiles(self, length, tile_size, min_overlap):
        """Lay `tile_size`-wide tiles over `length` pixels.

        The tile count is the smallest one whose union covers `length` while keeping every overlap at
        least `min_overlap`; the slack is then spread over the overlaps round-robin in whole
        `spatial_compression_ratio` steps so every boundary stays latent-aligned.
        """
        ratio = self.spatial_compression_ratio
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // ratio):
            overlaps[i % (num_tiles - 1)] += ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a, b, blend_extent, dim):
        """Linearly cross-fade the tail of `a` with the head of `b` along `dim`."""
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        """Blend a grid of tiles back together (vertical blend uses the raw tile of the row above)."""
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode_tile(self, x):
        # The encoder is a causal video encoder; an image is a single frame, i.e. a single latent frame.
        return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]

    def _encode(self, x):
        ratio = self.spatial_compression_ratio
        height, width = x.shape[-2], x.shape[-1]
        y_starts, y_lens, y_overlaps = self._split_tiles(height, self.tile_size, self.tile_overlap_min)
        x_starts, x_lens, x_overlaps = self._split_tiles(width, self.tile_size, self.tile_overlap_min)
        rows = []
        for i_pos, i_len in zip(y_starts, y_lens):
            row = []
            for j_pos, j_len in zip(x_starts, x_lens):
                row.append(self._encode_tile(x[..., i_pos: i_pos + i_len, j_pos: j_pos + j_len]))
            rows.append(row)
        return self._stitch_tiles(rows, [overlap // ratio for overlap in y_overlaps],
                                  [overlap // ratio for overlap in x_overlaps])

    def _decode(self, z):
        ratio = self.spatial_compression_ratio
        height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
        y_starts, y_lens, y_overlaps = self._split_tiles(height, self.tile_size, self.tile_overlap_min)
        x_starts, x_lens, x_overlaps = self._split_tiles(width, self.tile_size, self.tile_overlap_min)
        rows = []
        for i_pos, i_len in zip(y_starts, y_lens):
            row = []
            for j_pos, j_len in zip(x_starts, x_lens):
                tile = z[..., i_pos // ratio: i_pos // ratio + i_len // ratio,
                         j_pos // ratio: j_pos // ratio + j_len // ratio]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def encode(self, x, tiling=True, sample=False, generator=None):
        if hasattr(self.encoder.conv_in, "computation_dtype") and hasattr(self.encoder.conv_in, "computation_device"):
            x = x.to(dtype=self.encoder.conv_in.computation_dtype, device=self.encoder.conv_in.computation_device)
        else:
            x = x.to(self.encoder.conv_in.weight.dtype)
        moments = self._encode(x) if tiling else self._encode_tile(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample(generator=generator) if sample else posterior

    def decode(self, z, tiling=True):
        if hasattr(self.post_quant_conv, "computation_dtype") and hasattr(self.post_quant_conv, "computation_device"):
            z = z.to(dtype=self.post_quant_conv.computation_dtype, device=self.post_quant_conv.computation_device)
        else:
            z = z.to(self.post_quant_conv.weight.dtype)
        return self._decode(z) if tiling else self.decoder(self.post_quant_conv(z))

    def forward(self, x, tiling=True, sample=False, generator=None):
        posterior = self.encode(x, tiling=tiling)
        z = posterior.sample(generator=generator) if sample else posterior.mode()
        return self.decode(z, tiling=tiling)


__all__ = ["MiniMaxH3ImageVAE"]
