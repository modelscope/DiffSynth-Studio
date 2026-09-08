from __future__ import annotations

import dataclasses
import itertools
import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Final, List, Literal, NamedTuple, Protocol, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import functional


class Disposable:
    pass
class VideoDecoder:
    pass
def _clip_generators(count, generator):
    if isinstance(generator, Sequence):
        if len(generator) != count:
            raise ValueError(f"decode_single_frames got {count} latents and {len(generator)} generators")
        return generator
    return [generator] * count


def iter_decoded_single_frames(decoder, latents, generator=None):
    generators = _clip_generators(len(latents), generator)
    for index, (latent, item_generator) in enumerate(zip(latents, generators, strict=True)):
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError(
                f"decode_single_frames expects (B, C, 1, H, W) latents, got {tuple(latent.shape)} at index {index}"
            )
        chunks = list(decoder.decode_video(latent, tiling_config=None, generator=item_generator))
        if not chunks:
            raise RuntimeError(f"Decoder returned no pixels for single-frame latent {index}")
        yield torch.cat(chunks, dim=0)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
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


class VideoPixelShape(NamedTuple):
    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):
    time: int
    height: int
    width: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, height=32, width=32)

    @classmethod
    def from_blocks(cls, blocks: list, patch_size: int) -> "SpatioTemporalScaleFactors":
        spatial_steps = 0
        temporal_steps = 0
        for block_name, _ in blocks:
            if block_name.startswith(("compress_space", "compress_all")):
                spatial_steps += 1
            if block_name.startswith(("compress_time", "compress_all")):
                temporal_steps += 1
        spatial = patch_size * (2**spatial_steps)
        return cls(time=2**temporal_steps, height=spatial, width=spatial)

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    @staticmethod
    def from_pixel_shape(
        shape: VideoPixelShape,
        latent_channels: int = 128,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors.time + 1
        height = shape.height // scale_factors.height
        width = shape.width // scale_factors.width

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )


class AudioLatentShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class Audio:
    waveform: torch.Tensor
    sampling_rate: int

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))


@dataclass(frozen=True)
class GeneratedKeyframeLayout:
    pixel_frame_indices: tuple[int, ...]
    tokens_per_keyframe: int
    first_token: int

    @property
    def num_keyframes(self) -> int:
        return len(self.pixel_frame_indices)

    @property
    def num_tokens(self) -> int:
        return self.num_keyframes * self.tokens_per_keyframe

@dataclass(frozen=True)
class LatentState:
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None
    generated_keyframes: torch.Tensor | None = None
    frozen: bool = False

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            keyframes_mask=self.keyframes_mask.clone() if self.keyframes_mask is not None else None,
            generated_keyframe_layout=self.generated_keyframe_layout,
            generated_keyframes=(self.generated_keyframes.clone() if self.generated_keyframes is not None else None),
            frozen=self.frozen,
        )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] = 0
    if right_ramp > 0:
        mask[-right_ramp:] = 0
    return mask


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class DimensionIntervals:
    intervals: list[DimensionInterval]


@dataclass(frozen=True)
class LatentIntervals:
    original_shape: torch.Size
    dimension_intervals: tuple[DimensionIntervals, ...]


SplitOperation = Callable[[int], DimensionIntervals]


MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(intervals=[DimensionInterval(start=0, end=length, left_ramp=0, right_ramp=0)])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def untiled_mask_1d() -> torch.Tensor:
    return torch.ones(1)


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor]]:
    return [slice(0, None)], [untiled_mask_1d()]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


def _grow_last_tile_to_min(intervals: list[DimensionInterval], min_tile_size: int) -> list[DimensionInterval]:
    if len(intervals) <= 1:
        return list(intervals)
    last = intervals[-1]
    if last.end - last.start >= min_tile_size:
        return list(intervals)
    new_start = last.end - min_tile_size
    prev = intervals[-2]
    new_overlap = prev.end - new_start
    return [
        *intervals[:-2],
        replace(prev, right_ramp=new_overlap),
        replace(last, start=new_start, left_ramp=new_overlap),
    ]


def _validate_tile_intervals(intervals: list[DimensionInterval], *, dim_size: int, min_tile_size: int) -> None:
    if not intervals or intervals[0].start != 0 or intervals[-1].end != dim_size:
        raise ValueError(f"tiles must cover [0, {dim_size})")
    for i, iv in enumerate(intervals):
        length = iv.end - iv.start
        if length < min_tile_size:
            raise ValueError(f"tile {i} length {length} is below min_tile_size={min_tile_size}")
        if iv.left_ramp < 0 or iv.right_ramp < 0 or iv.left_ramp > length or iv.right_ramp > length:
            raise ValueError(f"tile {i} has invalid ramps: left={iv.left_ramp}, right={iv.right_ramp}, length={length}")
        if i == 0:
            continue
        overlap = intervals[i - 1].end - iv.start
        if overlap < 0 or intervals[i - 1].right_ramp != overlap or iv.left_ramp != overlap:
            raise ValueError(f"tiles {i - 1}/{i}: ramp/overlap mismatch (overlap={overlap})")


def split_by_size(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if min_tile_size is not None and dimension_size < min_tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        intervals = [
            DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
            *(
                DimensionInterval(
                    start=i * (size - overlap),
                    end=i * (size - overlap) + size,
                    left_ramp=overlap,
                    right_ramp=overlap,
                )
                for i in range(1, amount - 1)
            ),
            DimensionInterval(
                start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0
            ),
        ]
        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dimension_size, min_tile_size=min_tile_size)
        return DimensionIntervals(intervals=intervals)

    return split


def split_temporal_causal(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    non_causal_split = split_by_size(size, overlap, min_tile_size=min_tile_size)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count_temporal_causal(
    num_tiles: int, overlap: int = 0, min_tile_size: int | None = None
) -> SplitOperation:
    non_causal_split = split_by_count(num_tiles, overlap, min_tile_size=min_tile_size)

    def split(dimension_size: int) -> DimensionIntervals:
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_at_seams(boundaries: Sequence[int], num_tiles: int, overlap: int = 0) -> SplitOperation:
    boundaries = tuple(boundaries)
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError(f"boundaries must start at 0 and hold at least one segment, got {list(boundaries)}")
    if any(b <= a for a, b in itertools.pairwise(boundaries)):
        raise ValueError(f"boundaries must be strictly increasing, got {list(boundaries)}")
    n_segments = len(boundaries) - 1
    n_tiles = min(num_tiles, n_segments)
    base, leftover = divmod(n_segments, n_tiles)
    counts = [base + (1 if index < leftover else 0) for index in range(n_tiles)]

    def split(dim_size: int) -> DimensionIntervals:
        if boundaries[-1] != dim_size - 1:
            raise ValueError(f"boundaries must end at the last cell ({dim_size - 1}), got {boundaries[-1]}")
        intervals: list[DimensionInterval] = []
        cursor = 0
        for tile_index, count in enumerate(counts):
            resume = boundaries[cursor] + 1
            start = 0 if tile_index == 0 else max(0, resume - overlap)
            cursor += count
            intervals.append(
                DimensionInterval(
                    start=start,
                    end=boundaries[cursor] + 1,
                    left_ramp=0 if tile_index == 0 else resume - start,
                    right_ramp=0,
                )
            )
        return DimensionIntervals(intervals=intervals)

    return split


def split_by_count(num_tiles: int, overlap: int = 0, min_tile_size: int | None = None) -> SplitOperation:
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dim_size: int) -> DimensionIntervals:
        if num_tiles > dim_size:
            raise ValueError(
                f"num_tiles ({num_tiles}) exceeds dim_size ({dim_size}). Cannot assign at least 1 unit per tile."
            )
        if num_tiles == 1:
            return DEFAULT_SPLIT_OPERATION(dim_size)

        total = dim_size + overlap * (num_tiles - 1)
        tile_size = total // num_tiles
        if tile_size <= overlap:
            raise ValueError(
                f"split_by_count produced size={tile_size} <= overlap={overlap} "
                f"for dim_size={dim_size}, num_tiles={num_tiles}"
            )
        remainder = total % num_tiles

        base_intervals = split_by_size(tile_size, overlap)(dim_size - remainder).intervals

        # First `remainder` tiles each absorb 1 extra unit; shift subsequent boundaries.
        intervals: list[DimensionInterval] = []
        for i, iv in enumerate(base_intervals):
            shift = min(i, remainder)
            grow = 1 if i < remainder else 0
            intervals.append(replace(iv, start=iv.start + shift, end=iv.end + shift + grow))

        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dim_size, min_tile_size=min_tile_size)

        return DimensionIntervals(intervals=intervals)

    return split


def identity_mapping_operation(
    intervals: DimensionIntervals,
    *,
    rectangular: bool = False,
) -> tuple[list[slice], list[torch.Tensor]]:
    mask_1d = compute_rectangular_mask_1d if rectangular else compute_trapezoidal_mask_1d
    out_slices: list[slice] = []
    masks: list[torch.Tensor] = []
    for iv in intervals.intervals:
        out_slices.append(slice(iv.start, iv.end))
        masks.append(mask_1d(iv.end - iv.start, iv.left_ramp, iv.right_ramp))
    return out_slices, masks


class Tile(NamedTuple):
    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def scale_by_masks_1d(x: torch.Tensor, masks_1d: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(masks_1d) != x.ndim:
        raise ValueError(f"masks_1d length {len(masks_1d)} != x.ndim {x.ndim}")
    out = x
    for axis, mask in enumerate(masks_1d):
        view_shape = [1] * x.ndim
        view_shape[axis] = -1
        out = out * mask.reshape(*view_shape)
    return out


def masks_are_complementary(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
    *,
    atol: float = 1e-5,
) -> bool:
    if not tiles:
        return True
    ndim = len(full_shape)
    for tile in tiles:
        if len(tile.out_coords) != ndim or len(tile.masks_1d) != ndim:
            raise ValueError(
                f"Tile out_coords/masks_1d rank {len(tile.out_coords)}/{len(tile.masks_1d)} != full_shape rank {ndim}"
            )
    for axis, length in enumerate(full_shape):
        # Explicit CPU float32: masks may live on CUDA; a non-CPU default device
        # must not place ``acc`` on GPU (device-mismatch on ``acc[sl] +=``).
        acc = torch.zeros(length, dtype=torch.float32, device="cpu")
        seen: set[tuple[int | None, int | None]] = set()
        for tile in tiles:
            sl = tile.out_coords[axis]
            key = (sl.start, sl.stop)
            if key in seen:
                continue
            seen.add(key)
            # Length-1 untiled masks broadcast over ``acc[sl]``.
            acc[sl] += tile.masks_1d[axis].detach().float().cpu()
        if not torch.allclose(acc, torch.ones(length, dtype=torch.float32), atol=atol, rtol=0.0):
            return False
    return True


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: list[MappingOperation],
) -> list[Tile]:
    full_dim_input_slices: list[list[slice]] = []
    full_dim_output_slices: list[list[slice]] = []
    full_dim_masks_1d: list[list[torch.Tensor]] = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        input_slices = [slice(interval.start, interval.end) for interval in dimension_intervals.intervals]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        n_intervals = len(input_slices)
        if len(output_slices) != n_intervals or len(masks_1d) != n_intervals:
            raise ValueError(
                f"Axis {axis_index}: mapper produced {len(output_slices)} output slices and "
                f"{len(masks_1d)} masks for {n_intervals} input intervals"
            )
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    return [
        Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d)
        for in_coord, out_coord, mask_1d in zip(
            itertools.product(*full_dim_input_slices),
            itertools.product(*full_dim_output_slices),
            itertools.product(*full_dim_masks_1d),
            strict=True,
        )
    ]


def create_tiles(
    latent_shape: torch.Size,
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    if not tiles:
        return []

    groups = []
    current_slice = tiles[0].out_coords[2]
    current_group = []

    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]

    if current_group:
        groups.append(current_group)

    return groups


@dataclass(frozen=True)
class DimensionTilingConfig:
    num_tiles: int = 1
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    def is_tiled(self) -> bool:
        return self.num_tiles > 1 or self.overlap > 0

@dataclass(frozen=True)
class DimensionSizeConfig:
    tile_size: int = 0
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.tile_size < 0:
            raise ValueError(f"tile_size must be >= 0, got {self.tile_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.tile_size == 0:
            if self.overlap != 0:
                raise ValueError("untiled axis (tile_size=0) must have overlap=0")
            return
        if self.overlap >= self.tile_size:
            raise ValueError(f"Overlap must be less than tile size, got {self.overlap} and {self.tile_size}")

    def is_tiled(self) -> bool:
        return self.tile_size > 0


@dataclass(frozen=True)
class TileCountConfig:
    frames: DimensionTilingConfig = DimensionTilingConfig()
    height: DimensionTilingConfig = DimensionTilingConfig()
    width: DimensionTilingConfig = DimensionTilingConfig()

    def validate(self, scale_factors: SpatioTemporalScaleFactors, video_shape: VideoPixelShape) -> None:
        check_temporal = _assert_video_on_vae_grid(scale_factors, video_shape)
        latent_h = video_shape.height // scale_factors.height
        latent_w = video_shape.width // scale_factors.width
        _validate_count_axis(self.height, latent_h, "height")
        _validate_count_axis(self.width, latent_w, "width")
        if check_temporal:
            latent_f = (video_shape.frames - 1) // scale_factors.time + 1
            _validate_count_axis(self.frames, latent_f, "frames")

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        del scale_factors
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def axis_split(cfg: DimensionTilingConfig, axis_min: int | None, *, temporal: bool) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            if temporal and causal_temporal:
                return split_by_count_temporal_causal(cfg.num_tiles, cfg.overlap, min_tile_size=axis_min)
            return split_by_count(cfg.num_tiles, cfg.overlap, min_tile_size=axis_min)

        return (
            axis_split(self.frames, min_t, temporal=True),
            axis_split(self.height, min_h, temporal=False),
            axis_split(self.width, min_w, temporal=False),
        )

@dataclass(frozen=True)
class TileSizeConfig:
    frames: DimensionSizeConfig = DimensionSizeConfig()
    height: DimensionSizeConfig = DimensionSizeConfig()
    width: DimensionSizeConfig = DimensionSizeConfig()

    def validate(self, scale_factors: SpatioTemporalScaleFactors, video_shape: VideoPixelShape) -> None:
        check_temporal = _assert_video_on_vae_grid(scale_factors, video_shape)
        _validate_size_axis(self.height, scale_factors.height, "height")
        _validate_size_axis(self.width, scale_factors.width, "width")
        if check_temporal:
            _validate_size_axis(self.frames, scale_factors.time, "frames")

    @classmethod
    def default(cls) -> TileSizeConfig:
        return cls(
            frames=DimensionSizeConfig(tile_size=80, overlap=24),
            height=DimensionSizeConfig(tile_size=768, overlap=64),
            width=DimensionSizeConfig(tile_size=768, overlap=64),
        )

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def enable_size_axis(
            factor: int,
            axis_min: int | None,
            cfg: DimensionSizeConfig,
            axis_name: str,
            *,
            temporal: bool,
        ) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            _validate_size_axis(cfg, factor, axis_name)
            size = cfg.tile_size // factor
            overlap = cfg.overlap // factor
            lower_threshold = max(2, overlap + 1)
            tile = max(lower_threshold, size)
            if temporal and causal_temporal:
                return split_temporal_causal(tile, overlap, min_tile_size=axis_min)
            return split_by_size(tile, overlap, min_tile_size=axis_min)

        return (
            enable_size_axis(scale_factors.time, min_t, self.frames, "frames", temporal=True),
            enable_size_axis(scale_factors.height, min_h, self.height, "height", temporal=False),
            enable_size_axis(scale_factors.width, min_w, self.width, "width", temporal=False),
        )

TilingConfig = TileSizeConfig | TileCountConfig




def _assert_video_on_vae_grid(
    scale_factors: SpatioTemporalScaleFactors,
    video_shape: VideoPixelShape,
) -> bool:
    if scale_factors.time < 1 or scale_factors.height < 1 or scale_factors.width < 1:
        raise ValueError(f"scale_factors must be >= 1 on each axis, got {scale_factors}")
    if video_shape.height < 1 or video_shape.width < 1:
        raise ValueError(f"video_shape height/width must be >= 1, got {video_shape.height}x{video_shape.width}")
    if video_shape.height % scale_factors.height != 0:
        raise ValueError(f"video height {video_shape.height} must be divisible by scale {scale_factors.height}")
    if video_shape.width % scale_factors.width != 0:
        raise ValueError(f"video width {video_shape.width} must be divisible by scale {scale_factors.width}")
    if video_shape.frames <= 0:
        return False
    if (video_shape.frames - 1) % scale_factors.time != 0:
        raise ValueError(f"video frames {video_shape.frames} must satisfy (frames - 1) % {scale_factors.time} == 0")
    return True


def _validate_size_axis(cfg: DimensionSizeConfig, factor: int, axis_name: str) -> None:
    if not cfg.is_tiled():
        return
    min_size = 2 * factor
    if cfg.tile_size < min_size:
        raise ValueError(f"{axis_name}.tile_size must be at least {min_size}, got {cfg.tile_size}")
    if cfg.tile_size % factor != 0:
        raise ValueError(f"{axis_name}.tile_size must be divisible by {factor}, got {cfg.tile_size}")
    if cfg.overlap % factor != 0:
        raise ValueError(f"{axis_name}.overlap must be divisible by {factor}, got {cfg.overlap}")


def _validate_count_axis(cfg: DimensionTilingConfig, latent_extent: int, axis_name: str) -> None:
    if not cfg.is_tiled():
        return
    if cfg.num_tiles > latent_extent:
        raise ValueError(f"{axis_name}.num_tiles {cfg.num_tiles} exceeds latent {axis_name} extent {latent_extent}")
    # split_by_count requires overlap < tile_size; tile_size grows with extent/n.
    max_overlap = latent_extent - cfg.num_tiles
    if cfg.overlap > max_overlap:
        raise ValueError(
            f"{axis_name}.overlap {cfg.overlap} exceeds latent bound {max_overlap} "
            f"for extent {latent_extent} with {cfg.num_tiles} tiles"
        )


def _validate_overlap(
    tiling_config: TilingConfig,
    *,
    min_overlap_frames: int,
    min_overlap_pixels: int,
) -> None:
    if not isinstance(tiling_config, TileSizeConfig):
        return

    for axis_name, cfg, recommended, unit in (
        ("frames", tiling_config.frames, min_overlap_frames, "frames"),
        ("height", tiling_config.height, min_overlap_pixels, "px"),
        ("width", tiling_config.width, min_overlap_pixels, "px"),
    ):
        if cfg.is_tiled() and cfg.overlap < recommended:
            raise ValueError(f"{axis_name} overlap {cfg.overlap} {unit} is below the required {recommended} {unit}.")


class DiffVAEMode(Enum):
    COMBINED_COMPILE = "combined_compile"
    CHUNKED_COMPILE = "chunked_compile"
    CHUNKED_EAGER = "chunked_eager"
    BLACKWELL_DSL = "blackwell_dsl"

    def resolve(self):
        return self


class NAttentionKind(Enum):
    TRITON = "triton"
    EAGER_SDPA = "eager_sdpa"


@dataclass(frozen=True)
class _ResolvedAttention:
    attention: NAttentionKind = NAttentionKind.EAGER_SDPA
    compile_blocks: bool = False


def resolve_attention_for_host(mode):
    del mode
    return _ResolvedAttention()


def frames_per_yuv_gemm(height: int, width: int) -> int:
    del height, width
    return 2**31 - 1


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)


"""Keyframe (dual-stream) inputs and coordinate math for DiffVAE decode.
A keyframe-aware decode carries two streams through the decoder: the video volume
``(B, T, H, W, C)`` and a stack of keyframe *planes* ``(B, P, H, W, C)`` whose plane
axis occupies video's temporal slot. Weights are fully shared; the streams only ever
mix inside one joint attention softmax (see ``transformer/fallback_na/joint_eager.py``).
Everything here is pure coordinate/geometry math with no module state, so the eager and
triton backends can share it and therefore agree exactly on slot selection.
Deviation from upstream worth knowing: upstream carries per-sample keyframe times and
masks (``(B, n_kf)``). Here they are batch-shared 1-D ``(P,)`` tensors, because our
decode path is single-sample and ``rope_math.rot_abs_axis_impl`` takes a 1-D position
vector per axis. That keeps the RoPE call and the slot tables batch-independent.
"""


KEYFRAME_CONTEXT_SLOTS = 2


@dataclass(frozen=True)
class DecodeKeyframes:
    latents: torch.Tensor
    pixel_frame_indices: torch.Tensor
    clip_start_frame: int = 0

    def validate(self, *, num_frames: int | None = None) -> None:
        if self.latents.ndim != 5:
            raise ValueError(f"keyframe latents must be (B, C, P, H, W), got {tuple(self.latents.shape)}")
        if self.pixel_frame_indices.ndim != 1:
            raise ValueError(f"pixel_frame_indices must be 1-D (P,), got {tuple(self.pixel_frame_indices.shape)}")
        if self.clip_start_frame < 0:
            raise ValueError(f"clip_start_frame must be non-negative, got {self.clip_start_frame}")
        planes = self.latents.shape[2]
        if planes != self.pixel_frame_indices.shape[0]:
            raise ValueError(
                f"keyframe plane count {planes} != len(pixel_frame_indices) {self.pixel_frame_indices.shape[0]}"
            )
        if planes == 0:
            # An empty stack is a plain decode wearing a keyframe decode's costs, and every
            # backend has to special-case it (the slot tables are all -1, and gathering plane 0
            # of an empty axis is an out-of-bounds read). Say so here instead.
            raise ValueError("keyframe decode needs at least one plane; use decode_video() for a plain decode")
        if planes and int(self.pixel_frame_indices.min()) < 0:
            raise ValueError("pixel_frame_indices must be non-negative (global pixel frames)")
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        # Planes may sit outside [clip_start_frame, clip_start_frame + num_frames): Dist tiles
        # keep the nearest plane on each side so |dt| matches a whole-clip decode. A far plane
        # on a full clip is the same geometry -- joint attention ranks it by distance.

    def crop_spatial(self, height: slice, width: slice) -> "DecodeKeyframes":
        return DecodeKeyframes(
            latents=self.latents[:, :, :, height, width],
            pixel_frame_indices=self.pixel_frame_indices,
            clip_start_frame=self.clip_start_frame,
        )

    @property
    def num_planes(self) -> int:
        return int(self.latents.shape[2])


@dataclass(frozen=True)
class KeyframeStream:
    x: torch.Tensor
    times: torch.Tensor
    valid: torch.Tensor

    def masked(self) -> KeyframeStream:
        return KeyframeStream(x=self.x * self.valid[None, :, None, None, None], times=self.times, valid=self.valid)

    def select_planes(self, keep: torch.Tensor) -> KeyframeStream:
        if keep.shape != (self.num_planes,):
            raise ValueError(f"keep must be ({self.num_planes},) bool, got {tuple(keep.shape)}")
        return KeyframeStream(x=self.x[:, keep], times=self.times[keep], valid=self.valid[keep])

    def crop_spatial(self, height: slice, width: slice) -> KeyframeStream:
        return KeyframeStream(x=self.x[:, :, height, width, :], times=self.times, valid=self.valid)

    @property
    def num_planes(self) -> int:
        return int(self.x.shape[1])


def keyframe_stage_times(pixel_frame_indices: torch.Tensor, remaining_time_stride: int) -> torch.Tensor:
    if remaining_time_stride < 1:
        raise ValueError(f"remaining_time_stride must be positive, got {remaining_time_stride}")
    frames = pixel_frame_indices.to(torch.float32)
    center_offset = (remaining_time_stride - 1) / 2
    times = (frames + center_offset) / remaining_time_stride
    return torch.where(frames == 0, torch.zeros_like(times), times)


def keyframe_clip_times(
    pixel_frame_indices: torch.Tensor,
    remaining_time_stride: int,
    clip_start_frame: int,
    extra_origin: float = 0.0,
) -> torch.Tensor:
    times = keyframe_stage_times(pixel_frame_indices, remaining_time_stride)
    origin = keyframe_stage_times(
        torch.as_tensor([clip_start_frame], dtype=torch.int64, device=pixel_frame_indices.device),
        remaining_time_stride,
    )
    return times - origin - extra_origin


def planes_for_tile(
    pixel_frame_indices: torch.Tensor,
    frame_lo: int,
    frame_hi: int,
    *,
    clip_start_frame: int = 0,
) -> torch.Tensor:
    frame_lo = frame_lo + clip_start_frame
    frame_hi = frame_hi + clip_start_frame
    indices = pixel_frame_indices.to(torch.int64)
    keep = (indices >= frame_lo) & (indices <= frame_hi)
    before = indices < frame_lo
    if bool(before.any()):
        # Latest plane strictly before the tile.
        keep[int(torch.where(before, indices, torch.full_like(indices, -1)).argmax())] = True
    after = indices > frame_hi
    if bool(after.any()):
        # Earliest plane strictly after the tile.
        sentinel = int(indices.max()) + 1
        keep[int(torch.where(after, indices, torch.full_like(indices, sentinel)).argmin())] = True
    return keep


def remaining_time_strides(upsamples: Sequence[torch.nn.Module]) -> tuple[int, ...]:
    strides = [int(up.stride[0]) for up in upsamples]
    remaining: list[int] = []
    for index in range(len(strides)):
        product = 1
        for stride in strides[index:]:
            product *= stride
        remaining.append(product)
    remaining.append(1)
    return tuple(remaining)


def upsample_keyframe_planes(upsample: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    planes = x.shape[1]
    flat = rearrange(x, "b p h w c -> (b p) 1 h w c")
    upsampled = upsample(flat, drop_leading_frame=True)
    if upsampled.shape[1] != 1:
        raise RuntimeError(f"isolated keyframe upsampling must preserve one temporal plane, got T={upsampled.shape[1]}")
    out = rearrange(upsampled[:, 0], "(b p) h w c -> b p h w c", p=planes)
    if out.shape[1] != planes:
        raise RuntimeError(f"keyframe plane count changed under upsample: {planes} -> {out.shape[1]}")
    return out


def _nearest_slots(
    query_times: torch.Tensor,
    candidate_times: torch.Tensor,
    candidate_valid: torch.Tensor | None,
    num_slots: int,
) -> torch.Tensor:
    distances = (query_times[:, None] - candidate_times[None, :]).abs().to(torch.float32)
    if candidate_valid is not None:
        distances = distances.masked_fill(~candidate_valid[None, :], float("inf"))
    order = torch.argsort(distances, dim=-1, stable=True)
    take = min(num_slots, candidate_times.shape[0])
    chosen = order[:, :take]
    # Drop slots that only exist because every remaining candidate was invalid.
    finite = torch.gather(distances, 1, chosen).isfinite()
    chosen = torch.where(finite, chosen, torch.full_like(chosen, -1))
    if take < num_slots:
        pad = torch.full((chosen.shape[0], num_slots - take), -1, dtype=chosen.dtype, device=chosen.device)
        chosen = torch.cat([chosen, pad], dim=1)
    return chosen


def video_keyframe_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    query = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    return _nearest_slots(query, keyframe_times.to(torch.float32), keyframe_valid, num_slots)


def keyframe_video_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    candidates = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    slots = _nearest_slots(keyframe_times.to(torch.float32), candidates, None, num_slots)
    return torch.where(keyframe_valid[:, None], slots, torch.full_like(slots, -1))


"""DiffVAE tiling helpers: schedule, pad/crop/size-floor, blend utilities.
Decode orchestration lives on ``DiffusionVideoDecoder``. This module owns the
geometry/schedule/mask pieces that tiling uses.
"""


ResizeAxisMode = Literal["repeat_last", "symmetric"]


_GIB: int = 1 << 30


@dataclass(frozen=True, slots=True)
class _StageFiveBudget:
    coef: float
    coef_keyframes: float
    reserve_bytes: int


_BUDGET_BY_MODE: dict[DiffVAEMode, _StageFiveBudget] = {
    DiffVAEMode.COMBINED_COMPILE: _StageFiveBudget(coef=11, coef_keyframes=15, reserve_bytes=2 * _GIB),
    DiffVAEMode.CHUNKED_COMPILE: _StageFiveBudget(coef=7, coef_keyframes=5, reserve_bytes=2 * _GIB),
    DiffVAEMode.CHUNKED_EAGER: _StageFiveBudget(coef=5, coef_keyframes=5, reserve_bytes=1 * _GIB),
    DiffVAEMode.BLACKWELL_DSL: _StageFiveBudget(coef=2.5, coef_keyframes=2.5, reserve_bytes=2 * _GIB),
}


_DEFAULT_ELEMENT_SIZE: int = 2  # bf16 features → fp16 accumulator / bf16 stage-5


_ACCUMULATOR_CHANNELS: int = 3  # RGB pixel blend buffer (decoder out_channels)


_MIN_MODEL_BYTES_FLOOR: int = 1 << 30  # never assume a free DiffVAE weight footprint


_BUDGET_SAFETY_BYTES_EAGER: int = 1 * _GIB


_BUDGET_SAFETY_BYTES_JOINT_MATERIALIZED: int = 2 * _GIB


def _falls_back_to_eager_na(mode: DiffVAEMode) -> bool:
    resolved = resolve_attention_for_host(mode.resolve())
    return resolved.attention in (NAttentionKind.TRITON, NAttentionKind.EAGER_SDPA) and not resolved.compile_blocks


def stage5_mem_coef(mode: DiffVAEMode, *, keyframes: bool = False) -> float:
    try:
        budget = _BUDGET_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported DiffVAEMode for tiling budget: {mode!r}") from exc
    if keyframes:
        return budget.coef_keyframes
    if _falls_back_to_eager_na(mode):
        return _BUDGET_BY_MODE[DiffVAEMode.CHUNKED_EAGER].coef
    return budget.coef


_CONVERT_PEAK_UV_CHANNELS_X2 = 5  # uv_full (2) + pooled uv (0.5)


_PACK_PEAK_UV_CHANNELS_X2 = 4  # pooled uv (0.5) + packed float (1.5)


_PACK_PEAK_UINT8_BYTES_X2 = 3  # 1.5 B/px


def max_emitted_frames(*, num_frames: int, tile_frames: int, overlap_frames: int) -> int:
    if tile_frames >= num_frames:
        return num_frames
    stride = tile_frames - overlap_frames
    if stride <= 0:
        return tile_frames
    n_tiles = 1 + -(-(num_frames - tile_frames) // stride)
    last_group = num_frames - (n_tiles - 1) * stride
    return min(tile_frames, max(stride, last_group + 1))


def emit_convert_bytes(
    *,
    tile_frames: int,
    height: int,
    width: int,
    out_channels: int,
    element_size: int,
) -> int:
    frames = int(tile_frames)
    gemm_frames = min(frames, frames_per_yuv_gemm(height, width))
    # Write-back reuses the RGB storage, so no full YUV tensor survives into the pack.
    resident_yuv_frames = frames if gemm_frames == frames else 0
    yuv_channels_x2 = 2 * int(out_channels)

    convert_peak_x2 = element_size * (yuv_channels_x2 * gemm_frames + _CONVERT_PEAK_UV_CHANNELS_X2 * frames)
    pack_peak_x2 = (
        element_size * (yuv_channels_x2 * resident_yuv_frames + _PACK_PEAK_UV_CHANNELS_X2 * frames)
        + _PACK_PEAK_UINT8_BYTES_X2 * frames
    )

    return int(height) * int(width) * max(convert_peak_x2, pack_peak_x2) // 2


def budget_safety_bytes(
    mode: DiffVAEMode,
    *,
    keyframes: bool = False,
    joint_sdpa_materializes: bool = False,
) -> int:
    try:
        budget = _BUDGET_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported DiffVAEMode for tiling budget: {mode!r}") from exc
    if keyframes and mode is not DiffVAEMode.BLACKWELL_DSL:
        return _BUDGET_SAFETY_BYTES_JOINT_MATERIALIZED if joint_sdpa_materializes else _BUDGET_SAFETY_BYTES_EAGER
    if _falls_back_to_eager_na(mode):
        return _BUDGET_SAFETY_BYTES_EAGER
    return budget.reserve_bytes


def accumulator_element_size(feature_dtype: torch.dtype) -> int:
    if feature_dtype is torch.bfloat16:
        return 2  # stored as fp16
    return int(torch.tensor([], dtype=feature_dtype).element_size())


def stage4_feature_bytes(
    *,
    height: int,
    width: int,
    num_frames: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    stage4_channels: int,
    element_size: int = _DEFAULT_ELEMENT_SIZE,
    natten_trailing_pad_latent_frames: int = 0,
) -> int:
    if stage4_channels < 1:
        raise ValueError(f"stage4_channels must be >= 1, got {stage4_channels}")
    if element_size < 1:
        raise ValueError(f"element_size must be >= 1, got {element_size}")
    if len(upsample_strides) < 3:
        raise ValueError(f"need at least 3 upsample strides, got {len(upsample_strides)}")
    if natten_trailing_pad_latent_frames < 0:
        raise ValueError(f"natten_trailing_pad_latent_frames must be >= 0, got {natten_trailing_pad_latent_frames}")

    # Local import: types ↔ tiling cycle avoidance at module import time.
    from ltx_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape  # noqa: PLC0415

    latent = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(batch=1, frames=int(num_frames), height=int(height), width=int(width), fps=24.0),
        scale_factors=VIDEO_SCALE_FACTORS,
    )
    s4_t, s4_h, s4_w = stage4_thw_from_latent(
        upsample_strides[:3],
        latent.frames + int(natten_trailing_pad_latent_frames),
        latent.height,
        latent.width,
        drop_leading_frame=True,
    )
    return int(s4_t) * int(s4_h) * int(s4_w) * int(stage4_channels) * int(element_size)


def recommended_decode_tiling_config(  # noqa: PLR0913
    *,
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
    min_tile_size_s4: Tuple[int, int, int],
    patch_size: int,
    height: int,
    width: int,
    num_frames: int,
    mode: DiffVAEMode,
    free_bytes: int,
    stage5_channels: int,
    stage4_channels: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    model_bytes: int = 0,
    element_size: int = _DEFAULT_ELEMENT_SIZE,
    natten_trailing_pad_latent_frames: int = 0,
    out_channels: int = _ACCUMULATOR_CHANNELS,
    keyframes: bool = False,
    joint_sdpa_materializes: bool = False,
) -> TileSizeConfig:
    if height < 1 or width < 1 or num_frames < 1:
        raise ValueError(f"height/width/num_frames must be >= 1, got {height}x{width}x{num_frames}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if stage5_channels < 1:
        raise ValueError(f"stage5_channels must be >= 1, got {stage5_channels}")
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    if element_size < 1:
        raise ValueError(f"element_size must be >= 1, got {element_size}")

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)

    ft, fh, fw = pixel_scale.time, pixel_scale.height, pixel_scale.width
    # Construction validates fixed 8/32/32; to_splitters uses pixel_scale - step both.
    step_t = math.lcm(ft, VIDEO_SCALE_FACTORS.time)
    step_h = math.lcm(fh, VIDEO_SCALE_FACTORS.height)
    step_w = math.lcm(fw, VIDEO_SCALE_FACTORS.width)
    min_t_px = _round_up(
        # ``2 * overlap`` so left+right ramps fit (else masks are not complementary and
        # decode allocates a full weights buffer ≈ another accumulator).
        max(2 * ft, 2 * overlap_t, _round_up(min_tile_size_s4[0] * ft, ft), 16),
        step_t,
    )
    min_h_px = _round_up(
        max(2 * fh, 2 * overlap_hw, _round_up(min_tile_size_s4[1] * fh, fh), 64),
        step_h,
    )
    min_w_px = _round_up(
        max(2 * fw, 2 * overlap_hw, _round_up(min_tile_size_s4[2] * fw, fw), 64),
        step_w,
    )

    model_cost = max(int(model_bytes), _MIN_MODEL_BYTES_FLOOR)
    coef = stage5_mem_coef(mode, keyframes=keyframes)
    s4_feat_bytes = stage4_feature_bytes(
        height=height,
        width=width,
        num_frames=num_frames,
        upsample_strides=upsample_strides,
        stage4_channels=stage4_channels,
        element_size=element_size,
        natten_trailing_pad_latent_frames=natten_trailing_pad_latent_frames,
    )
    reserve = budget_safety_bytes(mode, keyframes=keyframes, joint_sdpa_materializes=joint_sdpa_materializes)
    usable = max(0, int(free_bytes) - model_cost - reserve - s4_feat_bytes)
    s5_bytes_per_token = max(1.0, float(stage5_channels) * float(element_size) * coef)
    acc_bytes_per_pixel = int(out_channels) * int(element_size)

    t_cands = _axis_candidates(num_frames, overlap_t, min_t_px, step_t)
    h_cands = _axis_candidates(height, overlap_hw, min_h_px, step_h)
    w_cands = _axis_candidates(width, overlap_hw, min_w_px, step_w)

    scored: list[tuple[float, int, int, int, int, int]] = []
    # (waste, -volume, n_t*n_h*n_w, tile_t, tile_h, tile_w) - minimize waste, then launches.
    for tile_t, n_t in t_cands:
        # Current group buffer + still-live emit/stub during temporal handoff.
        acc_frames = 2 * int(tile_t)
        acc_bytes = acc_frames * int(height) * int(width) * acc_bytes_per_pixel
        # The consumer converts each yielded chunk while this decode is suspended.
        downstream_bytes = emit_convert_bytes(
            tile_frames=max_emitted_frames(num_frames=num_frames, tile_frames=tile_t, overlap_frames=overlap_t),
            height=height,
            width=width,
            out_channels=out_channels,
            element_size=element_size,
        )
        if acc_bytes + downstream_bytes >= usable:
            continue
        s5_budget_bytes = usable - acc_bytes - downstream_bytes
        max_s5_tokens = int(s5_budget_bytes // s5_bytes_per_token)
        for tile_h, n_h in h_cands:
            for tile_w, n_w in w_cands:
                if stage5_tokens_for_pixel_tile(tile_t, tile_h, tile_w, patch_size=patch_size) > max_s5_tokens:
                    continue
                waste = volumetric_overlap_waste(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tile_frames=tile_t,
                    tile_height=tile_h,
                    tile_width=tile_w,
                    n_t=n_t,
                    n_h=n_h,
                    n_w=n_w,
                )
                scored.append((waste, -tile_t * tile_h * tile_w, n_t * n_h * n_w, tile_t, tile_h, tile_w))

    if not scored:
        raise ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            f"min tile ~{min_t_px}f x {min_h_px}x{min_w_px}px "
            f"(overlaps T={overlap_t}, HW={overlap_hw}), "
            f"mode={mode.value}, keyframes={keyframes}, coef={coef}, stage5_channels={stage5_channels}, "
            f"stage4_feature_bytes={s4_feat_bytes}, usable_bytes={usable}. "
            "Reduce resolution, reduce num_frames (stage-4 features and the per-chunk "
            "encode buffers both scale with it), or free GPU memory."
        )

    scored.sort()
    _waste, _vol, _ntiles, tile_t, tile_h, tile_w = scored[0]
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=tile_t, overlap=overlap_t),
        height=DimensionSizeConfig(tile_size=tile_h, overlap=overlap_hw),
        width=DimensionSizeConfig(tile_size=tile_w, overlap=overlap_hw),
    )


def prepare_tile_schedule(
    stage4_shape_bcthw: torch.Size,
    tiling_config: TilingConfig | None,
    *,
    upsample3_stride: Tuple[int, int, int],
    patch_size: int,
    min_tile_size: Tuple[int, int, int],
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
) -> List[Tile]:
    pixel_scale = stage4_to_pixel_scale_factors(upsample3_stride, patch_size)
    if tiling_config is None:
        return [
            Tile(
                in_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                out_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                masks_1d=(
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                ),
            )
        ]

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)
    _validate_overlap(tiling_config, min_overlap_frames=overlap_t, min_overlap_pixels=overlap_hw)
    # Plain split (not split_temporal_causal): no start-1 / left_ramp+1 copycat of ConvVAE.
    t_split, h_split, w_split = tiling_config.to_splitters(
        pixel_scale, min_tile_size=min_tile_size, causal_temporal=False
    )
    st, sh, sw = upsample3_stride

    def axis_specs(
        split_op: SplitOperation,
        dim_len: int,
        stride_component: int,
        *,
        propagate_causal: bool,
        apply_patch: bool,
    ) -> list[tuple[slice, slice, torch.Tensor]]:
        if split_op is DEFAULT_SPLIT_OPERATION:
            return [(slice(None), slice(None), untiled_mask_1d())]
        intervals = split_op(dim_len).intervals
        specs = []
        for iv in intervals:
            stage5 = _propagate_interval_through_upsample_hops(iv, [stride_component], propagate_causal)
            if apply_patch:
                pixel = _propagate_interval_through_upsample_hops(stage5, [patch_size], causal=False)
            else:
                pixel = stage5
            # Symmetric ramps (left_starts_from_0=False) for partition-of-unity with
            # pixel-shuffle out_coords; ConvVAE sacrificial first-sample is not used.
            mask_pixel = compute_trapezoidal_mask_1d(
                pixel.end - pixel.start, pixel.left_ramp, pixel.right_ramp, left_starts_from_0=False
            )
            specs.append((slice(iv.start, iv.end), slice(pixel.start, pixel.end), mask_pixel))
        return specs

    # Temporal: pixel-shuffle propagate (drop-leading geometry); spatial: exact x stride.
    t_specs = axis_specs(t_split, stage4_shape_bcthw[2], st, propagate_causal=True, apply_patch=False)
    h_specs = axis_specs(h_split, stage4_shape_bcthw[3], sh, propagate_causal=False, apply_patch=True)
    w_specs = axis_specs(w_split, stage4_shape_bcthw[4], sw, propagate_causal=False, apply_patch=True)

    tiles: List[Tile] = []
    for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
        t_s4, t_px, t_mask = t_spec
        h_s4, h_px, h_mask = h_spec
        w_s4, w_px, w_mask = w_spec
        tiles.append(
            Tile(
                in_coords=(slice(None), t_s4, h_s4, w_s4, slice(None)),
                out_coords=(slice(None), slice(None), t_px, h_px, w_px),
                masks_1d=(untiled_mask_1d(), untiled_mask_1d(), t_mask, h_mask, w_mask),
            )
        )
    return tiles


def slice_stage4_tile(
    feat_s4: torch.Tensor,
    tile: Tile,
    *,
    content_frames: int,
) -> tuple[torch.Tensor, bool, bool, tuple[int, int, int]]:
    is_origin = tile.in_coords[1].start in (0, None)
    _, stop, _ = tile.in_coords[1].indices(content_frames)
    pad_trailing = stop == content_frames
    _b, t_coord, h_coord, w_coord, _c = tile.in_coords
    t0, t1, _ = t_coord.indices(content_frames)
    h0, h1, _ = h_coord.indices(feat_s4.shape[2])
    w0, w1, _ = w_coord.indices(feat_s4.shape[3])
    content_thw = (t1 - t0, h1 - h0, w1 - w0)
    if pad_trailing:
        t1 = feat_s4.shape[1]
    feat_tile = feat_s4[:, t0:t1, h_coord, w_coord, :]
    return feat_tile, is_origin, pad_trailing, content_thw


@dataclass(frozen=True)
class AxisPad:
    before: int
    after: int


def resize_axis(
    x: torch.Tensor,
    dim: int,
    size: int,
    *,
    mode: ResizeAxisMode,
) -> tuple[torch.Tensor, AxisPad]:
    if size < 1:
        raise ValueError(f"resize_axis target size must be >= 1, got {size}")
    if dim < 0:
        dim += x.ndim
    if not 0 <= dim < x.ndim:
        raise ValueError(f"dim {dim} out of range for rank-{x.ndim} tensor")

    length = x.shape[dim]
    if length == size:
        return x, AxisPad(0, 0)

    if length < size:
        need = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            expand_shape = list(x.shape)
            expand_shape[dim] = need
            pad = last.expand(expand_shape)
            return torch.cat([x, pad], dim=dim), AxisPad(0, need)

        before = need // 2
        after = need - before
        first = x.narrow(dim, 0, 1)
        last = x.narrow(dim, length - 1, 1)
        parts: list[torch.Tensor] = []
        if before:
            expand_shape = list(x.shape)
            expand_shape[dim] = before
            parts.append(first.expand(expand_shape))
        parts.append(x)
        if after:
            expand_shape = list(x.shape)
            expand_shape[dim] = after
            parts.append(last.expand(expand_shape))
        return torch.cat(parts, dim=dim), AxisPad(before, after)

    need = length - size
    if mode == "repeat_last":
        return x.narrow(dim, 0, size).contiguous(), AxisPad(0, need)

    before = need // 2
    after = need - before
    return x.narrow(dim, before, size).contiguous(), AxisPad(before, after)


def ensure_min_latent_shape(
    latent: torch.Tensor,
    min_tile_sizes: Tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[AxisPad, AxisPad, AxisPad]]:
    min_t, min_h, min_w = min_tile_sizes
    t_pad = AxisPad(0, 0)
    h_pad = AxisPad(0, 0)
    w_pad = AxisPad(0, 0)
    x = latent
    if x.shape[2] < min_t:
        x, t_pad = resize_axis(x, 2, min_t, mode="repeat_last")
    if x.shape[3] < min_h:
        x, h_pad = resize_axis(x, 3, min_h, mode="symmetric")
    if x.shape[4] < min_w:
        x, w_pad = resize_axis(x, 4, min_w, mode="symmetric")
    return x, (t_pad, h_pad, w_pad)


def scale_axis_pad(pad: AxisPad, scale: int) -> AxisPad:
    return AxisPad(pad.before * scale, pad.after * scale)


def crop_pixels_to_content(
    pixels: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    *,
    h_pad: AxisPad | None = None,
    w_pad: AxisPad | None = None,
    spatial_scale: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    x, _ = resize_axis(pixels, 2, frames, mode="repeat_last")
    scale_h, scale_w = spatial_scale
    if h_pad is not None:
        before = scale_axis_pad(h_pad, scale_h).before
        if before + height > x.shape[3]:
            raise ValueError(f"H crop out of range: before={before}, height={height}, got {x.shape[3]}")
        x = x.narrow(3, before, height).contiguous()
    else:
        x, _ = resize_axis(x, 3, height, mode="symmetric")
    if w_pad is not None:
        before = scale_axis_pad(w_pad, scale_w).before
        if before + width > x.shape[4]:
            raise ValueError(f"W crop out of range: before={before}, width={width}, got {x.shape[4]}")
        x = x.narrow(4, before, width).contiguous()
    else:
        x, _ = resize_axis(x, 4, width, mode="symmetric")
    return x


def stage5_pixel_shape_from_stage4(
    stage4_t: int,
    stage4_h: int,
    stage4_w: int,
    *,
    upsample_stride: Tuple[int, int, int],
    patch_size: int,
    stage5_kernel_t: int,
    drop_leading_frame: bool,
    pad_trailing: bool,
) -> tuple[int, int, int]:
    st, sh, sw = upsample_stride
    frames = stage4_t * st - 1 if drop_leading_frame and st == 2 else stage4_t * st
    if pad_trailing:
        frames = max(frames, stage5_kernel_t)
    return frames, stage4_h * sh * patch_size, stage4_w * sw * patch_size


def pad_trailing_latent_for_natten_border(latent: torch.Tensor, n_frames: int) -> torch.Tensor:
    if n_frames <= 0:
        return latent
    padded, _ = resize_axis(latent, 2, latent.shape[2] + n_frames, mode="repeat_last")
    return padded


def crop_trailing_context_natten_pad(
    context: torch.Tensor,
    *,
    n_latent_frames: int,
    time_scale: int,
    stage5_kernel_t: int,
) -> torch.Tensor:
    if n_latent_frames <= 0:
        return context
    ghost = n_latent_frames * time_scale
    content_t = max(context.shape[1] - ghost, 1)
    keep = min(context.shape[1], max(content_t, stage5_kernel_t))
    cropped, _ = resize_axis(context, 1, keep, mode="repeat_last")
    return cropped


def _weight_floor(dtype: torch.dtype) -> float:
    return max(1e-8, torch.finfo(dtype).tiny)


def stage4_thw_from_latent(
    upsample_strides: Sequence[Tuple[int, int, int]],
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    drop_leading_frame: bool = True,
) -> Tuple[int, int, int]:
    t, h, w = latent_t, latent_h, latent_w
    for st, sh, sw in upsample_strides[:3]:
        t, h, w = t * st, h * sh, w * sw
        if st == 2 and drop_leading_frame:
            t -= 1
    return t, h, w


def stage4_to_pixel_scale_factors(
    upsample_stride: Tuple[int, int, int],
    patch_size: int,
) -> SpatioTemporalScaleFactors:
    st, sh, sw = upsample_stride
    return SpatioTemporalScaleFactors(time=st, height=sh * patch_size, width=sw * patch_size)


def compute_tile_min_size(
    stage4_kernel: Tuple[int, int, int],
    stage5_kernel: Tuple[int, int, int],
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    return tuple(max(stage4_kernel[a], -(-stage5_kernel[a] // upsample3_stride[a])) for a in range(3))


def compute_tile_halos(
    stage4_kernel: Tuple[int, int, int],
    stage4_depth: int,
    stage5_kernel: Tuple[int, int, int],
    stage5_depth: int,
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    halo4 = tuple(stage4_depth * (stage4_kernel[a] // 2) for a in range(3))
    halo5 = tuple(-(-(stage5_depth * (stage5_kernel[a] // 2)) // upsample3_stride[a]) for a in range(3))
    return halo4, halo5  # type: ignore[return-value]


def _cumulative_upsample_strides(
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
) -> List[Tuple[int, int, int]]:
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    return cumulative


def all_stages_min_tile_size(
    stage_kernels: Sequence[Tuple[int, int, int]],
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    cumulative = _cumulative_upsample_strides(upsamples)
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return (mins[0], mins[1], mins[2])


def pixel_tile_shape(full_shape: tuple[int, ...], out_coords: tuple[slice, ...]) -> tuple[int, ...]:
    dims: list[int] = []
    for size, coord in zip(full_shape, out_coords, strict=True):
        start, stop, step = coord.indices(size)
        dims.append(len(range(start, stop, step)))
    return tuple(dims)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def recommended_pixel_overlaps(
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
) -> Tuple[int, int]:
    def dominant(axis: int) -> int:
        return max(tile_halos[i][axis] for i in range(len(tile_halos)))

    overlap_t = _round_up(dominant(0) * pixel_scale.time, 8)
    halo_hw = max(dominant(1), dominant(2))
    overlap_hw = _round_up(halo_hw * pixel_scale.height, 32)
    return overlap_t, overlap_hw


def stage5_tokens_for_pixel_tile(
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    *,
    patch_size: int,
) -> int:
    h5 = max(1, tile_height // patch_size)
    w5 = max(1, tile_width // patch_size)
    return tile_frames * h5 * w5


def _axis_candidates(length: int, overlap: int, min_size: int, multiple: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    max_size = max(_round_up(length, multiple), min_size)
    for size in range(min_size, max_size + multiple, multiple):
        if size <= overlap:
            continue
        n = len(split_by_size(size, overlap)(length).intervals)
        out.append((size, n))
    return out


def volumetric_overlap_waste(
    *,
    num_frames: int,
    height: int,
    width: int,
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    n_t: int,
    n_h: int,
    n_w: int,
) -> float:
    processed = n_t * n_h * n_w * tile_frames * tile_height * tile_width
    unique = max(1, num_frames * height * width)
    return processed / unique


def _propagate_interval_through_upsample_hops(
    interval: DimensionInterval,
    strides: Sequence[int],
    causal: bool,
) -> DimensionInterval:
    x = interval
    for stride in strides:
        if stride < 1:
            raise ValueError(f"upsample stride must be >= 1, got {stride}")
        start = x.start * stride
        end = x.end * stride
        left_ramp = x.left_ramp * stride
        right_ramp = x.right_ramp * stride
        if causal and stride == 2:
            end -= 1
            if x.start != 0:
                start -= 1
        x = DimensionInterval(start=start, end=end, left_ramp=left_ramp, right_ramp=right_ramp)
    return x


"""Shared small layers for the diffusion-VAE NA transformer stack."""


class ChannelLinear(nn.Linear):
    @property
    def in_channels(self) -> int:
        return self.in_features

    @property
    def out_channels(self) -> int:
        return self.out_features


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, self.proj_out_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(
            x,
            "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2 and drop_leading_frame:
            x = x[:, 1:, :, :, :]
        return x


class AdaLNZero(nn.Module):
    NUM_CHUNKS: int = 7  # scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(F.silu(t_emb))
        chunks = h.chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


"""Shared absolute-RoPE math helpers (no consumer policy)."""


DEFAULT_ABS_ROPE_NUM_TILES = 4


def t_positions(t: int, device: torch.device) -> torch.Tensor:
    return torch.arange(t, dtype=torch.float32, device=device)


def h_positions(h: int, device: torch.device) -> torch.Tensor:
    return torch.arange(h, dtype=torch.float32, device=device)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be a multiple of 8 for default split"
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    assert d_t > 0
    assert d_hw > 0
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    inv_freqs = 1.0 / np.power(float(base), exponents)
    return torch.from_numpy(inv_freqs).to(torch.float32)


def rot_abs_axis_impl(
    xc: torch.Tensor,
    pos: torch.Tensor,
    inv: torch.Tensor,
    axis: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(compute_dtype)
    xo = pairs[..., 1].to(compute_dtype)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c = ang.cos().to(compute_dtype)
    s = ang.sin().to(compute_dtype)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out


"""Opaque full-volume abs-RoPE for deterministic (pre-diffusion) NA.
Owns QKV + opaque ``custom_op`` packaging for det ``NA.forward``. Det stages
have differently shaped T/H/W; the opaque op keeps Dynamo from specializing
on each stage shape. Diffusion paths use ``diff_attn/`` RoPE — not this file.
No T/H/W origin/offset is threaded for tiled decode, and none is needed:
every attention call here is ``natten.na3d``, a local window with no
cross-tile tokens. Absolute-vs-local RoPE differs by a global phase that
cancels inside the attention softmax over that window, so the attention
output is unchanged. Since every tiled-decode call processes exactly one
tile in isolation, using each tile's local 0-based positions is identical
to using its true absolute origin. Absolute origin still matters for
whether a tile contains the latent's first frame (``drop_leading_frame``),
which is handled outside RoPE in the decoder stage / per-tile decode.
"""


def _apply_opaque_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    positions_t = t_positions(t, x.device) if t_pos is None else t_pos
    xt = rot_abs_axis_impl(x[..., :d_t], positions_t, inv_t, axis=1, compute_dtype=compute_dtype)
    xh = rot_abs_axis_impl(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = rot_abs_axis_impl(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_opaque_tiled_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_opaque_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
                t_pos=t_pos,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


@torch.library.custom_op("diffsynth_ltx25::abs_rope", mutates_args=())
def _abs_rope_op(
    x: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_opaque_tiled_rope(
        x,
        (d_t, d_h, d_w),
        (inv_t, inv_h, inv_w),
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )


def _apply_opaque_abs_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float32 or bfloat16, got {compute_dtype}")
    d_t, d_h, d_w = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    return _abs_rope_op(
        x,
        inv_t,
        inv_h,
        inv_w,
        d_t,
        d_h,
        d_w,
        num_tiles,
        compute_dtype == torch.bfloat16,
    )


@torch.library.custom_op("diffsynth_ltx25::abs_rope_at_t", mutates_args=())
def _abs_rope_at_t_op(
    x: torch.Tensor,
    t_pos: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    d_t: int,
    d_h: int,
    d_w: int,
    num_tiles: int,
    compute_dtype_is_bf16: bool,
) -> torch.Tensor:
    compute_dtype = torch.bfloat16 if compute_dtype_is_bf16 else torch.float32
    return _apply_opaque_tiled_rope(
        x,
        (d_t, d_h, d_w),
        (inv_t, inv_h, inv_w),
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
        t_pos=t_pos,
    )


def _rope_config(attn: object, x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], int, torch.dtype]:
    inv_freqs = (
        attn.rope_inv_t.to(device=x.device),  # type: ignore[attr-defined]
        attn.rope_inv_h.to(device=x.device),  # type: ignore[attr-defined]
        attn.rope_inv_w.to(device=x.device),  # type: ignore[attr-defined]
    )
    num_tiles = getattr(attn, "rope_num_tiles", DEFAULT_ABS_ROPE_NUM_TILES)
    compute_dtype = getattr(attn, "rope_compute_dtype", torch.float32)
    return inv_freqs, num_tiles, compute_dtype


def _det_project_qkv(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = attn.project_qkv(x)  # type: ignore[attr-defined]
    q = attn.q_norm(q)  # type: ignore[attr-defined]
    k = attn.k_norm(k)  # type: ignore[attr-defined]
    q = q * attn.scale  # type: ignore[attr-defined]
    return q, k, v


def det_qkv_rope_at_times(
    attn: object,
    x: torch.Tensor,
    t_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if t_pos.ndim != 1 or t_pos.shape[0] != x.shape[1]:
        raise ValueError(f"t_pos must be ({x.shape[1]},) to match the plane axis, got {tuple(t_pos.shape)}")
    q, k, v = _det_project_qkv(attn, x)

    inv_freqs, num_tiles, compute_dtype = _rope_config(attn, x)
    d_t, d_h, d_w = attn.rope_dim_split  # type: ignore[attr-defined]
    positions = t_pos.to(device=x.device, dtype=torch.float32)
    rotated = [
        _abs_rope_at_t_op(
            tensor,
            positions,
            *inv_freqs,
            d_t,
            d_h,
            d_w,
            num_tiles,
            compute_dtype == torch.bfloat16,
        )
        for tensor in (q, k)
    ]
    return rotated[0], rotated[1], v


def det_qkv_rope(attn: object, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = _det_project_qkv(attn, x)
    inv_freqs, num_tiles, compute_dtype = _rope_config(attn, x)
    q = _apply_opaque_abs_rope(
        q,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )
    k = _apply_opaque_abs_rope(
        k,
        attn.rope_dim_split,  # type: ignore[attr-defined]
        inv_freqs,
        num_tiles=num_tiles,
        compute_dtype=compute_dtype,
    )
    return q, k, v


def vram_ready_linear(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    # This decoder calls several projections functionally, bypassing the VRAM wrappers'
    # forward, so ask the wrapper for computation-ready weights instead of reading them raw.
    computation = getattr(module, "computation", None)
    if computation is not None:
        return computation()
    return module.weight, module.bias


class QKVProjections(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        linear = nn.Linear(dim, dim * 3, bias=True)
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x):
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        weights = weight.chunk(3, dim=0)
        biases = bias.chunk(3, dim=0)
        return tuple(F.linear(x, weight, bias) for weight, bias in zip(weights, biases, strict=True))


DEFAULT_SWIGLU_TILE_SIZE: Final[int] = 16_384
DEFAULT_SWIGLU_TILES: Final[int] = 4


@dataclass(frozen=True)
class SwiGLUTileSpec:
    num_tiles: int | None = None
    tile_size: int | None = None

    def __post_init__(self) -> None:
        if (self.num_tiles is None) == (self.tile_size is None):
            raise ValueError("Provide exactly one of num_tiles or tile_size")
        if self.num_tiles is not None and self.num_tiles < 1:
            raise ValueError("num_tiles must be >= 1")
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")

DEFAULT_SWIGLU_TILE_SPEC = SwiGLUTileSpec(tile_size=DEFAULT_SWIGLU_TILE_SIZE)


def _swiglu_chunk(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def swiglu_tiled(x, w_gate, w_up, w_down, tile, *, use_triton=None):
    del use_triton
    if x.numel() == 0:
        return x
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    if tile.tile_size is not None:
        chunk_size = tile.tile_size
    else:
        chunk_size = max(1, math.ceil(flat.shape[0] / tile.num_tiles))
    output = torch.cat([_swiglu_chunk(chunk, w_gate, w_up, w_down) for chunk in flat.split(chunk_size)], dim=0)
    return output.reshape(*leading, output.shape[-1])


def swiglu_weights(mlp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        vram_ready_linear(mlp.w_gate)[0],
        vram_ready_linear(mlp.w_up)[0],
        vram_ready_linear(mlp.w_down)[0],
    )


def plain_mlp(x, mlp, norm, tile):
    y = norm(x)
    if y.numel() == 0:
        return x
    return x + swiglu_tiled(y, *swiglu_weights(mlp), tile)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, tile: SwiGLUTileSpec = DEFAULT_SWIGLU_TILE_SPEC) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.tile = tile

    def forward(self, x):
        return swiglu_tiled(x, *swiglu_weights(self), self.tile)


"""Limited-workspace 3D neighborhood attention (NATTEN ``na3d`` semantics) in pure torch.
Vendored from comfy-kitchen ``backends/eager/na.py`` (Apache-2.0) for DiffVAE hosts
without natten or Triton. Queries are tiled; tiles that share window geometry stack
into batched ``scaled_dot_product_attention`` calls with one additive mask per group.
"""


NA_SCORE_BUDGET = 2**25


NA_KV_STACK_BUDGET = 2**28


def _window_bounds(length: int, kernel: int, causal: bool) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    if causal:
        for i in range(length):
            starts.append(max(0, i - kernel + 1))
            ends.append(i + 1)
    else:
        kernel = min(kernel, length)
        lo = length - kernel
        half = kernel // 2
        for i in range(length):
            start = min(max(i - half, 0), lo)
            starts.append(start)
            ends.append(start + kernel)
    return starts, ends


def _pick_tiles(dims: tuple[int, int, int], kernels: list[int]) -> list[int]:
    tiles = list(dims)

    def cost(ts: list[int]) -> int:
        nq = math.prod(ts)
        nk = math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims, strict=True))
        return nq * nk

    while cost(tiles) > NA_SCORE_BUDGET and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _group_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bools = []
    for starts, ends in rel_bounds:
        st = torch.tensor(starts, device=device)
        en = torch.tensor(ends, device=device)
        kj = torch.arange(int(en.max()), device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < en[:, None]))
    visible = (
        bools[0][:, None, None, :, None, None]
        & bools[1][None, :, None, None, :, None]
        & bools[2][None, None, :, None, None, :]
    )
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int] | tuple[int, ...],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kernels = [k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, dims, strict=True)]
    if scale is None:
        scale = hd**-0.5
    device = q.device
    if scale != 1.0:
        q = q * scale

    bounds = [_window_bounds(d, k_, c) for d, k_, c in zip(dims, kernels, causal, strict=True)]
    tile_t, tile_h, tile_w = _pick_tiles(dims, [min(k_, d) for k_, d in zip(kernels, dims, strict=True)])

    groups: dict[
        tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...]],
        ],
        list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]],
    ] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (
            tuple(s - rt0 for s in bounds[0][0][t0:t1]),
            tuple(e - rt0 for e in bounds[0][1][t0:t1]),
        )
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (
                tuple(s - rh0 for s in bounds[1][0][h0:h1]),
                tuple(e - rh0 for e in bounds[1][1][h0:h1]),
            )
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(s - rw0 for s in bounds[2][0][w0:w1]),
                    tuple(e - rw0 for e in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                    )
                )

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, NA_KV_STACK_BUDGET // max(1, batch * nh * nk * hd * 2)) if device.type == "cuda" else 1
        qs0, _ = tiles[0]
        tq = qs0[0].stop - qs0[0].start
        th = qs0[1].stop - qs0[1].start
        tw = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = functional.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out


"""Pure-torch joint (video + keyframe) 3D neighborhood attention.
Computes, for one softmax per query:
* **video query** at ``(t, h, w)``: its local ``Kt x Kh x Kw`` video window, clamped to the
  volume and masked where it hangs over the edge, **plus** the whole ``Kh x Kw`` window at the
  same ``(h, w)`` on each of the ``num_slots`` nearest keyframe planes. Keyframe visibility does
  not depend on ``Kt`` -- a plane far outside the temporal radius is still visible.
* **keyframe query** on plane ``i``: the ``Kh x Kw`` window on its own plane (there is no
  plane-to-plane attention), plus the same window on each of the nearest video frames.
Which planes and frames are "nearest" comes from :func:`video_keyframe_slots` /
:func:`keyframe_video_slots`, so every backend agrees on visibility.
No Triton, no natten, no ``torch.compile``: this is the backend that always exists -- CPU,
macOS/MPS, Windows without a built extra.
Structure
---------
Everything is arranged so the arithmetic happens inside ``F.scaled_dot_product_attention``:
**Query bricks.** Queries are grouped into ``(bt, bh, bw)`` bricks and many bricks ride one SDPA
call as its batch dimension, so a frame costs a handful of launches rather than thousands. All
queries in a brick share one gathered key slab, of extent ``(bt + Kt - 1, bh + Kh - 1, bw + Kw - 1)``.
**Brick shape.** Wasted work is ``Nk / keys_actually_visible`` and *grows* with the brick, so the
spatial face stays small and square (square minimizes the slab at a fixed query count). Depth is
the exception: the gather is the larger cost and it scales with ``Nk / Nq``, which *falls* with
depth, so :data:`DEFAULT_BRICK_DEPTH` frames deep beats one frame deep despite doing more
arithmetic. Both defaults sit on measured plateaus at the production stage-5 shape.
**One shared, 2D-broadcast mask.** The visible-key pattern is a property of the brick geometry,
identical for every brick, so it is built once and passed as ``(1, 1, Nq, Nk)``. That shape is
load-bearing: torch keeps the memory-efficient backend and expands neither the mask nor the
scores, whereas a pre-expanded ``(G, NH, Nq, Nk)`` bias halves throughput and costs gigabytes.
**Per-key validity rides in the keys.** Out-of-volume positions, empty slots (``-1``) and invalid
planes are data-dependent, so folding them into the mask would make it per-brick. Instead ``K``
carries one extra channel holding ``0`` for a live key and :data:`_DEAD` for a dead one, against a
constant ``1`` channel on ``Q``. Q arrives pre-scaled, so with ``scale=1.0`` that adds exactly the
bias to the score. The channel count is then rounded up to :data:`_HEAD_DIM_ALIGN`.
**Head-major staging.** Key slabs are gathered from a ``(B, NH, A, Hp, Wp, C)`` copy rather than
from the caller's channels-last layout, so the gather's innermost contiguous run is ``ew * C``
instead of ``C``.
**Runs of constant slot row.** A brick spanning several frames shares one keyframe key slab, so it
must not straddle a change of visible planes. ``T`` is cut into maximal runs of identical slot rows
and bricks are tiled inside a run -- which also means one plane gather per run, entering the slab
view with a **zero** group stride.
Both loops are budgeted by :data:`DEFAULT_WORKSPACE_BYTES`: frames per staging pass, then
``(bricks, brick rows)`` per SDPA call. Peak transient memory is therefore bounded by that budget
and not by the volume, which is what lets this sit next to a decoder that has its own memory plan.
The gather is the floor: SDPA needs materialized ``(G, NH, Nk, HD)`` keys, so every key is copied
``Nk / Nq`` times. Only a fused neighborhood kernel avoids that.
"""


_DEAD = -1.0e4


_HEAD_DIM_ALIGN = 8


DEFAULT_BRICK_QUERIES = 64


DEFAULT_BRICK_DEPTH = 4


DEFAULT_WORKSPACE_BYTES = 256 * 1024**2


_STAGING_FACTOR_FUSED = 4.75


_STAGING_FACTOR_MATERIALIZED = 22.1


def sdpa_materializes_scores(device: torch.device) -> bool:
    return device.type != "cuda"


def staging_factor(device: torch.device) -> float:
    return _STAGING_FACTOR_MATERIALIZED if sdpa_materializes_scores(device) else _STAGING_FACTOR_FUSED


def _key_channels(head_dim: int) -> int:
    return -(-(head_dim + 1) // _HEAD_DIM_ALIGN) * _HEAD_DIM_ALIGN


def _window(kernel: int) -> tuple[int, int]:
    lo = kernel // 2
    return lo, kernel - lo - 1


def pick_brick(
    time: int,
    height: int,
    width: int,
    target: int = DEFAULT_BRICK_QUERIES,
    depth: int = DEFAULT_BRICK_DEPTH,
) -> tuple[int, int, int]:
    side = max(1, round(math.sqrt(target)))
    return min(depth, time), min(side, height), min(side, width)


class _Geometry:
    def __init__(
        self,
        height: int,
        width: int,
        kernel: tuple[int, int, int],
        brick: tuple[int, int, int],
    ) -> None:
        kernel_t, kernel_h, kernel_w = kernel
        lo_h, hi_h = _window(kernel_h)
        lo_w, hi_w = _window(kernel_w)
        self.height, self.width = height, width
        self.brick = brick
        self.kernel = kernel
        self.grid = (-(-height // brick[1]), -(-width // brick[2]))
        # Slab extents: T grows with the brick depth, H/W with the spatial face.
        self.span_t = brick[0] + kernel_t - 1
        self.span = (brick[1] + kernel_h - 1, brick[2] + kernel_w - 1)
        # Halo, plus enough to cover the last (partial) brick's slab.
        self.pad_h = (lo_h, hi_h + self.grid[0] * brick[1] - height)
        self.pad_w = (lo_w, hi_w + self.grid[1] * brick[2] - width)
        self.pad_t = _window(kernel_t)
        self.queries = brick[0] * brick[1] * brick[2]
        self.footprint = self.span[0] * self.span[1]
        self.padded_height = height + sum(self.pad_h)
        self.padded_width = width + sum(self.pad_w)

    def row_extent(self, rows: int) -> int:
        return (rows - 1) * self.brick[1] + self.span[0]


class _Schedule:
    def __init__(
        self,
        geometry: _Geometry,
        blocks: int,
        heads: int,
        head_dim: int,
        axis_bricks: int,
        element_size: int,
        workspace_bytes: int,
        factor: float,
    ) -> None:
        channels = _key_channels(head_dim)
        # One (brick along the axis, brick row) pair's worth of gathered keys and values, plus its
        # score block on backends that materialize one.
        keys = blocks * geometry.footprint
        pair_bytes = geometry.grid[1] * heads * keys * (channels + head_dim) * element_size
        # ``factor`` folds in whatever the selected SDPA kernel allocates on top of the staging,
        # chiefly a materialized score block. See :data:`_STAGING_FACTOR_FUSED`.
        pairs = max(1, int(workspace_bytes / max(pair_bytes * factor, 1.0)))
        if pairs >= geometry.grid[0]:
            self.group_axis = min(axis_bricks, max(1, pairs // geometry.grid[0]))
            self.group_rows = geometry.grid[0]
        else:
            self.group_axis = 1
            self.group_rows = pairs
        staged = geometry.padded_height * geometry.padded_width * heads * (channels + head_dim) * element_size
        per_axis_brick = staged * geometry.brick[0]
        self.stage_axis = min(axis_bricks, max(self.group_axis, workspace_bytes // max(per_axis_brick, 1)))


def _banded(queries: int, span: int, kernel: int, device: torch.device) -> torch.Tensor:
    key = torch.arange(span, device=device)[None, :]
    query = torch.arange(queries, device=device)[:, None]
    return (key >= query) & (key < query + kernel)


def _joint_mask(geometry: _Geometry, num_slots: int, device: torch.device) -> torch.Tensor:
    brick_t, brick_h, brick_w = geometry.brick
    kernel_t, kernel_h, kernel_w = geometry.kernel
    spatial = (
        _banded(brick_h, geometry.span[0], kernel_h, device)[:, None, :, None]
        & _banded(brick_w, geometry.span[1], kernel_w, device)[None, :, None, :]
    ).reshape(brick_h * brick_w, geometry.footprint)
    temporal = _banded(brick_t, geometry.span_t, kernel_t, device)
    video = (temporal[:, None, :, None] & spatial[None, :, None, :]).reshape(
        geometry.queries, geometry.span_t * geometry.footprint
    )
    planes = (
        spatial[None, :, None, :]
        .expand(brick_t, brick_h * brick_w, num_slots, geometry.footprint)
        .reshape(geometry.queries, num_slots * geometry.footprint)
    )
    return torch.cat([video, planes], dim=1)[None, None].contiguous()


def _stage(
    x: torch.Tensor,
    geometry: _Geometry,
    pad_t: tuple[int, int],
    *,
    with_bias_channel: bool,
) -> torch.Tensor:
    batch, axis, height, width, heads, head_dim = x.shape
    channels = _key_channels(head_dim) if with_bias_channel else head_dim
    out = x.new_zeros((batch, heads, axis + sum(pad_t), geometry.padded_height, geometry.padded_width, channels))
    if with_bias_channel:
        out[..., head_dim] = _DEAD
    live = out[
        :,
        :,
        pad_t[0] : pad_t[0] + axis,
        geometry.pad_h[0] : geometry.pad_h[0] + height,
        geometry.pad_w[0] : geometry.pad_w[0] + width,
    ]
    live[..., :head_dim] = x.permute(0, 4, 1, 2, 3, 5)
    if with_bias_channel:
        live[..., head_dim] = 0.0
    return out


def _slabs(
    staged: torch.Tensor,
    geometry: _Geometry,
    bricks: int,
    rows: int,
    blocks: int,
    *,
    group_stride: int,
) -> torch.Tensor:
    batch, heads = staged.shape[0], staged.shape[1]
    stride_b, stride_nh, stride_a, stride_h, stride_w, _ = staged.stride()
    return staged.as_strided(
        (batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, staged.shape[-1]),
        (
            stride_b,
            group_stride * stride_a,
            geometry.brick[1] * stride_h,
            geometry.brick[2] * stride_w,
            stride_nh,
            stride_a,
            stride_h,
            stride_w,
            1,
        ),
    )


def _query_bricks(x: torch.Tensor, geometry: _Geometry, bricks: int, rows: int) -> torch.Tensor:
    batch, axis, height, width, heads, head_dim = x.shape
    brick_t, brick_h, brick_w = geometry.brick
    pad_t, pad_h, pad_w = bricks * brick_t - axis, rows * brick_h - height, geometry.grid[1] * brick_w - width
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    bricked = (
        x.reshape(batch, bricks, brick_t, rows, brick_h, geometry.grid[1], brick_w, heads, head_dim)
        .permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        .reshape(batch * bricks * rows * geometry.grid[1], heads, geometry.queries, head_dim)
    )
    out = bricked.new_zeros((*bricked.shape[:-1], _key_channels(head_dim)))
    out[..., :head_dim] = bricked
    out[..., head_dim] = 1.0
    return out


def _unbrick(
    attended: torch.Tensor,
    geometry: _Geometry,
    batch: int,
    bricks: int,
    rows: int,
    extent: tuple[int, int],
) -> torch.Tensor:
    brick_t, brick_h, brick_w = geometry.brick
    heads, head_dim = attended.shape[1], attended.shape[3]
    plane = (
        attended.reshape(batch, bricks, rows, geometry.grid[1], heads, brick_t, brick_h, brick_w, head_dim)
        .permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
        .reshape(batch, bricks * brick_t, rows * brick_h, geometry.grid[1] * brick_w, heads, head_dim)
    )
    return plane[:, : extent[0], : extent[1], : geometry.width]


def _with_null(slots: torch.Tensor, null_index: int) -> torch.Tensor:
    return torch.where(slots < 0, torch.full_like(slots, null_index), slots)


def _append_null(keys: torch.Tensor, values: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (keys.shape[0], keys.shape[1], 1, *keys.shape[3:])
    null_key = keys.new_zeros(shape)
    null_key[..., head_dim] = _DEAD
    null_value = values.new_zeros((*shape[:-1], values.shape[-1]))
    return torch.cat([keys, null_key], dim=2), torch.cat([values, null_value], dim=2)


def _slot_runs(slots: torch.Tensor) -> list[tuple[int, int]]:
    rows = slots.tolist()
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index] != rows[start]:
            runs.append((start, index))
            start = index
    runs.append((start, len(rows)))
    return runs


def _attend_group(
    query_slice: torch.Tensor,
    key_views: tuple[torch.Tensor, ...],
    value_views: tuple[torch.Tensor, ...],
    geometry: _Geometry,
    shape: tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    bricks, rows = shape
    batch = query_slice.shape[0]
    heads, head_dim = query_slice.shape[4], query_slice.shape[5]
    blocks = sum(view.shape[5] for view in key_views)
    channels = _key_channels(head_dim)
    keys = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, channels))
    values = query_slice.new_empty((batch, bricks, rows, geometry.grid[1], heads, blocks, *geometry.span, head_dim))
    start = 0
    for key_view, value_view in zip(key_views, value_views, strict=True):
        stop = start + key_view.shape[5]
        keys[:, :, :, :, :, start:stop].copy_(key_view)
        values[:, :, :, :, :, start:stop].copy_(value_view)
        start = stop
    count = batch * bricks * rows * geometry.grid[1]
    attended = F.scaled_dot_product_attention(
        _query_bricks(query_slice, geometry, bricks, rows),
        keys.view(count, heads, blocks * geometry.footprint, channels),
        values.view(count, heads, blocks * geometry.footprint, head_dim),
        attn_mask=mask,
        scale=1.0,
    )
    return _unbrick(attended, geometry, batch, bricks, rows, (query_slice.shape[1], query_slice.shape[2]))


def _row_groups(geometry: _Geometry, schedule: _Schedule) -> list[tuple[int, int, slice, slice]]:
    brick_h = geometry.brick[1]
    groups = []
    for row in range(0, geometry.grid[0], schedule.group_rows):
        rows = min(schedule.group_rows, geometry.grid[0] - row)
        groups.append(
            (
                row,
                rows,
                slice(row * brick_h, row * brick_h + geometry.row_extent(rows)),
                slice(row * brick_h, min((row + rows) * brick_h, geometry.height)),
            )
        )
    return groups


def _video_query_pass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    slots: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    time, heads, head_dim = q.shape[1], q.shape[4], q.shape[5]
    brick_t = geometry.brick[0]
    lo_t, hi_t = geometry.pad_t
    num_slots = slots.shape[1]
    blocks = geometry.span_t + num_slots

    plane_keys, plane_values = _append_null(
        _stage(keyframe_k, geometry, (0, 0), with_bias_channel=True),
        _stage(keyframe_v, geometry, (0, 0), with_bias_channel=False),
        head_dim,
    )
    slot_table = _with_null(slots, keyframe_k.shape[1])
    mask = _joint_mask(geometry, num_slots, q.device)
    schedule = _Schedule(
        geometry,
        blocks,
        heads,
        head_dim,
        -(-time // brick_t),
        q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(geometry, schedule)

    out = torch.empty_like(q)
    for run_start, run_stop in _slot_runs(slot_table):
        # One plane gather per run: every brick inside it sees the same slots.
        planes = plane_keys.index_select(2, slot_table[run_start])
        plane_vals = plane_values.index_select(2, slot_table[run_start])
        run_bricks = -(-(run_stop - run_start) // brick_t)
        for staged_brick in range(0, run_bricks, schedule.stage_axis):
            staged_bricks = min(schedule.stage_axis, run_bricks - staged_brick)
            first = run_start + staged_brick * brick_t
            last = first + staged_bricks * brick_t  # exclusive; may reach past the run or T
            source = slice(max(0, first - lo_t), min(time, last + hi_t))
            pad_t = (max(0, lo_t - first), max(0, last + hi_t - time))
            window_keys = _stage(k[:, source], geometry, pad_t, with_bias_channel=True)
            window_values = _stage(v[:, source], geometry, pad_t, with_bias_channel=False)

            for brick in range(staged_brick, staged_brick + staged_bricks, schedule.group_axis):
                count = min(schedule.group_axis, staged_brick + staged_bricks - brick)
                start = run_start + brick * brick_t
                stop = min(start + count * brick_t, run_stop)
                offset = (brick - staged_brick) * brick_t
                for _, rows, key_rows, out_rows in rows_groups:
                    tile = _attend_group(
                        q[:, start:stop, out_rows],
                        (
                            _slabs(
                                window_keys[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(planes[:, :, :, key_rows], geometry, count, rows, num_slots, group_stride=0),
                        ),
                        (
                            _slabs(
                                window_values[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(plane_vals[:, :, :, key_rows], geometry, count, rows, num_slots, group_stride=0),
                        ),
                        geometry,
                        (count, rows),
                        mask,
                    )
                    out[:, start:stop, out_rows] = tile
    return out


def _keyframe_query_pass(
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    keyframe_valid: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    planes_total, heads, head_dim = keyframe_q.shape[1], keyframe_q.shape[4], keyframe_q.shape[5]
    num_slots = slots.shape[1]
    blocks = 1 + num_slots
    time = k.shape[1]
    flat = _Geometry(geometry.height, geometry.width, (1, *geometry.kernel[1:]), (1, *geometry.brick[1:]))

    # Only the frames some plane actually points at get staged -- at most ``P * num_slots`` of them,
    # against the whole volume if this staged ``k`` wholesale. ``unique`` doubles as the remap: slot
    # rows are rewritten to index the compacted stack.
    wanted, inverse = torch.unique(_with_null(slots, time).reshape(-1), return_inverse=True)
    frame_keys = _stage(k.index_select(1, wanted.clamp(max=time - 1)), flat, (0, 0), with_bias_channel=True)
    frame_values = _stage(v.index_select(1, wanted.clamp(max=time - 1)), flat, (0, 0), with_bias_channel=False)
    # An empty slot clamped onto a real frame above; kill it here instead of appending a null row.
    frame_keys[:, :, wanted == time, ..., head_dim] = _DEAD
    own_keys = _stage(keyframe_k, flat, (0, 0), with_bias_channel=True)
    own_values = _stage(keyframe_v, flat, (0, 0), with_bias_channel=False)
    own_keys[:, :, ~keyframe_valid, ..., head_dim] = _DEAD
    slot_table = inverse.reshape(planes_total, num_slots)
    mask = _joint_mask(flat, num_slots, keyframe_q.device)
    schedule = _Schedule(
        flat,
        blocks,
        heads,
        head_dim,
        planes_total,
        keyframe_q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(flat, schedule)

    out = torch.empty_like(keyframe_q)
    for start in range(0, planes_total, schedule.group_axis):
        stop = min(start + schedule.group_axis, planes_total)
        count = stop - start
        picked = slot_table[start:stop].reshape(-1)
        frames = frame_keys.index_select(2, picked)
        frame_vals = frame_values.index_select(2, picked)
        for _, rows, key_rows, out_rows in rows_groups:
            tile = _attend_group(
                keyframe_q[:, start:stop, out_rows],
                (
                    _slabs(own_keys[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frames[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                (
                    _slabs(own_values[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frame_vals[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                flat,
                (count, rows),
                mask,
            )
            out[:, start:stop, out_rows] = tile
    # An invalid plane sees nothing; zero it rather than shipping the uniform mean.
    return out * keyframe_valid[None, :, None, None, None, None]


def joint_na3d(  # noqa: PLR0913
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
    brick: tuple[int, int, int] | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    time, height, width = q.shape[1], q.shape[2], q.shape[3]
    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    geometry = _Geometry(height, width, kernel_size, brick if brick is not None else pick_brick(time, height, width))
    if factor is None:
        factor = staging_factor(q.device)
    return (
        _video_query_pass(q, k, v, keyframe_k, keyframe_v, video_slots, geometry, workspace_bytes, factor),
        _keyframe_query_pass(
            keyframe_q,
            keyframe_k,
            keyframe_v,
            k,
            v,
            keyframe_slots,
            keyframe_valid,
            geometry,
            workspace_bytes,
            factor,
        ),
    )


class EagerNAAttention:
    def __call__(self, attn, q, k, v):
        return na3d(q, k, v, kernel_size=attn.kernel_size, scale=1.0)


class EagerJointNAAttention:
    def __call__(self, attn, q, k, v, keyframe_q, keyframe_k, keyframe_v, keyframe_times, keyframe_valid):
        return joint_na3d(
            q, k, v, keyframe_q, keyframe_k, keyframe_v,
            keyframe_times, keyframe_valid, kernel_size=attn.kernel_size,
        )


"""3D Neighborhood Attention via NATTEN + absolute RoPE prelude.
Parameter shell shared by det ``NABlock`` and both diff-attn roles.
Diffusion AdaLN residuals live in pathway packages (each owns its RoPE);
det stages use ``det_attn_rope`` from :meth:`NeighborhoodAttention3D.forward`.
``attention_function`` selects the NA backend (NATTEN, Triton/eager fallback, or CuTe DSL).
"""


if TYPE_CHECKING:
    from ltx_core.model.video_vae.keyframes import KeyframeStream


try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    natten = None  # type: ignore[assignment]
    _NATTEN_AVAILABLE = False


class NAAttentionCallable(Protocol):
    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor: ...


class JointNAAttentionCallable(Protocol):
    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keyframe_q: torch.Tensor,
        keyframe_k: torch.Tensor,
        keyframe_v: torch.Tensor,
        keyframe_times: torch.Tensor,
        keyframe_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class NeighborhoodAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        if rope_dim_split is None:
            rope_dim_split = default_rope_dim_split(head_dim)
        assert sum(rope_dim_split) == head_dim, f"rope_dim_split={rope_dim_split} must sum to head_dim={head_dim}"
        self.rope_dim_split = rope_dim_split
        self.rope_base = rope_base
        self.rope_num_tiles = DEFAULT_ABS_ROPE_NUM_TILES
        self.rope_compute_dtype = torch.float32
        # Kept for the chunked opaque residual (string arg); callable is the swap surface.
        self.natten_backend: str | None = None
        self.attention_function: NAAttentionCallable = EagerNAAttention()
        # Separate slot, installed for every mode; never NATTEN/DSL. Only the keyframe
        # decode path reads it, so keyframe-less decode keeps NATTEN when it is installed.
        self.joint_attention_function: JointNAAttentionCallable | None = EagerJointNAAttention()

        self.register_buffer("rope_inv_t", rope_inv_freqs(rope_dim_split[0], rope_base), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(rope_dim_split[1], rope_base), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(rope_dim_split[2], rope_base), persistent=False)

        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        # W-chunking configuration (consumed by ``chunked.attn``).
        self.w_chunks = 1  # 1 = no chunking

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, t, h, w, _ = x.shape
        kt, kh, kw = self.kernel_size
        if t < kt or h < kh or w < kw:
            raise ValueError(
                f"3D neighborhood attention requires spatial dims >= kernel_size; "
                f"got (T,H,W)=({t},{h},{w}) vs kernel={self.kernel_size}"
            )

        q, k, v = det_qkv_rope(self, x)
        # natten's CUTLASS kernel silently produces wrong output if inputs are non-contiguous.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = self.attention_function(self, q, k, v)
        out = out.reshape(batch, t, h, w, self.dim)
        return self.proj(out)

    def forward_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        if self.joint_attention_function is None:
            raise RuntimeError(
                "keyframe decode needs joint_attention_function installed; build the decoder "
                "through apply_diffvae_config / apply_diffvae_mode"
            )
        batch, t, h, w, _ = x.shape
        planes = keyframes.x.shape[1]

        q, k, v = det_qkv_rope(self, x)
        keyframe_q, keyframe_k, keyframe_v = det_qkv_rope_at_times(self, keyframes.x, keyframes.times)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        keyframe_q = keyframe_q.contiguous()
        keyframe_k = keyframe_k.contiguous()
        keyframe_v = keyframe_v.contiguous()

        out, keyframe_out = self.joint_attention_function(
            self,
            q,
            k,
            v,
            keyframe_q,
            keyframe_k,
            keyframe_v,
            keyframes.times,
            keyframes.valid,
        )
        out = self.proj(out.reshape(batch, t, h, w, self.dim))
        keyframe_out = self.proj(keyframe_out.reshape(batch, planes, h, w, self.dim))
        return out, dataclasses.replace(keyframes, x=keyframe_out)


"""NABlock and DiffusionNABlock parameter shells for DiffVAE.
Pathway subclasses live in ``chunked/`` and ``combined/``; ``apply`` installs
them via ``__class__`` swap (same pattern as ``Fp8CastLinear``). The shell owns
weights + shared AdaLN helpers only — no pathway forward.
"""


if TYPE_CHECKING:
    from ltx_core.model.video_vae.keyframes import KeyframeStream


__all__ = [
    "DiffusionNABlock",
    "NABlock",
]


class NABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = plain_mlp(x, self.mlp, self.norm2, self.mlp.tile)
        return x

    def forward_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        attn_out, keyframe_attn = self.attn.forward_with_keyframes(
            self.norm1(x),
            dataclasses.replace(keyframes, x=self.norm1(keyframes.x)),
        )
        x = x + attn_out
        keyframe_x = keyframes.x + keyframe_attn.x
        x = plain_mlp(x, self.mlp, self.norm2, self.mlp.tile)
        keyframe_x = plain_mlp(keyframe_x, self.mlp, self.norm2, self.mlp.tile)
        return x, dataclasses.replace(keyframes, x=keyframe_x)


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)
        self.attn.proj.reset_parameters()

    def _modulation(
        self, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_table = self.scale_shift_table.to(dtype=modulation[0].dtype, device=modulation[0].device)
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        return scale_msa, shift_msa, scale_mlp, shift_mlp


"""Combined context residual: project context half of ``context_and_x`` into ``x``."""


def combined(
    context_and_x: torch.Tensor,
    w_proj: torch.Tensor,
    b_proj: torch.Tensor | None,
) -> torch.Tensor:
    context_channels = w_proj.shape[1]
    latent_context = context_and_x[..., :context_channels]
    x = context_and_x[..., context_channels:]
    return x + F.linear(latent_context, w_proj, b_proj)


inject_context = combined


"""Combined* diffusion AdaLN residual attention (full-volume NA + nested RoPE).
Owns nested full-volume abs-RoPE for the Combined / ``w_chunks==1`` path.
Does not share a residual body with ``chunked`` — only the NA module weights.
"""


_rot_abs_axis = torch.compiler.nested_compile_region(rot_abs_axis_impl)


def _apply_nested_abs_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    positions_t = t_positions(t, x.device) if t_pos is None else t_pos
    xt = _rot_abs_axis(x[..., :d_t], positions_t, inv_t, axis=1, compute_dtype=compute_dtype)
    xh = _rot_abs_axis(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = _rot_abs_axis(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_nested_full_volume_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
    t_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_nested_abs_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
                t_pos=t_pos,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


def _qkv_nested_rope(
    attn: NeighborhoodAttention3D,
    x: torch.Tensor,
    t_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = attn.project_qkv(x)
    q = attn.q_norm(q) * attn.scale
    k = attn.k_norm(k)
    inv_freqs = (
        attn.rope_inv_t.to(device=x.device),
        attn.rope_inv_h.to(device=x.device),
        attn.rope_inv_w.to(device=x.device),
    )
    positions = None if t_pos is None else t_pos.to(device=x.device, dtype=torch.float32)
    q = _apply_nested_full_volume_rope(
        q,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
        t_pos=positions,
    )
    k = _apply_nested_full_volume_rope(
        k,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
        t_pos=positions,
    )
    return q, k, v


def full_with_keyframes(
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn.joint_attention_function is None:
        raise RuntimeError(
            "keyframe decode needs joint_attention_function installed; build the decoder "
            "through apply_diffvae_config / apply_diffvae_mode"
        )
    batch, t, h, w, _ = x.shape
    planes = keyframe_x.shape[1]

    y = norm(x) * (1.0 + scale) + shift
    keyframe_y = norm(keyframe_x) * (1.0 + scale) + shift

    q, k, v = _qkv_nested_rope(attn, y)
    keyframe_q, keyframe_k, keyframe_v = _qkv_nested_rope(attn, keyframe_y, keyframe_times)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    keyframe_q = keyframe_q.contiguous()
    keyframe_k = keyframe_k.contiguous()
    keyframe_v = keyframe_v.contiguous()

    out, keyframe_out = attn.joint_attention_function(
        attn,
        q,
        k,
        v,
        keyframe_q,
        keyframe_k,
        keyframe_v,
        keyframe_times,
        keyframe_valid,
    )
    x = x + attn.proj(out.reshape(batch, t, h, w, attn.dim))
    keyframe_x = keyframe_x + attn.proj(keyframe_out.reshape(batch, planes, h, w, attn.dim))
    return x, keyframe_x


def full(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    y = norm(x) * (1.0 + scale) + shift
    batch, t, h, w, _ = y.shape
    kt, kh, kw = attn.kernel_size
    if t < kt or h < kh or w < kw:
        raise ValueError(
            f"3D neighborhood attention requires spatial dims >= kernel_size; "
            f"got (T,H,W)=({t},{h},{w}) vs kernel={attn.kernel_size}"
        )

    q, k, v = _qkv_nested_rope(attn, y)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = attn.attention_function(attn, q, k, v)
    out = out.reshape(batch, t, h, w, attn.dim)
    return x + attn.proj(out)


residual_attn = full
residual_attn_with_keyframes = full_with_keyframes


"""Combined pathway MLP: out-of-place AdaLN SwiGLU residual."""


def residual_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
    tile: SwiGLUTileSpec,
) -> torch.Tensor:
    y = modulate(norm(x), scale, shift)
    if y.numel() == 0:
        return x
    return x + swiglu_tiled(y, *swiglu_weights(mlp), tile)


"""CombinedDiffusionNABlock: context_and_x inject + full-volume attn + residual MLP."""


class CombinedDiffusionNABlock(DiffusionNABlock):
    def forward_combined_with_keyframes(
        self,
        context_and_x: torch.Tensor,
        keyframe_context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        keyframe_times: torch.Tensor,
        keyframe_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        w_proj, b_proj = vram_ready_linear(self.context_proj)
        x = inject_context(context_and_x, w_proj, b_proj)
        keyframe_x = inject_context(keyframe_context_and_x, w_proj, b_proj)
        x, keyframe_x = residual_attn_with_keyframes(
            x,
            keyframe_x,
            self.attn,
            self.norm1,
            scale_msa,
            shift_msa,
            keyframe_times,
            keyframe_valid,
        )
        x = residual_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        keyframe_x = residual_mlp(keyframe_x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x, keyframe_x * keyframe_valid[None, :, None, None, None]

    def forward_combined(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        w_proj, b_proj = vram_ready_linear(self.context_proj)
        x = inject_context(context_and_x, w_proj, b_proj)
        x = residual_attn(x, self.attn, self.norm1, scale_msa, shift_msa)
        x = residual_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x

    def forward(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.forward_combined(context_and_x, modulation)


"""Diffusion (NATTEN) video VAE decoder."""


logger: logging.Logger = logging.getLogger(__name__)


_L_STAGE_CHANNELS: Tuple[int, ...] = (1024, 512, 256, 256, 128)


_L_STAGE_DEPTHS: Tuple[int, ...] = (4, 6, 4, 2, 2)


_L_UPSAMPLES: Tuple[Tuple[Tuple[int, int, int], int], ...] = (
    ((1, 2, 2), 2),  # compress_space x2
    ((2, 1, 1), 2),  # compress_time x2
    ((2, 2, 2), 1),  # compress_all x1 (channel-preserving)
    ((2, 2, 2), 2),  # compress_all x2
)


_L_STAGE_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (3, 7, 7),
    (3, 7, 7),
    (3, 5, 5),
    (3, 5, 5),
    (3, 3, 3),
)


_DIFF_STAGE5_KERNEL_DEFAULT: Tuple[int, int, int] = (3, 7, 7)


_DIFF_STAGE5_DEPTH_DEFAULT: int = 8


_DIFF_STAGE_DEPTHS_DEFAULT: Tuple[int, ...] = (*_L_STAGE_DEPTHS[:-1], _DIFF_STAGE5_DEPTH_DEFAULT)


class DiffusionVideoDecoder(nn.Module, Disposable, VideoDecoder):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: Tuple[int, int, int] | None = None,
        stage_channels: Tuple[int, ...] = _L_STAGE_CHANNELS,
        stage_depths: Tuple[int, ...] = _DIFF_STAGE_DEPTHS_DEFAULT,
        stage_kernels: Tuple[Tuple[int, int, int], ...] = _L_STAGE_KERNELS,
        upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...] = _L_UPSAMPLES,
        stage5_kernel: Tuple[int, int, int] = _DIFF_STAGE5_KERNEL_DEFAULT,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 2,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: Literal["v", "x0"] = "v",
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_depths) == len(stage_kernels)
        assert len(upsamples) == len(stage_channels) - 1
        for c in stage_channels:
            assert c % head_dim == 0, f"stage_channels {stage_channels} must each be a multiple of head_dim={head_dim}"

        self.patch_size = patch_size
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps, device="cpu"),
            persistent=False,
        )
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.base_channels = stage_channels[-1]
        self.causal = False
        self.timestep_conditioning = True
        self.video_downscale_factors = SpatioTemporalScaleFactors.default()
        self.stage5_kernel: Tuple[int, int, int] = tuple(stage5_kernel)  # type: ignore[assignment]
        # NATTEN last-frame border workaround: replicate last latent frame
        # ``(K_t // 2) * 2`` times through stages 1-4, then crop the appendix
        # off context before stage 5 down to at least ``stage5_kernel[0]``.
        self._natten_trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        # Encoder output is per-channel normalized; undo before conv_in (same as ConvVideoDecoder).
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.conv_in = ChannelLinear(in_channels, stage_channels[0], bias=True)
        # Keyframe-stream tag, added to un-normalized keyframe latents before the shared
        # ``conv_in`` and nowhere else. It is the only keyframe-specific weight in the
        # whole feature. Checkpoints predating the keyframe training have no such key, so
        # ``video_decoder_sd_ops_for_checkpoint`` synthesizes zeros -- a missing key would
        # otherwise leave the parameter on the meta device under ``strict=False`` load.
        self.type_emb = nn.Parameter(torch.zeros(in_channels))

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        n_det_stages = len(stage_channels) - 1
        for stage_i in range(n_det_stages):
            c = stage_channels[stage_i]
            depth = stage_depths[stage_i]
            kernel = stage_kernels[stage_i]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        NABlock(dim=c, kernel_size=kernel, head_dim=head_dim, rope_dim_split=rope_dim_split)
                        for _ in range(depth)
                    ]
                )
            )
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(
                LinearPixelShuffleUpsample(in_channels=c, stride=stride, out_channels_reduction_factor=reduction)
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        c_ctx = stage_channels[-1]
        self.context_channels = c_ctx
        c5 = stage5_channels if stage5_channels is not None else c_ctx
        d5 = stage_depths[-1]
        assert c5 % head_dim == 0, f"stage5_channels {c5} must be a multiple of head_dim={head_dim}"
        noised_pixel_channels = out_channels * (patch_size**2)

        # Latent-grid floor so stages 1-3 (full volume) never undershoot NA.
        self.stage_min_tile_sizes: Tuple[int, int, int] = all_stages_min_tile_size(
            stage_kernels, upsamples, stage5_kernel
        )
        # Stage-4-input tile floor / overlap halos (only stages 4-5 are tiled).
        up3_stride = upsamples[3][0]
        self.tile_min_sizes: Tuple[int, int, int] = compute_tile_min_size(
            stage_kernels[3], stage5_kernel, up3_stride
        )
        self.tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = compute_tile_halos(
            stage_kernels[3],
            stage_depths[3],
            stage5_kernel,
            stage_depths[-1],
            up3_stride,
        )
        self.conv_in_x_t = ChannelLinear(noised_pixel_channels, c5, bias=True)

        # Shared AdaLN-Zero (7-chunk for shape compat; gate slots unused in block).
        self.shared_adaln = AdaLNZero(dim=c5, t_emb_dim=t_emb_dim)

        self.diff_blocks = nn.ModuleList(
            [
                CombinedDiffusionNABlock(
                    dim=c5,
                    kernel_size=stage5_kernel,
                    context_channels=c_ctx,
                    head_dim=head_dim,
                    rope_dim_split=rope_dim_split,
                )
                for _ in range(d5)
            ]
        )

        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = ChannelLinear(c5, noised_pixel_channels, bias=True)

        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        # Set True by ``compile_diffusion_decoder`` so decode marks T/H/W dynamic.
        self.mark_dynamic_shapes = False
        # When True, skip stage-4 upsample and inject via deferred sequential upsample+proj.
        # Default False = combined pathway (``CombinedDiffusionNABlock``). Chunked DiffVAE
        # modes flip this via ``apply_diffvae_config``.
        self.deferred_stage4_upsample = False
        # Remaining temporal upsampling per stage input, plus 1 for stage 5: the divisor in
        # ``keyframe_stage_times``. (8, 8, 4, 2, 1) for the production ladder.
        self._keyframe_time_strides: Tuple[int, ...] = remaining_time_strides(self.upsamples)

    def _run_det_stage(self, x: torch.Tensor, stage_i: int, drop_leading_frame: bool) -> torch.Tensor:
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[stage_i]:
            x = block(x)
        return self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(
        self,
        z_noisy: torch.Tensor,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        z_noisy = self.per_channel_statistics.un_normalize(z_noisy)
        x = z_noisy.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i in range(3):
            x = self._run_det_stage(x, stage_i, drop_leading_frame)
        return x

    def _keyframe_stream_from_latents(
        self,
        keyframes: DecodeKeyframes,
        *,
        valid: torch.Tensor | None = None,
    ) -> KeyframeStream:
        latents = self.per_channel_statistics.un_normalize(keyframes.latents)
        x = latents.permute(0, 2, 3, 4, 1)
        x = x + self.type_emb.to(dtype=x.dtype, device=x.device).view(1, 1, 1, 1, -1)
        x = self.conv_in(x)
        planes = x.shape[1]
        if valid is None:
            valid = torch.ones(planes, dtype=torch.bool, device=x.device)
        times = keyframe_clip_times(
            keyframes.pixel_frame_indices,
            self._keyframe_time_strides[0],
            keyframes.clip_start_frame,
        )
        return KeyframeStream(x=x, times=times.to(device=x.device), valid=valid.to(device=x.device)).masked()

    def _run_det_stage_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        stage_i: int,
        drop_leading_frame: bool,
        pixel_frame_indices: torch.Tensor,
        next_time_origin: float,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[stage_i]:
            x, keyframes = block.forward_with_keyframes(x, keyframes)
        x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        keyframe_x = upsample_keyframe_planes(self.upsamples[stage_i], keyframes.x)
        next_times = keyframe_clip_times(
            pixel_frame_indices,
            self._keyframe_time_strides[stage_i + 1],
            clip_start_frame,
            extra_origin=next_time_origin,
        )
        return x, KeyframeStream(
            x=keyframe_x,
            times=next_times.to(device=keyframe_x.device),
            valid=keyframes.valid,
        ).masked()

    def forward_stages_1_to_3_with_keyframes(
        self,
        z_noisy: torch.Tensor,
        keyframes: DecodeKeyframes,
        drop_leading_frame: bool = True,
        *,
        keyframe_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        keyframes.validate()
        if z_noisy.shape[-2:] != keyframes.latents.shape[-2:]:
            raise ValueError(
                f"keyframe latents must share the video latent's H/W (identical padding), got "
                f"{tuple(keyframes.latents.shape[-2:])} vs {tuple(z_noisy.shape[-2:])}"
            )
        stream = self._keyframe_stream_from_latents(keyframes, valid=keyframe_valid)
        x = self.per_channel_statistics.un_normalize(z_noisy).permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i in range(3):
            x, stream = self._run_det_stage_with_keyframes(
                x,
                stream,
                stage_i,
                drop_leading_frame,
                keyframes.pixel_frame_indices,
                0.0,
                clip_start_frame=keyframes.clip_start_frame,
            )
        return x, stream

    def forward_stage_4_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
        *,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        # Rebuild from global indices rather than trusting the caller's stream: stages 1-3 of a
        # full-clip decode are global, and Dist has already folded clip_start into clip times.
        keyframes = dataclasses.replace(
            keyframes,
            times=keyframe_clip_times(
                pixel_frame_indices,
                self._keyframe_time_strides[3],
                clip_start_frame,
                extra_origin=stage4_time_origin,
            ).to(device=keyframes.x.device),
        )
        if self.deferred_stage4_upsample:
            return self._forward_stage_4_deferred_with_keyframes(
                x, keyframes, pixel_frame_indices, pad_trailing, pixel_time_origin, clip_start_frame
            )
        x, keyframes = self._run_det_stage_with_keyframes(
            x,
            keyframes,
            3,
            drop_leading_frame,
            pixel_frame_indices,
            pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )
        if pad_trailing:
            x = crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time,
                stage5_kernel_t=self.stage5_kernel[0],
            )
        return x, keyframes

    def _forward_stage_4_deferred_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        pad_trailing: bool,
        pixel_time_origin: float,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[3]:
            x, keyframes = block.forward_with_keyframes(x, keyframes)
        if pad_trailing:
            up_t = int(self.upsamples[3].stride[0])
            x = crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time // up_t,
                stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
            )
        stage5_times = keyframe_clip_times(
            pixel_frame_indices,
            self._keyframe_time_strides[4],
            clip_start_frame,
            extra_origin=pixel_time_origin,
        )
        return x, KeyframeStream(
            x=keyframes.x,
            times=stage5_times.to(device=keyframes.x.device),
            valid=keyframes.valid,
        ).masked()

    def forward_stage_4(
        self,
        x: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        if self.deferred_stage4_upsample:
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.mark_dynamic(x, dim)
            for block in self.det_stages[3]:
                x = block(x)
            if pad_trailing:
                up_t = int(self.upsamples[3].stride[0])
                x = crop_trailing_context_natten_pad(
                    x,
                    n_latent_frames=self._natten_trailing_pad_latent_frames,
                    time_scale=self.video_downscale_factors.time // up_t,
                    stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
                )
            return x

        x = self._run_det_stage(x, 3, drop_leading_frame)
        if pad_trailing:
            x = crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time,
                stage5_kernel_t=self.stage5_kernel[0],
            )
        return x

    def _context_and_x_for_diff_step(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        noised_pixels_patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(noised_pixels_patched.permute(0, 2, 3, 4, 1))
        return torch.cat([context, x], dim=-1)

    def _keyframe_context_and_x_for_diff_step(
        self,
        keyframe_context: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        keyframe_valid: torch.Tensor,
    ) -> torch.Tensor:
        patched = patchify(keyframe_x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(patched.permute(0, 2, 3, 4, 1))
        x = x * keyframe_valid[None, :, None, None, None]
        return torch.cat([keyframe_context, x], dim=-1)

    def _x_for_diff_step(self, x_t: torch.Tensor) -> torch.Tensor:
        noised_pixels_patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        return self.conv_in_x_t(noised_pixels_patched.permute(0, 2, 3, 4, 1))

    def _keyframe_x_for_diff_step(self, keyframe_x_t: torch.Tensor, keyframe_valid: torch.Tensor) -> torch.Tensor:
        patched = patchify(keyframe_x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(patched.permute(0, 2, 3, 4, 1))
        return (x * keyframe_valid[None, :, None, None, None]).contiguous()

    def forward_diff_step(
        self,
        context_and_x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x_half = context_and_x[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(context_and_x, dim)

        for block in self.diff_blocks:
            x_half.copy_(block.forward_combined(context_and_x, modulation))
        return self._pixels_from_stage5(x_half)

    def forward_diff_step_with_keyframes(
        self,
        context_and_x: torch.Tensor,
        keyframe_context_and_x: torch.Tensor,
        t: torch.Tensor,
        keyframe_times: torch.Tensor,
        keyframe_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_half = context_and_x[..., self.context_channels :]
        keyframe_half = keyframe_context_and_x[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(context_and_x, dim)
            # Keyframe dim 1 is the plane count, not T, and keyframe_times / keyframe_valid pin it.
            # Marking it dynamic and then specializing to P raises ConstraintViolationError.
            for dim in (2, 3):
                torch._dynamo.mark_dynamic(keyframe_context_and_x, dim)

        for block in self.diff_blocks:
            x_out, keyframe_out = block.forward_combined_with_keyframes(
                context_and_x,
                keyframe_context_and_x,
                modulation,
                keyframe_times,
                keyframe_valid,
            )
            x_half.copy_(x_out)
            keyframe_half.copy_(keyframe_out)

        return self._pixels_from_stage5(x_half), self._pixels_from_stage5(keyframe_half)

    def _pixels_from_stage5(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward_diff_step_deferred(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        t: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
                torch._dynamo.mark_dynamic(stage4_feat, dim)

        from ltx_core.model.video_vae.transformer.dsl_kernels import DSLDiffusionBlockChain  # noqa: PLC0415

        if isinstance(self.diff_blocks, DSLDiffusionBlockChain):
            # Ping-pong fused launches; same deferred (x, stage4_feat) contract.
            x = self.diff_blocks(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)
        else:
            for block in self.diff_blocks:
                x = block.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)

        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward_diff_step_deferred_with_keyframes(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        keyframe_x: torch.Tensor,
        keyframe_stage4_feat: torch.Tensor,
        t: torch.Tensor,
        keyframe_times: torch.Tensor,
        keyframe_valid: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
                torch._dynamo.mark_dynamic(stage4_feat, dim)
            # Plane count, not T -- see the note in the combined path above.
            for dim in (2, 3):
                torch._dynamo.mark_dynamic(keyframe_x, dim)
                torch._dynamo.mark_dynamic(keyframe_stage4_feat, dim)

        # The DSL chain drives itself, so both streams recycle output buffers instead of
        # allocating a volume per block per stream; the chunked blocks expose the same
        # ``forward_x_ctx_with_keyframes``, so the fallback loop drives those.
        from ltx_core.model.video_vae.transformer.dsl_kernels import DSLDiffusionBlockChain  # noqa: PLC0415

        if isinstance(self.diff_blocks, DSLDiffusionBlockChain):
            x, keyframe_x = self.diff_blocks.forward_x_ctx_with_keyframes(
                x,
                stage4_feat,
                keyframe_x,
                keyframe_stage4_feat,
                modulation,
                keyframe_times,
                keyframe_valid,
                drop_leading_frame=drop_leading_frame,
            )
        else:
            for block in self.diff_blocks:
                x, keyframe_x = block.forward_x_ctx_with_keyframes(
                    x,
                    stage4_feat,
                    keyframe_x,
                    keyframe_stage4_feat,
                    modulation,
                    keyframe_times,
                    keyframe_valid,
                    drop_leading_frame=drop_leading_frame,
                )

        return self._pixels_from_stage5(x), self._pixels_from_stage5(keyframe_x)

    def _euler_step(
        self, x_t: torch.Tensor, model_out: torch.Tensor, t_now: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        compute_dtype = x_t.dtype
        dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
        x_t_fp32 = x_t.to(torch.float32)
        v_pred = model_out if self.model_output_type == "v" else to_velocity(x_t_fp32, t_now, model_out)
        return (x_t_fp32 - dt * v_pred).to(compute_dtype)

    def _decode_one_tile(
        self,
        feat_tile: torch.Tensor,
        x_t_tile_init: torch.Tensor,
        *,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool,
    ) -> torch.Tensor:
        context_tile = self.forward_stage_4(
            feat_tile,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
        )

        x_t = x_t_tile_init
        _, num_steps = timestep.shape
        for i in range(num_steps - 1):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1]
            if self.deferred_stage4_upsample:
                x = self._x_for_diff_step(x_t)
                model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin).to(
                    torch.float32
                )
            else:
                context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
                model_out = self.forward_diff_step(context_and_x, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)

        t_now = timestep[:, -1]
        if self.deferred_stage4_upsample:
            x = self._x_for_diff_step(x_t)
            model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin)
        else:
            context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
            model_out = self.forward_diff_step(context_and_x, t_now)
        if self.model_output_type == "x0":
            return model_out
        return self._euler_step(x_t, model_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    def _stage5_canvas_from_context(
        self,
        context_tile: torch.Tensor,
        *,
        drop_leading_frame: bool,
    ) -> tuple[int, int, int]:
        t, h, w = context_tile.shape[1], context_tile.shape[2], context_tile.shape[3]
        if self.deferred_stage4_upsample:
            # Context is still pre-upsample, so this is the same geometry as a stage-4
            # input. Ghost crop already ran, so do not re-apply the kernel-T floor.
            return stage5_pixel_shape_from_stage4(
                t,
                h,
                w,
                upsample_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=drop_leading_frame,
                pad_trailing=False,
            )
        return t, h * self.patch_size, w * self.patch_size

    def _decode_one_tile_with_keyframes(  # noqa: PLR0913
        self,
        feat_tile: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        *,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool,
        generator: torch.Generator | None,
        compute_dtype: torch.dtype,
        x_t_tile_init: torch.Tensor | None = None,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> torch.Tensor:
        context_tile, keyframes = self.forward_stage_4_with_keyframes(
            feat_tile,
            keyframes,
            pixel_frame_indices,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
            stage4_time_origin=stage4_time_origin,
            pixel_time_origin=pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )

        batch = context_tile.shape[0]
        canvas_t, canvas_h, canvas_w = self._stage5_canvas_from_context(context_tile, drop_leading_frame=is_origin)
        randn_device = generator.device if generator is not None else feat_tile.device

        def _noise(frames: int) -> torch.Tensor:
            return torch.randn(
                (batch, self.out_channels, frames, canvas_h, canvas_w),
                dtype=compute_dtype,
                generator=generator,
                device=randn_device,
            ).to(feat_tile.device)

        x_t = _noise(canvas_t) if x_t_tile_init is None else x_t_tile_init
        keyframe_x_t = _noise(keyframes.num_planes)

        def _step(t_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.deferred_stage4_upsample:
                return self.forward_diff_step_deferred_with_keyframes(
                    self._x_for_diff_step(x_t),
                    context_tile,
                    self._keyframe_x_for_diff_step(keyframe_x_t, keyframes.valid),
                    keyframes.x,
                    t_now,
                    keyframes.times,
                    keyframes.valid,
                    drop_leading_frame=is_origin,
                )
            context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
            keyframe_context_and_x = self._keyframe_context_and_x_for_diff_step(
                keyframes.x, keyframe_x_t, keyframes.valid
            )
            return self.forward_diff_step_with_keyframes(
                context_and_x, keyframe_context_and_x, t_now, keyframes.times, keyframes.valid
            )

        _, num_steps = timestep.shape
        for i in range(num_steps - 1):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1]
            video_out, keyframe_out = _step(t_now)
            x_t = self._euler_step(x_t, video_out.to(torch.float32), t_now, t_next)
            keyframe_x_t = self._euler_step(keyframe_x_t, keyframe_out.to(torch.float32), t_now, t_next)

        t_now = timestep[:, -1]
        video_out, _ = _step(t_now)
        if self.model_output_type == "x0":
            return video_out
        return self._euler_step(x_t, video_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    def _decode_temporal_group_isolated_with_keyframes(  # noqa: PLR0913
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        stream: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        content_s4_frames: int,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
        clip_start_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = torch.float16 if feat_s4.dtype == torch.bfloat16 else feat_s4.dtype
        buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = feat_s4.dtype
        up3_stride = tuple(self.upsamples[3].stride)

        for tile_index, tile in enumerate(tiles):
            feat_tile, is_origin, pad_trailing, content_thw = slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            # Two origins at two scales -- see forward_stage_4_with_keyframes.
            stage4_origin = tile.in_coords[1].indices(content_s4_frames)[0]
            pixel_lo, pixel_hi, _ = tile.out_coords[2].indices(full_torch_shape[2])

            # ``out_coords`` are local to this latent. Dist slices keep global indices and
            # ``clip_start_frame`` as the origin, so a local ``[0, 72)`` still has to select
            # global ``[56, 127]``.
            keep = planes_for_tile(pixel_frame_indices, pixel_lo, pixel_hi - 1, clip_start_frame=clip_start_frame)
            if not bool(keep.any()):
                raise RuntimeError(
                    f"tile covering pixel frames [{pixel_lo + clip_start_frame}, "
                    f"{pixel_hi - 1 + clip_start_frame}] selected no keyframe planes "
                    f"out of {int(pixel_frame_indices.shape[0])}; planes_for_tile always keeps at least one"
                )
            tile_stream = stream.select_planes(keep.to(stream.valid.device)).crop_spatial(
                tile.in_coords[2], tile.in_coords[3]
            )
            # Not debug-only: the decode below needs this tile's plane positions.
            tile_indices = pixel_frame_indices[keep.to(pixel_frame_indices.device)]
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "keyframe decode: tile %d/%d frames [%d, %d), stage-4 extent %dx%dx%d, %d of %d planes at %s",
                    tile_index + 1,
                    len(tiles),
                    pixel_lo,
                    pixel_hi,
                    feat_tile.shape[1],
                    feat_tile.shape[2],
                    feat_tile.shape[3],
                    tile_stream.num_planes,
                    int(pixel_frame_indices.shape[0]),
                    tile_indices.tolist(),
                )

            x_t_tile_init: torch.Tensor | None = None
            if x_t_init is not None:
                stage5_f, stage5_h, stage5_w = stage5_pixel_shape_from_stage4(
                    content_thw[0],
                    content_thw[1],
                    content_thw[2],
                    upsample_stride=up3_stride,  # type: ignore[arg-type]
                    patch_size=self.patch_size,
                    stage5_kernel_t=self.stage5_kernel[0],
                    drop_leading_frame=is_origin,
                    pad_trailing=pad_trailing,
                )
                # Same edge policy as the plain path: expand/crop the shared noise field
                # rather than drawing fresh noise, since NA mixes padded values inward.
                x_t_tile_init = x_t_init[tile.out_coords]
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 2, stage5_f, mode="repeat_last")
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 3, stage5_h, mode="symmetric")
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 4, stage5_w, mode="symmetric")

            pixel_tile = self._decode_one_tile_with_keyframes(
                feat_tile,
                tile_stream,
                tile_indices,
                is_origin=is_origin,
                timestep=timestep,
                pad_trailing=pad_trailing,
                generator=generator,
                compute_dtype=compute_dtype,
                x_t_tile_init=x_t_tile_init,
                stage4_time_origin=float(stage4_origin),
                pixel_time_origin=float(pixel_lo),
                clip_start_frame=clip_start_frame,
            )
            content_pixel_shape = pixel_tile_shape(full_torch_shape, tile.out_coords)
            pixel_tile = crop_pixels_to_content(
                pixel_tile,
                content_pixel_shape[2],
                content_pixel_shape[3],
                content_pixel_shape[4],
            ).to(buffer.dtype)

            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def _decode_groups_with_keyframes(  # noqa: PLR0913, PLR0915
        self,
        feat_s4: torch.Tensor,
        stream: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
        timestep: torch.Tensor,
        generator: torch.Generator | None,
        *,
        content_pixel: VideoLatentShape,
        h_pad: AxisPad | None,
        w_pad: AxisPad | None,
        as_fhwc: bool,
        clip_start_frame: int = 0,
    ) -> Iterator[torch.Tensor]:
        full_video_shape = (
            VideoLatentShape.from_torch_shape(latent.shape)
            .upscale(self.video_downscale_factors)
            ._replace(channels=self.out_channels)
        )
        target_shape = full_video_shape.to_torch_shape()
        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
            tile_halos=self.tile_halos,
        )
        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"
        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=latent.dtype, generator=generator, device=randn_device
            ).to(latent.device)

        logger.info(
            "keyframe decode: %d tile(s) in %d temporal group(s), %d frames at %dx%d, %d planes",
            len(tiles),
            len(groups),
            content_pixel.frames,
            content_pixel.height,
            content_pixel.width,
            int(pixel_frame_indices.shape[0]),
        )

        scaled_h_pad = scale_axis_pad(h_pad, self.video_downscale_factors.height)
        scaled_w_pad = scale_axis_pad(w_pad, self.video_downscale_factors.width)
        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None

        def _emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            if frames_keep < 1:
                return None
            chunk = buf[:, :, :frames_keep]
            if wts is not None:
                floor = _weight_floor(wts.dtype)
                chunk = chunk / wts[:, :, :frames_keep].clamp(min=floor)
            chunk = crop_pixels_to_content(
                chunk.to(latent.dtype),
                frames_keep,
                content_pixel.height,
                content_pixel.width,
                h_pad=scaled_h_pad,
                w_pad=scaled_w_pad,
            )
            return chunk[0].permute(1, 2, 3, 0).contiguous() if as_fhwc else chunk

        for group_index, group in enumerate(groups):
            curr_temporal_slice = group_slices[group_index]
            logger.info(
                "keyframe decode: group %d/%d, frames [%d, %d)",
                group_index + 1,
                len(groups),
                curr_temporal_slice.start,
                curr_temporal_slice.stop,
            )
            buffer, weights = self._decode_temporal_group_isolated_with_keyframes(
                group,
                feat_s4,
                stream,
                pixel_frame_indices,
                s4_t,
                x_t_init,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator,
                complementary=complementary,
                clip_start_frame=clip_start_frame,
            )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    overlap_stub += buffer[:, :, :overlap_len]
                    buffer[:, :, :overlap_len] = overlap_stub
                    if not complementary:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if group_index + 1 < len(groups):
                next_start = group_slices[group_index + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                # Retain only the trailing overlap for the next group's handoff.
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted

    def _decode_pixels_with_keyframes(
        self,
        latent: torch.Tensor,
        keyframes: DecodeKeyframes,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        as_fhwc: bool = False,
    ) -> Iterator[torch.Tensor]:
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        keyframes.validate(num_frames=content_pixel.frames)

        latent, (_t_pad, h_pad, w_pad) = ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        # Same spatial floor for the keyframe planes, with the plane axis pinned by a
        # temporal minimum of 1. The pad is symmetric, so padding only one stream would
        # offset every plane from the video by half of it.
        _min_t, min_h, min_w = self.stage_min_tile_sizes
        keyframe_latents, (_, keyframe_h_pad, keyframe_w_pad) = ensure_min_latent_shape(
            keyframes.latents, (1, min_h, min_w)
        )
        if (keyframe_h_pad, keyframe_w_pad) != (h_pad, w_pad):
            raise RuntimeError(
                f"keyframe spatial pad {(keyframe_h_pad, keyframe_w_pad)} != video pad {(h_pad, w_pad)}; "
                "the two streams must share one spatial origin"
            )
        padded_keyframes = dataclasses.replace(keyframes, latents=keyframe_latents)

        # Ghost pad is a temporal-border workaround for the video stream; keyframe planes have
        # no temporal extent to pad. The appendix is cropped off context before stage 5.
        latent_padded = pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        feat_s4, stream = self.forward_stages_1_to_3_with_keyframes(
            latent_padded, padded_keyframes, drop_leading_frame=True
        )

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        if tiling_config is not None:
            yield from self._decode_groups_with_keyframes(
                feat_s4,
                stream,
                keyframes.pixel_frame_indices,
                latent,
                tiling_config,
                timestep,
                generator,
                content_pixel=content_pixel,
                h_pad=h_pad,
                w_pad=w_pad,
                as_fhwc=as_fhwc,
                clip_start_frame=keyframes.clip_start_frame,
            )
            return

        logger.info("keyframe decode: untiled, %d frames, %d planes", content_pixel.frames, stream.num_planes)
        pixels = self._decode_one_tile_with_keyframes(
            feat_s4,
            stream,
            keyframes.pixel_frame_indices,
            is_origin=True,
            timestep=timestep,
            pad_trailing=True,
            generator=generator,
            compute_dtype=latent.dtype,
            clip_start_frame=keyframes.clip_start_frame,
        )
        pixels = crop_pixels_to_content(
            pixels,
            content_pixel.frames,
            content_pixel.height,
            content_pixel.width,
            h_pad=scale_axis_pad(h_pad, self.video_downscale_factors.height),
            w_pad=scale_axis_pad(w_pad, self.video_downscale_factors.width),
        ).to(latent.dtype)
        if as_fhwc:
            yield pixels[0].permute(1, 2, 3, 0).contiguous()
        else:
            yield pixels

    def _decode_video_with_keyframes(
        self,
        latent: torch.Tensor,
        keyframes: DecodeKeyframes,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            return frames.add_(1).mul_(0.5).clamp_(0, 1)

        for chunk in self._decode_pixels_with_keyframes(
            latent, keyframes, tiling_config, generator=generator, as_fhwc=True
        ):
            yield to_rgb(chunk)

    def _decode_temporal_group_isolated(
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        content_s4_frames: int,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = torch.float16 if feat_s4.dtype == torch.bfloat16 else feat_s4.dtype
        buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = feat_s4.dtype
        randn_device = generator.device if generator is not None else feat_s4.device
        up3_stride = tuple(self.upsamples[3].stride)

        for tile in tiles:
            feat_tile, is_origin, pad_trailing, content_thw = slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            content_pixel_shape = pixel_tile_shape(full_torch_shape, tile.out_coords)
            stage5_f, stage5_h, stage5_w = stage5_pixel_shape_from_stage4(
                content_thw[0],
                content_thw[1],
                content_thw[2],
                upsample_stride=up3_stride,  # type: ignore[arg-type]
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=is_origin,
                pad_trailing=pad_trailing,
            )

            if x_t_init is None:
                x_t_tile_init = torch.randn(
                    (content_pixel_shape[0], content_pixel_shape[1], stage5_f, stage5_h, stage5_w),
                    dtype=compute_dtype,
                    generator=generator,
                    device=randn_device,
                ).to(feat_s4.device)
            else:
                # Expand/crop to stage-5 canvas with the same edge policy as latent
                # size-floor / ghost pad (not fresh noise - NA mixes padded values
                # into kept pixels near the boundary).
                x_t_tile_init = x_t_init[tile.out_coords]
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 2, stage5_f, mode="repeat_last")
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 3, stage5_h, mode="symmetric")
                x_t_tile_init, _ = resize_axis(x_t_tile_init, 4, stage5_w, mode="symmetric")

            pixel_tile = self._decode_one_tile(
                feat_tile,
                x_t_tile_init,
                is_origin=is_origin,
                timestep=timestep,
                pad_trailing=pad_trailing,
            )
            pixel_tile = crop_pixels_to_content(
                pixel_tile,
                content_pixel_shape[2],
                content_pixel_shape[3],
                content_pixel_shape[4],
            ).to(buffer.dtype)

            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def _decode_pixels(  # noqa: PLR0912, PLR0915
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        as_fhwc: bool = False,
    ) -> Iterator[torch.Tensor]:
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)

        latent, (_t_pad, h_pad, w_pad) = ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        spatial_scale = (self.video_downscale_factors.height, self.video_downscale_factors.width)
        work_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_video_shape = work_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_video_shape.to_torch_shape()

        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
            tile_halos=self.tile_halos,
        )

        latent_padded = pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.mark_dynamic(latent_padded, dim)

        feat_s4 = self.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"

        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            compute_dtype = latent.dtype
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device
            ).to(latent.device)

        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        # Keep only the trailing temporal overlap of the previous group (not the full
        # chunk). Exclusive frames are yielded before the next group is decoded; the
        # consumer may still hold that emit while the next buffer is live (~2x tile).
        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None

        def _finalize(buf: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
            if complementary:
                return buf.to(latent.dtype)
            assert wts is not None
            wts = wts.clamp(min=_weight_floor(wts.dtype))
            return (buf / wts).to(latent.dtype)

        def _narrow_content_cfhw(t: torch.Tensor, frames_keep: int) -> torch.Tensor:
            x = t[:, :, :frames_keep]
            th, tw = content_pixel.height, content_pixel.width
            scale_h, scale_w = spatial_scale
            if h_pad is not None:
                before = scale_axis_pad(h_pad, scale_h).before
                x = x.narrow(3, before, th)
            else:
                need = x.shape[3] - th
                if need > 0:
                    x = x.narrow(3, need // 2, th)
                elif need < 0:
                    x, _ = resize_axis(x, 3, th, mode="symmetric")
            if w_pad is not None:
                before = scale_axis_pad(w_pad, scale_w).before
                x = x.narrow(4, before, tw)
            else:
                need = x.shape[4] - tw
                if need > 0:
                    x = x.narrow(4, need // 2, tw)
                elif need < 0:
                    x, _ = resize_axis(x, 4, tw, mode="symmetric")
            return x

        def _crop_emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            if frames_keep < 1:
                return None
            if not as_fhwc:
                chunk = _finalize(buf[:, :, :frames_keep], None if wts is None else wts[:, :, :frames_keep])
                return crop_pixels_to_content(
                    chunk,
                    frames_keep,
                    content_pixel.height,
                    content_pixel.width,
                    h_pad=h_pad,
                    w_pad=w_pad,
                    spatial_scale=spatial_scale,
                )

            # One materialize: contiguous FHWC in latent.dtype, still [-1, 1].
            # CFHW→FHWC cannot be inplace; range mapping is left to to_rgb.
            cfhw = _narrow_content_cfhw(buf, frames_keep)
            src = cfhw[0]  # C, F, H, W (view into accumulator)
            video = torch.empty(
                src.shape[1],
                src.shape[2],
                src.shape[3],
                src.shape[0],
                dtype=latent.dtype,
                device=src.device,
            )
            video.copy_(src.permute(1, 2, 3, 0))
            if not complementary:
                assert wts is not None
                w_cfhw = _narrow_content_cfhw(wts, frames_keep)
                wview = w_cfhw[0].permute(1, 2, 3, 0)
                # Inplace floor on exclusive weight region only (discarded after emit).
                wview.clamp_min_(_weight_floor(w_cfhw.dtype))
                video.div_(wview)
            return video

        for gi, group in enumerate(groups):
            curr_temporal_slice = group_slices[gi]
            buffer, weights = self._decode_temporal_group_isolated(
                group,
                feat_s4,
                s4_t,
                x_t_init,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator=generator,
                complementary=complementary,
            )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    # Stub is exactly the region overlapping this group (cloned when
                    # the previous group finished); blend then write back into buffer.
                    overlap_stub += buffer[:, :, :overlap_len]
                    if complementary:
                        buffer[:, :, :overlap_len] = overlap_stub
                    else:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        buffer[:, :, :overlap_len] = overlap_stub
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if gi + 1 < len(groups):
                next_start = group_slices[gi + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _crop_emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                # Retain only the trailing overlap for the next handoff.
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _crop_emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return next(self._decode_pixels(sample, tiling_config=None, generator=generator))

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: DecodeKeyframes | None = None,
    ) -> Iterator[torch.Tensor]:
        if keyframes is not None:
            yield from self._decode_video_with_keyframes(latent, keyframes, tiling_config, generator)
            return

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            return frames.add_(1).mul_(0.5).clamp_(0, 1)

        for chunk in self._decode_pixels(latent, tiling_config, generator=generator, as_fhwc=True):
            yield to_rgb(chunk)

class LTX25DiffusionVideoDecoder(DiffusionVideoDecoder):
    def forward(self, sample, generator=None, keyframes=None):
        return self.decode(sample, generator=generator, keyframes=keyframes)

    def auto_tiling_config(self, latent, keyframes=None):
        pixel_shape = (
            VideoLatentShape.from_torch_shape(latent.shape)
            .upscale(self.video_downscale_factors)
            ._replace(channels=self.out_channels)
        )
        device = latent.device
        if device.type == "cuda":
            # Cached allocator blocks from a previous decode would otherwise make the
            # free-memory query report a budget of zero for back-to-back decodes.
            torch.cuda.empty_cache()
            free_bytes = torch.cuda.mem_get_info(device.index)[0]
        else:
            free_bytes = 0
        if free_bytes <= 0:
            return None

        # Budget estimate must not read weight dtype/device: parameters may be meta
        # or disk-offloaded here, so assume bf16 storage for the footprint estimate.
        model_bytes = sum(parameter.numel() for parameter in self.parameters()) * 2
        upsample_strides = [tuple(upsample.stride) for upsample in self.upsamples]
        element_size = accumulator_element_size(latent.dtype)
        return recommended_decode_tiling_config(
            tile_halos=self.tile_halos,
            pixel_scale=stage4_to_pixel_scale_factors(upsample_strides[3], self.patch_size),
            min_tile_size_s4=self.tile_min_sizes,
            patch_size=self.patch_size,
            height=pixel_shape.height,
            width=pixel_shape.width,
            num_frames=pixel_shape.frames,
            mode=DiffVAEMode.CHUNKED_EAGER,
            free_bytes=free_bytes,
            stage5_channels=self.stage_channels[-1],
            stage4_channels=self.stage_channels[3],
            upsample_strides=upsample_strides,
            model_bytes=model_bytes,
            element_size=element_size,
            natten_trailing_pad_latent_frames=self._natten_trailing_pad_latent_frames,
            keyframes=keyframes is not None,
        )

    def decode(
        self,
        latent,
        tiled=False,
        seed=None,
        rand_device="cpu",
        keyframes=None,
        **kwargs,
    ):
        generator = torch.Generator(device=rand_device).manual_seed(seed) if seed is not None else None
        tiling_config = None
        if tiled:
            tiling_config = self.auto_tiling_config(latent, keyframes=keyframes)
            if tiling_config is None:
                raise ValueError("Automatic DiffVAE tiling requires a CUDA device with queryable free memory.")
        iterator = (
            self._decode_pixels_with_keyframes(latent, keyframes, tiling_config, generator=generator)
            if keyframes is not None
            else self._decode_pixels(latent, tiling_config, generator=generator)
        )
        chunks = list(iterator)
        if not chunks:
            raise RuntimeError("Diffusion decoder produced no output chunks")
        return torch.cat(chunks, dim=2)

