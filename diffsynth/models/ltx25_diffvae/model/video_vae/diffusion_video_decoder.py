from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from diffsynth.models.ltx25_diffvae.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffsynth.models.ltx25_diffvae.model.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer import (
    AdaLNZero,
    ChannelLinear,
    CombinedDiffusionNABlock,
    LinearPixelShuffleUpsample,
    NABlock,
)


class DiffusionVideoDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        stage_channels: tuple[int, ...] = (1024, 512, 256, 256, 128),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (3, 7, 7)),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
        stage5_kernel: tuple[int, int, int] | None = None,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 1,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: Literal["v", "x0"] = "x0",
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(stage_depths) or len(stage_channels) != len(stage_kernels):
            raise ValueError("stage_channels, stage_depths, and stage_kernels must have the same length")
        if len(upsamples) != len(stage_channels) - 1:
            raise ValueError("one fewer upsample than decoder stages is required")
        if any(channels % head_dim for channels in stage_channels):
            raise ValueError("every stage channel count must be divisible by head_dim")

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.stage_kernels = stage_kernels
        self.upsample_strides = tuple(stride for stride, _ in upsamples)
        self.stage5_kernel = tuple(stage5_kernel or stage_kernels[-1])
        self.model_output_type = model_output_type
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps),
            persistent=False,
        )

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.conv_in = ChannelLinear(in_channels, stage_channels[0], bias=True)
        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for channels, depth, kernel, (stride, reduction) in zip(
            stage_channels[:-1], stage_depths[:-1], stage_kernels[:-1], upsamples, strict=True
        ):
            self.det_stages.append(
                nn.ModuleList(
                    [NABlock(channels, kernel, head_dim=head_dim, rope_dim_split=rope_dim_split) for _ in range(depth)]
                )
            )
            self.upsamples.append(LinearPixelShuffleUpsample(channels, stride, reduction))

        context_channels = stage_channels[-1]
        diffusion_channels = stage5_channels or context_channels
        if diffusion_channels % head_dim:
            raise ValueError("stage5_channels must be divisible by head_dim")
        self.context_channels = context_channels
        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)
        self.conv_in_x_t = ChannelLinear(out_channels * patch_size**2, diffusion_channels, bias=True)
        self.shared_adaln = AdaLNZero(dim=diffusion_channels, t_emb_dim=t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                CombinedDiffusionNABlock(
                    diffusion_channels,
                    self.stage5_kernel,
                    context_channels,
                    head_dim=head_dim,
                    rope_dim_split=rope_dim_split,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(diffusion_channels, eps=1e-6)
        self.conv_out = ChannelLinear(diffusion_channels, out_channels * patch_size**2, bias=True)

        self.min_latent_shape = self._minimum_latent_shape()

        self._trailing_latent_frames = (stage_kernels[0][0] // 2) * 2

    def _minimum_latent_shape(self) -> tuple[int, int, int]:
        cumulative = [1, 1, 1]
        minimum = [1, 1, 1]
        for kernel, stride in zip(self.stage_kernels[:-1], self.upsample_strides, strict=True):
            for axis in range(3):
                minimum[axis] = max(minimum[axis], -(-kernel[axis] // cumulative[axis]))
                cumulative[axis] *= stride[axis]
        for axis in range(3):
            minimum[axis] = max(minimum[axis], -(-self.stage5_kernel[axis] // cumulative[axis]))
        return tuple(minimum)

    @staticmethod
    def _pad_axis(x: torch.Tensor, axis: int, size: int, *, trailing_only: bool) -> tuple[torch.Tensor, int]:
        current = x.shape[axis]
        if current >= size:
            return x, 0
        missing = size - current
        before = 0 if trailing_only else missing // 2
        after = missing - before
        pieces = []
        if before:
            pieces.append(x.narrow(axis, 0, 1).expand(*x.shape[:axis], before, *x.shape[axis + 1 :]))
        pieces.append(x)
        if after:
            pieces.append(x.narrow(axis, current - 1, 1).expand(*x.shape[:axis], after, *x.shape[axis + 1 :]))
        return torch.cat(pieces, dim=axis), before

    def _pad_to_minimum(self, latent: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        latent, _ = self._pad_axis(latent, 2, self.min_latent_shape[0], trailing_only=True)
        latent, h_before = self._pad_axis(latent, 3, self.min_latent_shape[1], trailing_only=False)
        latent, w_before = self._pad_axis(latent, 4, self.min_latent_shape[2], trailing_only=False)
        return latent, h_before, w_before

    def _run_det_stage(self, x: torch.Tensor, stage_index: int, drop_leading_frame: bool) -> torch.Tensor:
        for block in self.det_stages[stage_index]:
            x = block(x)
        return self.upsamples[stage_index](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(self, z_noisy: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        x = self.per_channel_statistics.un_normalize(z_noisy).permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_index in range(3):
            x = self._run_det_stage(x, stage_index, drop_leading_frame)
        return x

    def forward_stage_4(
        self,
        x: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        x = self._run_det_stage(x, 3, drop_leading_frame)
        if pad_trailing and self._trailing_latent_frames:
            ghost_frames = self._trailing_latent_frames * 8
            keep = min(x.shape[1], max(x.shape[1] - ghost_frames, self.stage5_kernel[0]))
            x = x[:, :keep]
        return x

    def _context_and_x_for_diff_step(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        pixels = patchify(x_t, patch_size_hw=self.patch_size).permute(0, 2, 3, 4, 1)
        return torch.cat([context, self.conv_in_x_t(pixels)], dim=-1)

    def forward_diff_step(self, context_and_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = context_and_x[..., self.context_channels :]
        modulation = self.shared_adaln(self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype))
        for block in self.diff_blocks:
            x = block(context_and_x, modulation)
            context_and_x[..., self.context_channels :].copy_(x)
        x = self.conv_out(self.norm_out(x)).permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size)

    def _euler_step(self, x_t: torch.Tensor, model_out: torch.Tensor, t_now: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
        if self.model_output_type == "x0":
            model_out = (x_t.to(torch.float32) - model_out.to(torch.float32)) / t_now.view(
                -1, *([1] * (x_t.ndim - 1))
            ).to(torch.float32)
        return (x_t.to(torch.float32) - dt * model_out.to(torch.float32)).to(x_t.dtype)

    def forward(self, sample: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        frames = (sample.shape[2] - 1) * 8 + 1
        height, width = sample.shape[3] * 32, sample.shape[4] * 32
        latent, h_before, w_before = self._pad_to_minimum(sample)
        trailing = latent[:, :, -1:].expand(-1, -1, self._trailing_latent_frames, -1, -1)
        context = self.forward_stages_1_to_3(torch.cat([latent, trailing], dim=2))
        context = self.forward_stage_4(context)

        pixel_shape = (sample.shape[0], self.out_channels, context.shape[1], context.shape[2] * self.patch_size, context.shape[3] * self.patch_size)
        x_t = torch.randn(pixel_shape, dtype=sample.dtype, device=sample.device, generator=generator)
        timesteps = self.default_inference_timesteps.to(sample.device).expand(sample.shape[0], -1)
        for index in range(timesteps.shape[1] - 1):
            prediction = self.forward_diff_step(self._context_and_x_for_diff_step(context, x_t), timesteps[:, index])
            x_t = self._euler_step(x_t, prediction, timesteps[:, index], timesteps[:, index + 1])
        prediction = self.forward_diff_step(self._context_and_x_for_diff_step(context, x_t), timesteps[:, -1])
        pixels = prediction if self.model_output_type == "x0" else self._euler_step(x_t, prediction, timesteps[:, -1], torch.zeros_like(timesteps[:, -1]))
        return pixels[:, :, :frames, h_before * 32 : h_before * 32 + height, w_before * 32 : w_before * 32 + width].contiguous()

    def decode(
        self,
        latent: torch.Tensor,
        tiled: bool = True,
        tile_size_in_pixels: int = 512,
        tile_overlap_in_pixels: int = 128,
        tile_size_in_frames: int = 128,
        tile_overlap_in_frames: int = 24,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del tiled, tile_size_in_pixels, tile_overlap_in_pixels, tile_size_in_frames, tile_overlap_in_frames
        return self.forward(latent, generator=generator)
