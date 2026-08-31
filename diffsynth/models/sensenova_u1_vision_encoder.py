import torch
import torch.nn as nn

from .sensenova_u1_common import build_abs_positions_from_grid_hw


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
