from __future__ import annotations

import torch
import torch.nn as nn

from ..core.gradient import gradient_checkpoint_forward
from ..core.vram.initialization import skip_model_initialization
from .minimax_h3_dit import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTBlock,
    MiniMaxH3Rope,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenRefiner,
    _modulate_scale_shift,
)

# MiniMax-H3-Image is the MiniMax-H3 video+audio DiT without its audio stream. One packed 1-D sequence holds the text
# rows, the rows of the reference images and the rows of the generated image, and a single stack of
# `MiniMaxH3DiTBlock`s runs over it. The block / attention / MLP / RoPE / timestep-embedding / AdaLN classes are the
# ones in `minimax_h3_dit.py`; only the top-level wiring and the per-step output heads differ.
#
# Release key names are kept verbatim, so the checkpoint's `state_dict` loads with no renames:
#   video_patch_proj, condition_proj, time_embedder.proj_in/proj_out, rope.inv_freq,
#   token_refiner.blocks.N.*, token_refiner.final_norm, blocks.N.*, final_layer.norm,
#   final_layer.adaln_proj.linear
# The checkpoint also carries `final_layer.video_out.{weight,bias}`, a single base output head that the release never
# uses: the per-step heads ship in the separate `pdd_fused_heads.safetensors` and replace it (see
# `MiniMaxH3ImagePDDHeads`). It is kept here anyway, unused, so that the state dict of this module is key-for-key and
# shape-for-shape the released one: the checkpoint loads with no converter, and a fine-tuned checkpoint this module
# saves has exactly the layout of the original release.
#
# Precision: the released checkpoint is mixed (float32 `video_patch_proj` / `time_embedder` / PDD heads, bf16
# everything else), but the module never assumes that split — every input is cast to the dtype of the projection it
# enters, so a uniformly fp32 or uniformly bf16 load works too. The one place where the *order* of the cast matters is
# the AdaLN activation; see `MiniMaxH3ImageAdalnProj`.


def _parameter_dtype(module: nn.Module) -> torch.dtype:
    # VRAM management replaces the projections with wrappers that compute in `computation_dtype`, which can differ from
    # the dtype the parameters were stored in (a disk-offloaded fp32 projection is still computed in bf16). Prefer the
    # wrapper's compute dtype so that inputs are aligned with the weights the matmul actually sees.
    computation_dtype = getattr(module, "computation_dtype", None)
    if computation_dtype is not None and not isinstance(computation_dtype, str):
        return computation_dtype
    # The dtype of the module's first parameter, i.e. what a projection in it was loaded as. The release is
    # mixed-precision (fp32 patch/time/output projections, bf16 blocks), so every input is aligned with the dtype of the
    # projection it enters instead of a single global dtype.
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


class MiniMaxH3ImageAdalnProj(MiniMaxH3AdalnProj):
    """`MiniMaxH3AdalnProj` that activates in the incoming dtype and rounds afterwards.

    The video+audio port activates the timestep embedding in whatever dtype `t_emb` arrives in. The release keeps
    `time_embedder` float32 while every AdaLN projection is bfloat16, and the reference rounds only the *result* of the
    activation (`self.linear(silu(temb).to(linear.weight.dtype))`): every block reads the same `temb`, so rounding it
    before the activation would bias all 50 blocks' modulation identically at every sampling step and accumulate over
    the 6-step trajectory. Keeping the activation in the timestep embedding's own precision and casting at the
    projection input is what makes the mixed-precision forward bit-identical to the reference; with a uniform dtype it
    is a no-op. Parameters, submodules and state-dict keys are the parent's (`adaln_proj.linear.{weight,bias}`).
    """

    def forward(self, t_emb):
        x = nn.functional.silu(t_emb)
        x = self.linear(x.to(_parameter_dtype(self.linear)))
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


def _to_fp32_activation_adaln(adaln_proj: MiniMaxH3AdalnProj) -> MiniMaxH3ImageAdalnProj:
    # Rebind in place: allocating a second 2688 -> 96768 projection per block just to discard it would cost more memory
    # than the whole checkpoint. `__class__` keeps the module tree, the parameters and the state-dict keys intact.
    adaln_proj.__class__ = MiniMaxH3ImageAdalnProj
    return adaln_proj


class MiniMaxH3ImageFinalLayer(nn.Module):
    """Final RMSNorm plus per-timestep AdaLN modulation, and the release's unused base output head.

    The output head is selected per denoising step and lives in `MiniMaxH3ImagePDDHeads` (a separate file in the
    release), so `forward` stops after `norm` + `adaln_proj` and the chosen head is applied by the caller.
    `video_out` is the release's single base head: the six per-step heads replace it, so it is never run, but keeping
    it makes this module's state dict identical to the released one.

    The modulation table holds one row per timestep (no modality axis), and the two halves of the projection are
    `shift` then `scale`, unlike diffusers' `AdaLayerNormContinuous`.
    """

    def __init__(self, hidden_size, time_embed_dim, final_adaln_out_features, latents_dim, patch_size, final_norm_eps):
        super().__init__()
        patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.adaln_proj = MiniMaxH3ImageAdalnProj(
            hidden_size, time_embed_dim, final_adaln_out_features, expand_ratio=2, modality_num=1
        )
        self.video_out = nn.Linear(hidden_size, patch_dim, bias=True)

    def forward(self, x, *, t_emb, timestep_indices):
        shift, scale = self.adaln_proj(t_emb)
        h = self.norm(x)
        return _modulate_scale_shift(h, shift, scale, timestep_indices)


class _MiniMaxH3ImagePDDHeadParameters(nn.Module):
    # Parameter container whose state-dict keys are exactly `video_out.weight` / `video_out.bias`, the layout of the
    # released `pdd_fused_heads.safetensors`. One `nn.Linear` per step would be `video_out.{0..5}.weight` instead, which
    # is not what the file holds.
    def __init__(self, num_heads: int, hidden_size: int, patch_dim: int):
        super().__init__()
        weight = torch.empty(num_heads, patch_dim, hidden_size, dtype=torch.float32)
        nn.init.normal_(weight, std=hidden_size**-0.5)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_heads, patch_dim, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, step_index: int) -> torch.Tensor:
        return nn.functional.linear(hidden_states, self.weight[step_index], self.bias[step_index])


class MiniMaxH3ImagePDDHeads(nn.Module):
    """The per-step output heads of MiniMax-H3-Image (`pdd_fused_heads.safetensors`).

    The release keeps one output head per denoising step instead of a single shared head: step `i` of the sampler
    projects the packed sequence with head `i`. All heads are stacked in one pair of float32 tensors, which is exactly
    the layout of the released file:

        video_out.weight  [num_heads, patch_dim, hidden_size]   float32
        video_out.bias    [num_heads, patch_dim]                float32

    The heads stay float32 even when the transformer is loaded in bf16 (the release stores them in float32 and the
    checkpoint's output projection is float32 too); `forward` expects the activation to be cast already.
    """

    def __init__(self, num_heads: int = 6, hidden_size: int = 5376, patch_dim: int = 96):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.patch_dim = patch_dim
        self.video_out = _MiniMaxH3ImagePDDHeadParameters(num_heads, hidden_size, patch_dim)

    def forward(self, hidden_states: torch.Tensor, step_index: int) -> torch.Tensor:
        # Run through the container module rather than reading its parameters: VRAM management replaces it with a
        # wrapper whose `forward` moves the weights onto the accelerator first.
        return self.video_out(hidden_states, step_index)

    @property
    def dtype(self) -> torch.dtype:
        # VRAM management wraps the container; ask the wrapper which dtype its projection computes in.
        return _parameter_dtype(self.video_out)


class MiniMaxH3ImageDiT(nn.Module):
    """MiniMax-H3-Image DiT (text-to-image and image editing), one denoising step's velocity prediction.

    The caller owns the packed layout: `x` holds the patchified image rows (reference rows first, then the generated
    image) and `prompt_embeds` the text rows. Every structural tensor (`position_ids`, `timestep_indices`,
    `token_tags`, `img_pos`, `text_pos`) describes how those rows are scattered into the single packed sequence
    `[text | reference images | generated image]`.

    `forward` returns the data-ward velocity of *all* image rows, in the row order of `img_pos`. Reference rows are
    included (they are simply never written by the sampler) and the sign is not flipped.

    The release's single base output head `final_layer.video_out` is kept, unused: the six per-step heads of
    `MiniMaxH3ImagePDDHeads` replace it, and holding the parameter keeps this module's state dict key-for-key and
    shape-for-shape identical to the released checkpoint, so both the released weights and any fine-tuned checkpoint
    of this module load with no conversion.
    """

    _repeated_blocks = ["MiniMaxH3DiTBlock", "MiniMaxH3TokenRefinerBlock"]

    def __init__(
        self,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        patch_size: tuple = (1, 2, 2),
        text_dim: int = 5120,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        final_adaln_out_features: int = 10752,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_channels_latents = latents_dim
        self.patch_size = tuple(patch_size)
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_embedder = MiniMaxH3TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim)
        self.rope = MiniMaxH3Rope(rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            token_refiner_num_layers, hidden_size, num_attention_heads, attention_head_dim,
            ffn_hidden_size, norm_eps, qk_norm_eps, final_norm_eps,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim,
                    adaln_out_features, norm_eps, qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        for block in self.blocks:
            _to_fp32_activation_adaln(block.adaln_proj)
        self.final_layer = MiniMaxH3ImageFinalLayer(
            hidden_size, time_embed_dim, final_adaln_out_features, latents_dim, self.patch_size, final_norm_eps
        )

    def _embed(self, *, x, prompt_embeds, seq_len, img_pos, text_pos, device):
        # The checkpoint is mixed-precision: `video_patch_proj` is float32 and `condition_proj` + the block stack are
        # bfloat16. Align every input with its own projection's dtype; the text stream then sets the dtype of the
        # packed sequence.
        image_embeds = self.video_patch_proj(x.to(_parameter_dtype(self.video_patch_proj)))
        text_embeds = self.condition_proj(prompt_embeds.to(_parameter_dtype(self.condition_proj)))
        text_embeds = self.token_refiner(
            text_embeds,
            cu_seqlens=torch.tensor([0, text_embeds.shape[0]], dtype=torch.int32, device=device),
            max_seqlen=text_embeds.shape[0],
        )

        embeddings = text_embeds.new_zeros((seq_len, self.hidden_size))
        embeddings.index_copy_(0, text_pos, text_embeds[: text_pos.shape[0]].to(embeddings.dtype))
        embeddings.index_copy_(0, img_pos, image_embeds[: img_pos.shape[0]].to(embeddings.dtype))
        return embeddings

    def forward(
        self,
        x,
        prompt_embeds,
        position_ids,
        unique_timesteps,
        timestep_indices,
        token_tags,
        img_pos,
        text_pos,
        pdd_heads: MiniMaxH3ImagePDDHeads | None = None,
        pdd_step_index: int = 0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Predict one denoising step's velocity for the image rows of the packed sequence.

        Args:
            x: `[N_img, patch_dim]` patchified latent rows, reference rows first, then the generated image.
            prompt_embeds: `[N_text, text_dim]` text conditioning rows.
            position_ids: `[1, S, 3]` float, the `(t, h, w)` rotary coordinate of every row of the packed sequence.
            unique_timesteps: `[T]` float32 in `[0, 1]`, the distinct noise levels present in the packed sequence.
            timestep_indices: `[S]` long, index into `unique_timesteps` of each row of the packed sequence.
            token_tags: `[S]` long, modality of each row: `0` image, `1` text.
            img_pos: `[N_img]` long, positions of the image rows in the packed sequence.
            text_pos: `[N_text]` long, positions of the text rows in the packed sequence.
            pdd_heads: `MiniMaxH3ImagePDDHeads`; falls back to `self.pdd_heads` when not given.
            pdd_step_index: the denoising step, which selects the output head.

        Returns:
            `[N_img, patch_dim]` data-ward velocity in the row order of `img_pos`.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be [N_img, patch_dim], got {list(x.shape)}")
        if position_ids.dim() != 3 or position_ids.shape[0] != 1 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must be [1, S, 3], got {list(position_ids.shape)}")
        if pdd_heads is None:
            pdd_heads = getattr(self, "pdd_heads", None)
        if pdd_heads is None:
            raise ValueError(
                "MiniMaxH3ImageDiT has no usable output head: pass `pdd_heads` (MiniMaxH3ImagePDDHeads). The "
                "`final_layer.video_out` parameter is kept only so the state dict matches the release; the model "
                "never runs it."
            )

        device = x.device
        seq_len = int(position_ids.shape[1])
        img_pos = img_pos.view(-1).to(device=device, dtype=torch.long)
        text_pos = text_pos.view(-1).to(device=device, dtype=torch.long)
        timestep_indices = timestep_indices.view(-1).to(device=device, dtype=torch.long)
        token_tags = token_tags.view(-1).to(device=device, dtype=torch.long)
        unique_timesteps = unique_timesteps.view(-1).to(device=device)
        if timestep_indices.shape[0] != seq_len or token_tags.shape[0] != seq_len:
            raise ValueError(
                "timestep_indices and token_tags must both be [S] matching position_ids, got "
                f"{timestep_indices.shape[0]} and {token_tags.shape[0]} for S={seq_len}."
            )

        rope_freqs = self.rope(position_ids.to(device))

        hidden = self._embed(
            x=x, prompt_embeds=prompt_embeds, seq_len=seq_len,
            img_pos=img_pos, text_pos=text_pos, device=device,
        )

        # One timestep embedding per distinct noise level, at the time embedder's own (float32) precision. The AdaLN
        # projections round to their own dtype themselves, after their activation (see `MiniMaxH3ImageAdalnProj`), so
        # this one value serves every block whether the model is loaded in fp32, bf16 or the release's mixed
        # precision, and the activation of the shared embedding is never rounded first.
        t_emb = self.time_embedder(unique_timesteps, dtype=_parameter_dtype(self.time_embedder.proj_in))

        # Row -> AdaLN table row. The table has one row per (timestep, modality) pair, three modalities per timestep
        # (0 image, 1 text, 2 audio, unused here).
        combined_indices = (timestep_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        for block in self.blocks:
            hidden = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
            )

        hidden = self.final_layer(hidden, t_emb=t_emb, timestep_indices=timestep_indices)

        # The step's head runs over every row of the packed sequence, then the image rows are gathered. The heads are
        # float32 while the block stack follows the checkpoint dtype, so align the activation with them.
        output = pdd_heads(hidden.to(pdd_heads.dtype), pdd_step_index)
        return output.index_select(0, img_pos)


# The distilled schedule visits six sigma-grid timesteps plus the reference-image level; one AdaLN table row each.
MINIMAX_H3_IMAGE_ADALN_TABLE_ROWS = 7


class MiniMaxH3ImageAdalnTable(nn.Module):
    """Table of precomputed AdaLN outputs, one row per distilled timestep.

    `forward` takes the table row of every distinct timestep instead of a timestep embedding: row `i` holds the
    `expand_ratio` chunks of the distilled schedule's step `i`, laid out as `MiniMaxH3AdalnProj` lays out its
    projection output. The last row belongs to the reference-image level.
    """

    def __init__(self, hidden_size, modality_num, expand_ratio, num_timesteps):
        super().__init__()
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = hidden_size
        # A parameter, not a buffer: disk offload only restores parameters of the modules in the VRAM map.
        self.table = nn.Parameter(
            torch.empty(num_timesteps, modality_num, expand_ratio, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, timestep_rows):
        x = self.table.index_select(0, timestep_rows).reshape(-1, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


def _to_adaln_table(proj: MiniMaxH3AdalnProj) -> MiniMaxH3ImageAdalnTable:
    return MiniMaxH3ImageAdalnTable(
        proj.hidden_size, proj.modality_num, proj.expand_ratio, MINIMAX_H3_IMAGE_ADALN_TABLE_ROWS
    )


class MiniMaxH3ImageDiTPruned(MiniMaxH3ImageDiT):
    """`MiniMaxH3ImageDiT` with every AdaLN projection replaced by a lookup table.

    The tables hold the AdaLN outputs of the distilled schedule's timesteps (six grid steps plus the reference-image
    level), so the `2688 -> 96768` projections and the timestep embedder are gone. Step `i` reads row `i` and the
    reference rows read the last row, which makes the model inference-only on the distilled six-step schedule.
    """

    def __init__(self, **kwargs):
        with skip_model_initialization():
            super().__init__(**kwargs)
            for block in self.blocks:
                block.adaln_proj = _to_adaln_table(block.adaln_proj)
            self.final_layer.adaln_proj = _to_adaln_table(self.final_layer.adaln_proj)
            del self.final_layer.video_out
            self.time_embedder = None

    def forward(
        self,
        x,
        prompt_embeds,
        position_ids,
        unique_timesteps,
        timestep_indices,
        token_tags,
        img_pos,
        text_pos,
        pdd_heads: MiniMaxH3ImagePDDHeads | None = None,
        pdd_step_index: int = 0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Same contract as `MiniMaxH3ImageDiT.forward`, with the AdaLN tables read by step index."""
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            raise ValueError("MiniMaxH3ImageDiTPruned is inference-only and does not support gradient checkpointing.")
        if x.dim() != 2:
            raise ValueError(f"x must be [N_img, patch_dim], got {list(x.shape)}")
        if pdd_heads is None:
            pdd_heads = getattr(self, "pdd_heads", None)
        if pdd_heads is None:
            raise ValueError("MiniMaxH3ImageDiTPruned has no usable output head: pass `pdd_heads`.")

        device = x.device
        seq_len = int(position_ids.shape[1])
        img_pos = img_pos.view(-1).to(device=device, dtype=torch.long)
        text_pos = text_pos.view(-1).to(device=device, dtype=torch.long)
        timestep_indices = timestep_indices.view(-1).to(device=device, dtype=torch.long)
        token_tags = token_tags.view(-1).to(device=device, dtype=torch.long)

        rope_freqs = self.rope(position_ids.to(device))
        hidden = self._embed(
            x=x, prompt_embeds=prompt_embeds, seq_len=seq_len,
            img_pos=img_pos, text_pos=text_pos, device=device,
        )

        # Where the released model embeds the timesteps, the pruned one picks table rows: the step's own row, and the
        # last row for the reference-image level.
        timestep_rows = torch.full(
            (unique_timesteps.shape[0],), int(pdd_step_index), dtype=torch.long, device=device
        )
        timestep_rows[1:] = MINIMAX_H3_IMAGE_ADALN_TABLE_ROWS - 1

        combined_indices = (timestep_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        for block in self.blocks:
            hidden = block(
                hidden, t_emb=timestep_rows, combined_indices=combined_indices,
                rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=seq_len,
            )

        hidden = self.final_layer(hidden, t_emb=timestep_rows, timestep_indices=timestep_indices)

        output = pdd_heads(hidden.to(pdd_heads.dtype), pdd_step_index)
        return output.index_select(0, img_pos)
