"""
ActionConditionedDiT
====================
Wraps WanModel or CogDiT with configurable multi-stream action conditioning
for ablation experiments.

Conditioning streams (all independently switchable via config):
    obs_image     : always input_concat  (fixed spatial anchor, not ablated)
    action        : cross_attn | adaln
    masked_images : input_concat | cross_attn
    history       : input_concat | cross_attn

Expected tensor shapes per injection type
-----------------------------------------
    input_concat  ->  (B, C * n_frames, T, H, W)   concatenated along channel dim
    cross_attn    ->  (B, C, T_seq, H, W)           flattened to spatial tokens
    adaln         ->  action only; encoder output is mean-pooled to (B, backbone_dim)

Note on cross_attn for spatial latents
---------------------------------------
Flattening (T, H, W) into tokens can be expensive for large spatial dims.
Consider applying average pooling or a small Perceiver before the projection
if the sequence length becomes prohibitive.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ActionConditioningConfig, ConditionStreamConfig
from .encoders import PerceiverActionEncoder, MLPActionEncoder, ActionEncoder
from .injectors import CrossAttnInjector, InputConcatInjector, AdaLNInjector


def _build_action_encoder(
    cfg: ConditionStreamConfig,
    action_dim: int,
    perceiver_kwargs: dict,
) -> ActionEncoder:
    if cfg.encoder_type == "perceiver":
        return PerceiverActionEncoder(
            action_dim=action_dim,
            embed_dim=cfg.embed_dim,
            **perceiver_kwargs,
        )
    if cfg.encoder_type == "mlp":
        return MLPActionEncoder(action_dim=action_dim, embed_dim=cfg.embed_dim)
    raise ValueError(f"Unsupported encoder_type for action stream: {cfg.encoder_type!r}")


class ActionConditionedDiT(nn.Module):
    """
    Backbone-agnostic wrapper for action-conditioned video generation.

    Swap injection strategies via config — no code changes needed.

    Parameters
    ----------
    backbone : WanModel or CogDiT instance (pre-built, weights already loaded)
    config   : ActionConditioningConfig
    """

    def __init__(self, backbone: nn.Module, config: ActionConditioningConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

        perceiver_kw = dict(
            num_queries=config.perceiver_num_queries,
            depth=config.perceiver_depth,
            num_heads=config.perceiver_num_heads,
            ff_mult=config.perceiver_ff_mult,
        )

        # ── Action injector ─────────────────────────────────────────────────
        if config.action.enabled:
            encoder = _build_action_encoder(config.action, config.action_dim, perceiver_kw)
            if config.action.injection_type == "cross_attn":
                self.action_injector: Optional[nn.Module] = CrossAttnInjector(
                    encoder, config.action.embed_dim, config.backbone_dim
                )
            elif config.action.injection_type == "adaln":
                self.action_injector = AdaLNInjector(
                    encoder, config.action.embed_dim, config.backbone_dim
                )
            else:
                raise ValueError(
                    f"action.injection_type must be 'cross_attn' or 'adaln', "
                    f"got {config.action.injection_type!r}"
                )
        else:
            self.action_injector = None

        # ── Masked-images injector ───────────────────────────────────────────
        self.masked_proj: Optional[nn.Linear] = None
        if config.masked.enabled:
            if config.masked.injection_type == "input_concat":
                self.masked_injector: Optional[nn.Module] = InputConcatInjector(
                    config.latent_channels
                )
            elif config.masked.injection_type == "cross_attn":
                self.masked_injector = None
                self.masked_proj = nn.Linear(config.latent_channels, config.backbone_dim)
            else:
                raise ValueError(
                    f"masked.injection_type must be 'input_concat' or 'cross_attn', "
                    f"got {config.masked.injection_type!r}"
                )
        else:
            self.masked_injector = None

        # ── History injector ─────────────────────────────────────────────────
        self.history_proj: Optional[nn.Linear] = None
        if config.history.enabled:
            if config.history.injection_type == "input_concat":
                self.history_injector: Optional[nn.Module] = InputConcatInjector(
                    config.latent_channels * config.history_len
                )
            elif config.history.injection_type == "cross_attn":
                self.history_injector = None
                self.history_proj = nn.Linear(config.latent_channels, config.backbone_dim)
            else:
                raise ValueError(
                    f"history.injection_type must be 'input_concat' or 'cross_attn', "
                    f"got {config.history.injection_type!r}"
                )
        else:
            self.history_injector = None

        self._extend_patch_embed()

    # ── Patch-embed surgery ─────────────────────────────────────────────────

    def _extra_input_channels(self) -> int:
        cfg = self.config
        extra = cfg.latent_channels + 1  # obs_image (always) + mask channel
        if cfg.masked.enabled and cfg.masked.injection_type == "input_concat":
            extra += cfg.latent_channels
        if cfg.history.enabled and cfg.history.injection_type == "input_concat":
            extra += cfg.latent_channels * cfg.history_len
        return extra

    def _replace_conv3d(self, old: nn.Conv3d, extra_ch: int) -> nn.Conv3d:
        """Return a new Conv3d with extra_ch more input channels, zero-initialised."""
        new_in = old.in_channels + extra_ch
        new = nn.Conv3d(
            new_in, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :old.in_channels] = old.weight
            new.weight[:, old.in_channels:] = 0.0
            if old.bias is not None:
                new.bias.copy_(old.bias)
        return new

    def _extend_patch_embed(self):
        extra_ch = self._extra_input_channels()
        if extra_ch == 0:
            return
        if self.config.backbone == "wan":
            self.backbone.patch_embedding = self._replace_conv3d(
                self.backbone.patch_embedding, extra_ch
            )
            # Disable the backbone's own image-concat logic; we pre-concat everything.
            self.backbone.has_image_input = False
        elif self.config.backbone == "cogvideo":
            self.backbone.patchify.proj = self._replace_conv3d(
                self.backbone.patchify.proj, extra_ch
            )

    # ── Input tensor assembly ───────────────────────────────────────────────

    def _build_input(
        self,
        noisy_latent: torch.Tensor,
        obs_latent: torch.Tensor,
        masked_latents: Optional[torch.Tensor],
        history_latents: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Concatenate all input_concat streams along the channel dimension.

        noisy_latent    : (B, C, T, H, W)
        obs_latent      : (B, C, T, H, W) or (B, C, 1, H, W) -- auto-expanded
        masked_latents  : (B, C, T, H, W)  for input_concat
        history_latents : (B, C*history_len, T, H, W)  for input_concat
        mask            : (B, 1, T, H, W)
        """
        cfg = self.config
        B, _, T, H, W = noisy_latent.shape

        # Broadcast single-frame obs over the generation horizon
        if obs_latent.shape[2] == 1:
            obs_latent = obs_latent.expand(-1, -1, T, -1, -1)

        if mask is None:
            mask = torch.zeros(
                B, 1, T, H, W,
                device=noisy_latent.device, dtype=noisy_latent.dtype,
            )

        parts = [noisy_latent, obs_latent, mask]

        if cfg.masked.enabled and cfg.masked.injection_type == "input_concat":
            parts.append(
                masked_latents
                if masked_latents is not None
                else torch.zeros_like(noisy_latent)
            )

        if cfg.history.enabled and cfg.history.injection_type == "input_concat":
            if history_latents is not None:
                parts.append(history_latents)
            else:
                n_ch = cfg.latent_channels * cfg.history_len
                parts.append(
                    torch.zeros(B, n_ch, T, H, W,
                                device=noisy_latent.device, dtype=noisy_latent.dtype)
                )

        return torch.cat(parts, dim=1)

    # ── Context token assembly ──────────────────────────────────────────────

    def _build_context(
        self,
        text_context: torch.Tensor,
        actions: Optional[torch.Tensor],
        masked_latents: Optional[torch.Tensor],
        history_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Pre-project text tokens via the backbone's own projection, then
        concatenate all cross_attn conditioning tokens in backbone_dim space.

        Passing context_is_projected=True to the backbone skips the internal
        text projection so tokens are not double-projected.

        text_context : (B, S, text_dim)  raw text embeddings
        Returns      : (B, S+N, backbone_dim)
        """
        cfg = self.config

        if cfg.backbone == "wan":
            projected_text = self.backbone.text_embedding(text_context)
        elif cfg.backbone == "cogvideo":
            projected_text = self.backbone.context_embedder(text_context)
        else:
            projected_text = text_context

        parts = [projected_text]

        if cfg.action.enabled and cfg.action.injection_type == "cross_attn" and actions is not None:
            parts.append(self.action_injector.get_context_tokens(actions))

        if (
            cfg.masked.enabled
            and cfg.masked.injection_type == "cross_attn"
            and masked_latents is not None
        ):
            B, C, T_m, H_m, W_m = masked_latents.shape
            tokens = masked_latents.permute(0, 2, 3, 4, 1).reshape(B, T_m * H_m * W_m, C)
            parts.append(self.masked_proj(tokens))

        if (
            cfg.history.enabled
            and cfg.history.injection_type == "cross_attn"
            and history_latents is not None
        ):
            B, C, T_h, H_h, W_h = history_latents.shape
            tokens = history_latents.permute(0, 2, 3, 4, 1).reshape(B, T_h * H_h * W_h, C)
            parts.append(self.history_proj(tokens))

        return torch.cat(parts, dim=1)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        obs_latent: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        history_latents: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **backbone_kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_latent    : (B, C, T, H, W)
        timestep        : (B,)
        text_context    : (B, S, text_dim)       raw text embeddings (pre-projection)
        obs_latent      : (B, C, T|1, H, W)      VAE-encoded observation image
        actions         : (B, T, action_dim)      optional
        masked_latents  : (B, C, T, H, W)         VAE-encoded trajectory-overlay images
                          shape for cross_attn: (B, C, T_m, H_m, W_m)
        history_latents : (B, C*history_len, T, H, W)  stacked past frames (input_concat)
                          shape for cross_attn: (B, C, T_hist, H, W)
        mask            : (B, 1, T, H, W)         0=generate, 1=conditioned  (optional)
        """
        x = self._build_input(noisy_latent, obs_latent, masked_latents, history_latents, mask)
        context = self._build_context(text_context, actions, masked_latents, history_latents)

        adaln_extra = None
        if (
            self.config.action.enabled
            and self.config.action.injection_type == "adaln"
            and actions is not None
        ):
            adaln_extra = self.action_injector.get_time_delta(actions)

        if self.config.backbone in ("wan", "cogvideo"):
            return self.backbone(
                x, timestep, context,
                adaln_extra=adaln_extra,
                context_is_projected=True,
                **backbone_kwargs,
            )
        raise ValueError(f"Unknown backbone: {self.config.backbone!r}")
