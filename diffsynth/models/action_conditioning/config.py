from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConditionStreamConfig:
    """
    Configuration for a single conditioning stream.

    injection_type
    --------------
    "cross_attn"   : tokens appended to the cross-attention context sequence
    "input_concat" : latents concatenated as extra input channels before patchify
    "adaln"        : pooled embedding added to the time embedding (action only)

    encoder_type
    ------------
    "perceiver"    : Resampler / Perceiver (maps variable-length seq → fixed N tokens)
    "mlp"          : per-timestep MLP (preserves sequence length)
    "vae"          : no learnable encoder; VAE latents used directly
    "identity"     : pass-through placeholder
    """

    injection_type: Literal["cross_attn", "input_concat", "adaln"]
    encoder_type: Literal["perceiver", "mlp", "vae", "identity"]
    embed_dim: int = 1024
    enabled: bool = True


@dataclass
class ActionConditioningConfig:
    """
    Full configuration for action-conditioned DiT ablation experiments.

    The three conditioning streams (action, masked, history) each have an
    independently switchable injection_type.  Swap injection_type in the
    config YAML — no code changes needed.

    obs_image is always input_concat (the fixed spatial anchor).

    Typical experiment configs
    --------------------------
    Exp-A  all cross-attention:
        action  = cross_attn / perceiver
        masked  = cross_attn / vae
        history = cross_attn / vae

    Exp-B  EVAC-style mixed (recommended baseline):
        action  = cross_attn / perceiver
        masked  = input_concat / vae
        history = input_concat / vae

    Exp-C  action via AdaLN:
        action  = adaln / mlp
        masked  = input_concat / vae
        history = input_concat / vae
    """

    backbone: Literal["wan", "cogvideo"] = "wan"

    # Hidden dimension of the backbone *after* its internal projections.
    # Wan 1.3B: 1536  |  Wan 14B: 5120  |  CogVideo: 3072
    backbone_dim: int = 5120

    # Raw text embedding dimension (input to backbone's text projection).
    # Wan & CogVideo: 4096
    text_dim: int = 4096

    # VAE latent channels
    latent_channels: int = 16

    # Raw action feature size (e.g. 14-dim delta actions from EVAC)
    action_dim: int = 14

    # Number of past frames kept as rolling history (3-4 recommended)
    history_len: int = 4

    # ── Per-stream configs ──────────────────────────────────────────────────
    # obs_image is always input_concat; it is the fixed anchor and not ablated.

    action: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="cross_attn",
            encoder_type="perceiver",
            embed_dim=1024,
        )
    )
    masked: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="input_concat",
            encoder_type="vae",
            embed_dim=16,
        )
    )
    history: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="input_concat",
            encoder_type="vae",
            embed_dim=16,
        )
    )

    # ── Perceiver hyper-params (when encoder_type="perceiver") ──────────────
    perceiver_num_queries: int = 16
    perceiver_depth: int = 4
    perceiver_num_heads: int = 8
    perceiver_ff_mult: int = 4
