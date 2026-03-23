import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange
from .wan_video_dit import AttentionModule, RMSNorm
from ..core import gradient_checkpoint_forward

class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, base: float, dim: int, device=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PerFrameAttentionPooling(nn.Module):
    """
    Per-frame multi-head attention pooling.

    Given a flattened token sequence [B, L, D] and grid size (T, H, W), perform a
    single-query attention pooling over the H*W tokens for each time frame, producing
    [B, T, D].

    Inspired by SigLIP's Multihead Attention Pooling head (without MLP/residual stack).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, L, D], where L = T*H*W
            grid_size: (T, H, W)
        Returns:
            pooled: [B, T, D]
        """
        B, L, D = x.shape
        T, H, W = grid_size
        assert D == self.dim, f"Channel dimension mismatch: D={D} vs dim={self.dim}"
        assert L == T * H * W, f"Flattened length mismatch: L={L} vs T*H*W={T*H*W}"

        S = H * W
        # Re-arrange tokens grouped by frame.
        x_bt_s_d = x.view(B, T, S, D).contiguous().view(B * T, S, D)  # [B*T, S, D]

        # A learnable probe as the query (one query per frame).
        probe = self.probe.expand(B * T, -1, -1)  # [B*T, 1, D]

        # Attention pooling: query=probe, key/value=H*W tokens within the frame.
        pooled_bt_1_d = self.attention(probe, x_bt_s_d, x_bt_s_d, need_weights=False)[0]  # [B*T, 1, D]
        pooled_bt_d = pooled_bt_1_d.squeeze(1)  # [B*T, D]

        # Restore to [B, T, D].
        pooled = pooled_bt_d.view(B, T, D)
        pooled = self.layernorm(pooled)
        return pooled


class CrossModalInteractionController:
    """
    Strategy class that controls interactions between two towers.
    Manages the interaction mapping between visual DiT (e.g. 30 layers) and audio DiT (e.g. 30 layers).
    """

    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(self, strategy: str = "shallow_focus") -> Dict[str, List[Tuple[int, int]]]:
        """
        Get interaction layer mappings.

        Args:
            strategy: interaction strategy
                - "shallow_focus": emphasize shallow layers to avoid deep-layer asymmetry
                - "distributed": distributed interactions across the network
                - "progressive": dense shallow interactions, sparse deeper interactions
                - "custom": custom interaction layers

        Returns:
            A dict containing mappings for 'v2a' (visual -> audio) and 'a2v' (audio -> visual).
        """

        if strategy == "shallow_focus":
            # Emphasize the first ~1/3 layers to avoid deep-layer asymmetry.
            num_interact = min(10, self.min_layers // 3)
            interact_layers = list(range(0, num_interact))

        elif strategy == "distributed":
            # Distribute interactions across the network (every few layers).
            step = 3
            interact_layers = list(range(0, self.min_layers, step))

        elif strategy == "progressive":
            # Progressive: dense shallow interactions, sparse deeper interactions.
            shallow = list(range(0, min(8, self.min_layers)))  # Dense for the first 8 layers.
            if self.min_layers > 8:
                deep = list(range(8, self.min_layers, 3))  # Every 3 layers afterwards.
                interact_layers = shallow + deep
            else:
                interact_layers = shallow

        elif strategy == "custom":
            # Custom strategy: adjust as needed.
            interact_layers = [0, 2, 4, 6, 8, 12, 16, 20]  # Explicit layer indices.
            interact_layers = [i for i in interact_layers if i < self.min_layers]

        elif strategy == "full":
            interact_layers = list(range(0, self.min_layers))

        else:
            raise ValueError(f"Unknown interaction strategy: {strategy}")

        # Build bidirectional mapping.
        mapping = {
            'v2a': [(i, i) for i in interact_layers],  # visual layer i -> audio layer i
            'a2v': [(i, i) for i in interact_layers]   # audio layer i -> visual layer i
        }

        return mapping

    def should_interact(self, layer_idx: int, direction: str, interaction_mapping: Dict) -> bool:
        """
        Check whether a given layer should interact.

        Args:
            layer_idx: current layer index
            direction: interaction direction ('v2a' or 'a2v')
            interaction_mapping: interaction mapping table

        Returns:
            bool: whether to interact
        """
        if direction not in interaction_mapping:
            return False

        return any(src == layer_idx for src, _ in interaction_mapping[direction])


class ConditionalCrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q_dim = dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = self.q_dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        if x_freqs is not None:
            x_cos, x_sin = x_freqs
            B, L, _ = q.shape
            q_view = rearrange(q, 'b l (h d) -> b l h d', d=self.head_dim)
            x_cos = x_cos.to(q_view.dtype).to(q_view.device)
            x_sin = x_sin.to(q_view.dtype).to(q_view.device)
            # Expect x_cos/x_sin shape: [B or 1, L, head_dim]
            q_view, _ = apply_rotary_pos_emb(q_view, q_view, x_cos, x_sin, unsqueeze_dim=2)
            q = rearrange(q_view, 'b l h d -> b l (h d)')
        if y_freqs is not None:
            y_cos, y_sin = y_freqs
            Bc, Lc, _ = k.shape
            k_view = rearrange(k, 'b l (h d) -> b l h d', d=self.head_dim)
            y_cos = y_cos.to(k_view.dtype).to(k_view.device)
            y_sin = y_sin.to(k_view.dtype).to(k_view.device)
            # Expect y_cos/y_sin shape: [B or 1, L, head_dim]
            _, k_view = apply_rotary_pos_emb(k_view, k_view, y_cos, y_sin, unsqueeze_dim=2)
            k = rearrange(k_view, 'b l h d -> b l (h d)')
        x = self.attn(q, k, v)
        return self.o(x)


# from diffusers.models.attention import AdaLayerNorm
class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 2:
            scale, shift = temb.chunk(2, dim=2)
            # print(f"{x.shape = }, {scale.shape = }, {shift.shape = }")
        elif self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX and OmniGen for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class ConditionalCrossAttentionBlock(nn.Module):
    """
    A thin wrapper around ConditionalCrossAttention.
    Applies LayerNorm to the conditioning input `y` before cross-attention.
    """
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6, pooled_adaln: bool = False):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(kv_dim, num_heads=num_heads, eps=eps)
            self.adaln = AdaLayerNorm(kv_dim, output_dim=dim*2, chunk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            assert video_grid_size is not None, "video_grid_size must not be None"
            pooled_y = self.per_frame_pooling(y, video_grid_size)
            # Interpolate pooled_y along its temporal dimension to match x's sequence length.
            if pooled_y.shape[1] != x.shape[1]:
                pooled_y = F.interpolate(
                    pooled_y.permute(0, 2, 1),  # [B, C, T]
                    size=x.shape[1],
                    mode='linear',
                    align_corners=False,
                ).permute(0, 2, 1)  # [B, T, C]
            x = self.adaln(x, temb=pooled_y)
        y = self.y_norm(y)
        return self.inner(x=x, y=y, x_freqs=x_freqs, y_freqs=y_freqs)


class DualTowerConditionalBridge(nn.Module):
    """
    Dual-tower conditional bridge.
    """
    def __init__(self,
                 visual_layers: int = 40,
                 audio_layers: int = 30,
                 visual_hidden_dim: int = 5120,    # visual DiT hidden state dimension
                 audio_hidden_dim: int = 1536,     # audio DiT hidden state dimension
                 audio_fps: float = 50.0,
                 head_dim: int = 128,              # attention head dimension
                 interaction_strategy: str = "full",
                 apply_cross_rope: bool = True,   # whether to apply RoPE in cross-attention
                 apply_first_frame_bias_in_rope: bool = False,  # whether to account for 1/video_fps bias for the first frame in RoPE alignment
                 trainable_condition_scale: bool = False,
                 pooled_adaln: bool = False,
                 ):
        super().__init__()

        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln
        if self.trainable_condition_scale:
            self.condition_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.condition_scale = 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(interaction_strategy)

        # Conditional cross-attention modules operating at the DiT hidden-state level.
        self.audio_to_video_conditioners = nn.ModuleDict()  # audio hidden states -> visual DiT conditioning
        self.video_to_audio_conditioners = nn.ModuleDict()  # visual hidden states -> audio DiT conditioning

        # Build conditioners for layers that should interact.
        # audio hidden states condition the visual DiT
        self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)
        for v_layer, _ in self.interaction_mapping['a2v']:
            self.audio_to_video_conditioners[str(v_layer)] = ConditionalCrossAttentionBlock(
                dim=visual_hidden_dim,     # 3072 (visual DiT hidden states)
                kv_dim=audio_hidden_dim,    # 1536 (audio DiT hidden states)
                num_heads=visual_hidden_dim // head_dim, # derive number of heads from hidden dim
                pooled_adaln=False # a2v typically does not need pooled AdaLN
            )

        # visual hidden states condition the audio DiT
        for a_layer, _ in self.interaction_mapping['v2a']:
            self.video_to_audio_conditioners[str(a_layer)] = ConditionalCrossAttentionBlock(
                dim=audio_hidden_dim,      # 1536 (audio DiT hidden states)
                kv_dim=visual_hidden_dim,   # 3072 (visual DiT hidden states)
                num_heads=audio_hidden_dim // head_dim, # safe head count derivation
                pooled_adaln=self.pooled_adaln
            )

    @torch.no_grad()
    def build_aligned_freqs(self,
                            video_fps: float,
                            grid_size: Tuple[int, int, int],
                            audio_steps: int,
                            device: Optional[torch.device] = None,
                            dtype: Optional[torch.dtype] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build aligned RoPE (cos, sin) pairs based on video fps, video grid size (f_v, h, w),
        and audio sequence length `audio_steps` (with fixed audio fps = 44100/2048).

        Returns:
            visual_freqs: (cos_v, sin_v), shape [1, f_v*h*w, head_dim]
            audio_freqs:  (cos_a, sin_a), shape [1, audio_steps, head_dim]
        """
        f_v, h, w = grid_size
        L_v = f_v * h * w
        L_a = int(audio_steps)

        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        # Audio positions: 0,1,2,...,L_a-1 (audio as reference).
        audio_pos = torch.arange(L_a, device=device, dtype=torch.float32).unsqueeze(0)

        # Video positions: align video frames to audio-step units.
        # FIXME(dhyu): hard-coded VAE temporal stride = 4
        if self.apply_first_frame_bias_in_rope:
            # Account for the "first frame lasts 1/video_fps" bias.
            video_effective_fps = float(video_fps) / 4.0
            if f_v > 0:
                t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
                if f_v > 1:
                    t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(f_v - 1, device=device, dtype=torch.float32) * (1.0 / video_effective_fps)
            else:
                t_starts = torch.zeros((0,), device=device, dtype=torch.float32)
            # Convert to audio-step units.
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            # No first-frame bias: uniform alignment.
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = torch.arange(f_v, device=device, dtype=torch.float32) * scale
        # Flatten to f*h*w; tokens within the same frame share the same time position.
        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

        # print(f"video fps: {video_fps}, audio fps: {self.audio_fps}, scale: {scale}")
        # print(f"video pos: {video_pos.shape}, audio pos: {audio_pos.shape}")

        # Build dummy x to produce cos/sin, dim=head_dim.
        dummy_v = torch.zeros((1, L_v, self.head_dim), device=device, dtype=dtype)
        dummy_a = torch.zeros((1, L_a, self.head_dim), device=device, dtype=dtype)

        cos_v, sin_v = self.rotary(dummy_v, position_ids=video_pos)
        cos_a, sin_a = self.rotary(dummy_a, position_ids=audio_pos)

        return (cos_v, sin_v), (cos_a, sin_a)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(layer_idx, direction, self.interaction_mapping)

    def apply_conditional_control(
        self,
        layer_idx: int,
        direction: str,
        primary_hidden_states: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
        use_gradient_checkpointing: Optional[bool] = False,
        use_gradient_checkpointing_offload: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Apply conditional control (at the DiT hidden-state level).

        Args:
            layer_idx: current layer index
            direction: conditioning direction
                - 'a2v': audio hidden states -> visual DiT
                - 'v2a': visual hidden states -> audio DiT
            primary_hidden_states: primary DiT hidden states [B, L, hidden_dim]
            condition_hidden_states: condition DiT hidden states [B, L, hidden_dim]
            condition_scale: conditioning strength (similar to CFG scale)

        Returns:
            Conditioned primary DiT hidden states [B, L, hidden_dim]
        """

        if not self.controller.should_interact(layer_idx, direction, self.interaction_mapping):
            return primary_hidden_states

        if direction == 'a2v':
            # audio hidden states condition the visual DiT
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]

        elif direction == 'v2a':
            # visual hidden states condition the audio DiT
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditioned_features = gradient_checkpoint_forward(
            conditioner,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x=primary_hidden_states,
            y=condition_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            video_grid_size=video_grid_size,
        )

        if self.trainable_condition_scale and condition_scale is not None:
            print(
                "[WARN] This model has a trainable condition_scale, but an external "
                f"condition_scale={condition_scale} was provided. The trainable condition_scale "
                "will be ignored in favor of the external value."
            )

        scale = condition_scale if condition_scale is not None else self.condition_scale

        primary_hidden_states = primary_hidden_states + conditioned_features * scale

        return primary_hidden_states

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
        use_gradient_checkpointing: Optional[bool] = False,
        use_gradient_checkpointing_offload: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional conditional control to both visual/audio towers.

        Args:
            layer_idx: current layer index
            visual_hidden_states: visual DiT hidden states
            audio_hidden_states: audio DiT hidden states
            x_freqs / y_freqs: cross-modal RoPE (cos, sin) pairs.
                If provided, x_freqs is assumed to correspond to the primary tower and y_freqs
                to the conditioning tower.
            a2v_condition_scale: audio->visual conditioning strength (overrides global condition_scale)
            v2a_condition_scale: visual->audio conditioning strength (overrides global condition_scale)
            condition_scale: fallback conditioning strength when per-direction scale is None
            video_grid_size: (F, H, W), used on the audio side when pooled_adaln is enabled

        Returns:
            (visual_hidden_states, audio_hidden_states), both conditioned in their respective directions.
        """

        visual_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="a2v",
            primary_hidden_states=visual_hidden_states,
            condition_hidden_states=audio_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            condition_scale=a2v_condition_scale if a2v_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        audio_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="v2a",
            primary_hidden_states=audio_hidden_states,
            condition_hidden_states=visual_hidden_states,
            x_freqs=y_freqs,
            y_freqs=x_freqs,
            condition_scale=v2a_condition_scale if v2a_condition_scale is not None else condition_scale,
            video_grid_size=video_grid_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

        return visual_conditioned, audio_conditioned
