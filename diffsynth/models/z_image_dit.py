import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .general_modules import RMSNorm
from ..core.attention import attention_forward
from ..core.device.npu_compatible_device import IS_NPU_AVAILABLE, get_device_type
from ..core.gradient import gradient_checkpoint_forward


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
X_PAD_DIM = 64


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast(get_device_type(), enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))
        return t_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim_inner, q_dim, bias=bias_out)])

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)
    
    # Apply RoPE
    def apply_rotary_emb(self, x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(get_device_type(), enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)  # todo

    def forward(self, hidden_states, freqs_cis, attention_mask):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # Apply Norms
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = self.apply_rotary_emb(query, freqs_cis)
            key = self.apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Compute joint attention
        hidden_states = attention_forward(
            query,
            key,
            value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=attention_mask,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:  # dropout
            output = self.to_out[1](output)

        return output


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
    )


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            q_dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]

            if noise_mask is not None:
                # Per-token modulation: different modulation for noisy/clean tokens
                mod_noisy = self.adaLN_modulation(adaln_noisy)
                mod_clean = self.adaLN_modulation(adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                # Global modulation: same modulation for all tokens (avoid double select)
                mod = self.adaLN_modulation(adaln_input)
                scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Attention block
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]

        if noise_mask is not None:
            # Per-token modulation
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            # Original global modulation
            assert c is not None, "Either c or (c_noisy, c_clean) must be provided"
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            if IS_NPU_AVAILABLE:
                result.append(torch.index_select(self.freqs_cis[i], 0, index))
            else:
                result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageDiT(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        siglip_feat_dim=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        # Optional SigLIP components (for Omni variant)
        self.siglip_feat_dim = siglip_feat_dim
        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps), nn.Linear(siglip_feat_dim, dim, bias=True)
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        2000 + layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        norm_eps,
                        qk_norm,
                        modulation=False,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size = 2,
        f_patch_size = 1,
        x_pos_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            # Omni: extract target image from unified sequence (cond_images + target)
            result = []
            for i in range(bsz):
                unified_x = x[i][x_pos_offsets[i][0] : x_pos_offsets[i][1]]
                cu_len = 0
                x_item = None
                for j in range(len(size[i])):
                    if size[i][j] is None:
                        ori_len = 0
                        pad_len = SEQ_MULTI_OF
                        cu_len += pad_len + ori_len
                    else:
                        F, H, W = size[i][j]
                        ori_len = (F // pF) * (H // pH) * (W // pW)
                        pad_len = (-ori_len) % SEQ_MULTI_OF
                        x_item = (
                            unified_x[cu_len : cu_len + ori_len]
                            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                            .permute(6, 0, 3, 1, 4, 2, 5)
                            .reshape(self.out_channels, F, H, W)
                        )
                        cu_len += ori_len + pad_len
                result.append(x_item)  # Return only the last (target) image
            return result
        else:
            # Original mode: simple unpatchify
            for i in range(bsz):
                F, H, W = size[i]
                ori_len = (F // pF) * (H // pH) * (W // pW)
                # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
                x[i] = (
                    x[i][:ori_len]
                    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F, H, W)
                )
            return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return all_image_out, all_cap_feats_out, {
            "x_size": all_image_size,
            "x_pos_ids": all_image_pos_ids,
            "cap_pos_ids": all_cap_pos_ids,
            "x_pad_mask": all_image_pad_mask,
            "cap_pad_mask": all_cap_pad_mask
        }
    # (
    #         all_img_out,
    #         all_cap_out,
    #         all_img_size,
    #         all_img_pos_ids,
    #         all_cap_pos_ids,
    #         all_img_pad_mask,
    #         all_cap_pad_mask,
    #     )

    def patchify_controlnet(
        self,
        all_image: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        cap_padding_len: int = None,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        for i, image in enumerate(all_image):
            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
        )
    
    def _prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask: Optional[List[List[int]]] = None,
        device: torch.device = None,
    ):
        """Prepare sequence: apply pad token, RoPE embed, pad to batch, create attention mask."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Pad token
        feats_cat = torch.cat(feats, dim=0)
        feats_cat[torch.cat(inner_pad_mask)] = pad_token.to(dtype=feats_cat.dtype, device=feats_cat.device)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        # Pad to batch
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor
    
    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: List[int],
        x_noise_mask: Optional[List[List[int]]],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: List[int],
        cap_noise_mask: Optional[List[List[int]]],
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[torch.Tensor],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask: Optional[List[List[int]]],
        omni_mode: bool,
        device: torch.device,
    ):
        """Build unified sequence: x, cap, and optionally siglip.
        Basic mode order: [x, cap]; Omni mode order: [cap, x, siglip]
        """
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        unified_noise_mask = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]

            if omni_mode:
                # Omni: [cap, x, siglip]
                if siglip is not None and siglip_seqlens is not None:
                    sig_len = siglip_seqlens[i]
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len], siglip[i][:sig_len]]))
                    unified_freqs.append(
                        torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len], siglip_freqs[i][:sig_len]])
                    )
                    unified_noise_mask.append(
                        torch.tensor(
                            cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i], dtype=torch.long, device=device
                        )
                    )
                else:
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len]]))
                    unified_freqs.append(torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len]]))
                    unified_noise_mask.append(
                        torch.tensor(cap_noise_mask[i] + x_noise_mask[i], dtype=torch.long, device=device)
                    )
            else:
                # Basic: [x, cap]
                unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
                unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        # Compute unified seqlens
        if omni_mode:
            if siglip is not None and siglip_seqlens is not None:
                unified_seqlens = [a + b + c for a, b, c in zip(cap_seqlens, x_seqlens, siglip_seqlens)]
            else:
                unified_seqlens = [a + b for a, b in zip(cap_seqlens, x_seqlens)]
        else:
            unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]

        max_seqlen = max(unified_seqlens)

        # Pad to batch
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if omni_mode:
            noise_mask_tensor = pad_sequence(unified_noise_mask, batch_first=True, padding_value=0)[
                :, : unified.shape[1]
            ]

        return unified, unified_freqs, attn_mask, noise_mask_tensor
    
    def _pad_with_ids(
        self,
        feat: torch.Tensor,
        pos_grid_size: Tuple,
        pos_start: Tuple,
        device: torch.device,
        noise_mask_val: Optional[int] = None,
    ):
        """Pad feature to SEQ_MULTI_OF, create position IDs and pad mask."""
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len

        # Pos IDs
        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        noise_mask = [noise_mask_val] * total_len if noise_mask_val is not None else None  # token level
        return padded_feat, pos_ids, pad_mask, total_len, noise_mask
    
    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        """Patchify a single image tensor: (C, F, H, W) -> (num_patches, patch_dim)."""
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F, H, W = image.size()
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F, H, W), (F_tokens, H_tokens, W_tokens)
    
    def patchify_and_embed_omni(
        self,
        all_x: List[List[torch.Tensor]],
        all_cap_feats: List[List[torch.Tensor]],
        all_siglip_feats: List[List[torch.Tensor]],
        patch_size: int = 2,
        f_patch_size: int = 1,
        images_noise_mask: List[List[int]] = None,
    ):
        """Patchify for omni mode: multiple images per batch item with noise masks."""
        bsz = len(all_x)
        device = all_x[0][-1].device
        dtype = all_x[0][-1].dtype

        all_x_out, all_x_size, all_x_pos_ids, all_x_pad_mask, all_x_len, all_x_noise_mask = [], [], [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask, all_cap_len, all_cap_noise_mask = [], [], [], [], []
        all_sig_out, all_sig_pos_ids, all_sig_pad_mask, all_sig_len, all_sig_noise_mask = [], [], [], [], []

        for i in range(bsz):
            num_images = len(all_x[i])
            cap_feats_list, cap_pos_list, cap_mask_list, cap_lens, cap_noise = [], [], [], [], []
            cap_end_pos = []
            cap_cu_len = 1

            # Process captions
            for j, cap_item in enumerate(all_cap_feats[i]):
                noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
                cap_out, cap_pos, cap_mask, cap_len, cap_nm = self._pad_with_ids(
                    cap_item,
                    (len(cap_item) + (-len(cap_item)) % SEQ_MULTI_OF, 1, 1),
                    (cap_cu_len, 0, 0),
                    device,
                    noise_val,
                )
                cap_feats_list.append(cap_out)
                cap_pos_list.append(cap_pos)
                cap_mask_list.append(cap_mask)
                cap_lens.append(cap_len)
                cap_noise.extend(cap_nm)
                cap_cu_len += len(cap_item)
                cap_end_pos.append(cap_cu_len)
                cap_cu_len += 2  # for image vae and siglip tokens

            all_cap_out.append(torch.cat(cap_feats_list, dim=0))
            all_cap_pos_ids.append(torch.cat(cap_pos_list, dim=0))
            all_cap_pad_mask.append(torch.cat(cap_mask_list, dim=0))
            all_cap_len.append(cap_lens)
            all_cap_noise_mask.append(cap_noise)

            # Process images
            x_feats_list, x_pos_list, x_mask_list, x_lens, x_size, x_noise = [], [], [], [], [], []
            for j, x_item in enumerate(all_x[i]):
                noise_val = images_noise_mask[i][j]
                if x_item is not None:
                    x_patches, size, (F_t, H_t, W_t) = self._patchify_image(x_item, patch_size, f_patch_size)
                    x_out, x_pos, x_mask, x_len, x_nm = self._pad_with_ids(
                        x_patches, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device, noise_val
                    )
                    x_size.append(size)
                else:
                    x_len = SEQ_MULTI_OF
                    x_out = torch.zeros((x_len, X_PAD_DIM), dtype=dtype, device=device)
                    x_pos = self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(x_len, 1)
                    x_mask = torch.ones(x_len, dtype=torch.bool, device=device)
                    x_nm = [noise_val] * x_len
                    x_size.append(None)
                x_feats_list.append(x_out)
                x_pos_list.append(x_pos)
                x_mask_list.append(x_mask)
                x_lens.append(x_len)
                x_noise.extend(x_nm)

            all_x_out.append(torch.cat(x_feats_list, dim=0))
            all_x_pos_ids.append(torch.cat(x_pos_list, dim=0))
            all_x_pad_mask.append(torch.cat(x_mask_list, dim=0))
            all_x_size.append(x_size)
            all_x_len.append(x_lens)
            all_x_noise_mask.append(x_noise)

            # Process siglip
            if all_siglip_feats[i] is None:
                all_sig_len.append([0] * num_images)
                all_sig_out.append(None)
            else:
                sig_feats_list, sig_pos_list, sig_mask_list, sig_lens, sig_noise = [], [], [], [], []
                for j, sig_item in enumerate(all_siglip_feats[i]):
                    noise_val = images_noise_mask[i][j]
                    if sig_item is not None:
                        sig_H, sig_W, sig_C = sig_item.size()
                        sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                        sig_out, sig_pos, sig_mask, sig_len, sig_nm = self._pad_with_ids(
                            sig_flat, (1, sig_H, sig_W), (cap_end_pos[j] + 1, 0, 0), device, noise_val
                        )
                        # Scale position IDs to match x resolution
                        if x_size[j] is not None:
                            sig_pos = sig_pos.float()
                            sig_pos[..., 1] = sig_pos[..., 1] / max(sig_H - 1, 1) * (x_size[j][1] - 1)
                            sig_pos[..., 2] = sig_pos[..., 2] / max(sig_W - 1, 1) * (x_size[j][2] - 1)
                            sig_pos = sig_pos.to(torch.int32)
                    else:
                        sig_len = SEQ_MULTI_OF
                        sig_out = torch.zeros((sig_len, self.siglip_feat_dim), dtype=dtype, device=device)
                        sig_pos = (
                            self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(sig_len, 1)
                        )
                        sig_mask = torch.ones(sig_len, dtype=torch.bool, device=device)
                        sig_nm = [noise_val] * sig_len
                    sig_feats_list.append(sig_out)
                    sig_pos_list.append(sig_pos)
                    sig_mask_list.append(sig_mask)
                    sig_lens.append(sig_len)
                    sig_noise.extend(sig_nm)

                all_sig_out.append(torch.cat(sig_feats_list, dim=0))
                all_sig_pos_ids.append(torch.cat(sig_pos_list, dim=0))
                all_sig_pad_mask.append(torch.cat(sig_mask_list, dim=0))
                all_sig_len.append(sig_lens)
                all_sig_noise_mask.append(sig_noise)

        # Compute x position offsets
        all_x_pos_offsets = [(sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_x_len[i])) for i in range(bsz)]

        return (
            all_x_out,
            all_cap_out,
            all_sig_out,
            all_x_size,
            all_x_pos_ids,
            all_cap_pos_ids,
            all_sig_pos_ids,
            all_x_pad_mask,
            all_cap_pad_mask,
            all_sig_pad_mask,
            all_x_pos_offsets,
            all_x_noise_mask,
            all_cap_noise_mask,
            all_sig_noise_mask,
        )
        return all_x_out, all_cap_out, all_sig_out, {
            "x_size": x_size,
            "x_pos_ids": all_x_pos_ids,
            "cap_pos_ids": all_cap_pos_ids,
            "sig_pos_ids": all_sig_pos_ids,
            "x_pad_mask": all_x_pad_mask,
            "cap_pad_mask": all_cap_pad_mask,
            "sig_pad_mask": all_sig_pad_mask,
            "x_pos_offsets": all_x_pos_offsets,
            "x_noise_mask": all_x_noise_mask,
            "cap_noise_mask": all_cap_noise_mask,
            "sig_noise_mask": all_sig_noise_mask,
        }

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        siglip_feats = None,
        image_noise_mask = None,
        patch_size=2,
        f_patch_size=1,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        assert patch_size in self.all_patch_size and f_patch_size in self.all_f_patch_size
        omni_mode = isinstance(x[0], list)
        device = x[0][-1].device if omni_mode else x[0].device

        if omni_mode:
            # Dual embeddings: noisy (t) and clean (t=1)
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x[0][-1])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x[0][-1])
            adaln_input = None
        else:
            # Single embedding for all tokens
            adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
            t_noisy = t_clean = None

        # Patchify
        if omni_mode:
            (
                x,
                cap_feats,
                siglip_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                siglip_pos_ids,
                x_pad_mask,
                cap_pad_mask,
                siglip_pad_mask,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                siglip_noise_mask,
            ) = self.patchify_and_embed_omni(x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask)
        else:
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # x embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
        )

        for layer in self.noise_refiner:
            x = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=x, attn_mask=x_mask, freqs_cis=x_freqs, adaln_input=adaln_input, noise_mask=x_noise_tensor, adaln_noisy=t_noisy, adaln_clean=t_clean,
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )

        for layer in self.context_refiner:
            cap_feats = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=cap_feats,
                attn_mask=cap_mask,
                freqs_cis=cap_freqs,
            )

        # Siglip embed & refine
        siglip_seqlens = siglip_freqs = None
        if omni_mode and siglip_feats[0] is not None and self.siglip_embedder is not None:
            siglip_seqlens = [len(si) for si in siglip_feats]
            siglip_feats = self.siglip_embedder(torch.cat(siglip_feats, dim=0))  # embed
            siglip_feats, siglip_freqs, siglip_mask, _, _ = self._prepare_sequence(
                list(siglip_feats.split(siglip_seqlens, dim=0)),
                siglip_pos_ids,
                siglip_pad_mask,
                self.siglip_pad_token,
                None,
                device,
            )

            for layer in self.siglip_refiner:
                siglip_feats = gradient_checkpoint_forward(
                    layer,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                    x=siglip_feats, attn_mask=siglip_mask, freqs_cis=siglip_freqs,
                )

        # Unified sequence
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip_feats,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            omni_mode,
            device,
        )

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=unified, attn_mask=unified_mask, freqs_cis=unified_freqs, adaln_input=adaln_input, noise_mask=unified_noise_tensor, adaln_noisy=t_noisy, adaln_clean=t_clean
            )

        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified, noise_mask=unified_noise_tensor, c_noisy=t_noisy, c_clean=t_clean
            )
            if omni_mode
            else self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        # Unpatchify
        x = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets)

        return x
