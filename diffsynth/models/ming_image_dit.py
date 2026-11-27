# Copyright (c) Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math, torch
from typing import List, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
from ..core.distributed import sequence_parallel_recipe


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
LEARNED_PADDING = "learned"
ZERO_MASKED_PADDING = "zero_masked"
VALID_PADDING_MODES = {LEARNED_PADDING, ZERO_MASKED_PADDING}


def mask_out_alignment_padding(attention_mask, pad_masks, offsets):
    for item_index, (pad_mask, offset) in enumerate(zip(pad_masks, offsets)):
        if pad_mask is None or pad_mask.numel() == 0:
            continue
        offset = int(offset)
        end = offset + pad_mask.shape[0]
        attention_mask[item_index, offset:end].masked_fill_(pad_mask, False)
    return attention_mask


class MingImageRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states


class MingImageAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = MingImageRMSNorm(dim_head, eps=eps)
        self.norm_k = MingImageRMSNorm(dim_head, eps=eps)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        def apply_rotary_emb(x_in, freqs_cis):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=attention_mask,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:
            output = self.to_out[1](output)
        return output


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
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
        t_freq = t_freq.to(t.dtype)
        t_emb = self.mlp(t_freq)
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


@sequence_parallel_recipe(
    split_dimensions_for_inputs={"x": 1, "freqs_cis": 1},
    gather_dimensions_for_outputs={"x": 1},
)
class TransformerBlock(nn.Module):
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

        self.attention = MingImageAttention(
            dim=dim,
            heads=n_heads,
            dim_head=dim // n_heads,
            eps=1e-5,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = MingImageRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = MingImageRMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = MingImageRMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = MingImageRMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
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

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
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
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class MingImageDiT(nn.Module):
    _repeated_blocks = ["TransformerBlock"]

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
        axes_lens=[20480, 512, 512],
        alignment_padding_mode="zero_masked",
        multi_frame_output=False,
    ) -> None:
        super().__init__()
        if alignment_padding_mode not in VALID_PADDING_MODES:
            raise ValueError(
                "alignment_padding_mode must come from the checkpoint capability "
                "contract (transformer/config.json, or the legacy "
                f"inference_profile.json) and be one of "
                f"{sorted(VALID_PADDING_MODES)}, got {alignment_padding_mode!r}"
            )
        if type(multi_frame_output) is not bool:
            raise ValueError(
                "multi_frame_output must come from the checkpoint capability "
                "contract (transformer/config.json, or the legacy "
                "inference_profile.json) as a boolean"
            )
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.alignment_padding_mode = alignment_padding_mode
        self.multi_frame_output = multi_frame_output

        self.rope_theta = rope_theta
        self.t_scale = t_scale

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
                TransformerBlock(
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
                TransformerBlock(
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
        self.cap_embedder = nn.Sequential(MingImageRMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        if self.alignment_padding_mode == LEARNED_PADDING:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.register_parameter("x_pad_token", None)
            self.register_parameter("cap_pad_token", None)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
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
            if not self.multi_frame_output:
                x[i] = x[i][:, :1, :, :]
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
        patch_size: int,
        f_patch_size: int,
        all_image_ref: List[torch.Tensor] = None,
        all_cap_feats_2: List[torch.Tensor] = None,
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
        all_cap_feats_2_out = []

        if all_image_ref is None:
            all_image_ref = [None]*len(all_image)

        assert all_cap_feats is not None or all_cap_feats_2 is not None

        if all_cap_feats is None:
            all_cap_feats = [None for _ in range(len(all_image))]
        else:
            assert not any([i is None for i in all_cap_feats])
        
        if all_cap_feats_2 is None:
            all_cap_feats_2 = [None for _ in range(len(all_image))]
        else:
            assert not any([i is None for i in all_cap_feats_2])

        for i, (image, cap_feat, cap_feat_2, image_ref) in enumerate(zip(all_image, all_cap_feats, all_cap_feats_2, all_image_ref)):
            ### Process Caption
            cap_ori_len = 0
            if cap_feat is not None:
                cap_ori_len += len(cap_feat)
            
            if cap_feat_2 is not None:
                cap_ori_len += len(cap_feat_2)
            
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask if cap_padding_len > 0 else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)
            )

            if cap_feat_2 is not None:
                if cap_feat is not None:
                    all_cap_feats_out.append(cap_feat)
                cap_padded_feat = torch.cat([cap_feat_2, (cap_feat_2[-1:] * 0).repeat(cap_padding_len, 1)], dim=0)
                all_cap_feats_2_out.append(cap_padded_feat)
            else:
                cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
                all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            if image_ref is not None:
                image = torch.cat([image, image_ref], dim=1)

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
            image_padded_pos_ids = torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
            all_cap_feats_2_out,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        ref_x=None,
        cap_feats_2=None,
        return_dict: bool = True,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = t.to(x[0].dtype)
        t = self.t_embedder(t)
        
        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
            cap_feats_2,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size, ref_x, cap_feats_2)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        if self.alignment_padding_mode == LEARNED_PADDING:
            x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        else:
            x[torch.cat(x_inner_pad_mask)] = 0.0
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1
        if self.alignment_padding_mode == ZERO_MASKED_PADDING:
            mask_out_alignment_padding(x_attn_mask, x_inner_pad_mask, [0] * bsz)

        for layer in self.noise_refiner:
            x = gradient_checkpoint_forward(
                layer, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                x, x_attn_mask, x_freqs_cis, adaln_input,
            )


        if len(cap_feats) > 0:
            # cap embed & refine
            cap_item_seqlens = [len(_) for _ in cap_feats]
            cap_feats = torch.cat(cap_feats, dim=0)
            cap_feats = self.cap_embedder(cap_feats)

            if len(cap_feats_2) > 0:
                cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
                assert len(cap_feats) == len(cap_feats_2)
                assert cap_feats[0].ndim == 2
                assert cap_feats_2[0].ndim == 2
                cap_feats = [torch.cat([i, j], dim=0) for i, j in zip(cap_feats, cap_feats_2)]
                cap_item_seqlens = [len(_) for _ in cap_feats]
                cap_feats = torch.cat(cap_feats, dim=0)
        else:
            assert len(cap_feats_2) > 0
            cap_item_seqlens = [len(_) for _ in cap_feats_2]
            cap_feats = torch.cat(cap_feats_2, dim=0)
        
        cap_max_item_seqlen = max(cap_item_seqlens)
        if self.alignment_padding_mode == LEARNED_PADDING:
            cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        else:
            cap_feats[torch.cat(cap_inner_pad_mask)] = 0.0

        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1
        if self.alignment_padding_mode == ZERO_MASKED_PADDING:
            mask_out_alignment_padding(cap_attn_mask, cap_inner_pad_mask, [0] * bsz)

        for layer in self.context_refiner:
            cap_feats = gradient_checkpoint_forward(
                layer, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                cap_feats, cap_attn_mask, cap_freqs_cis,
            )

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
        if self.alignment_padding_mode == ZERO_MASKED_PADDING:
            # Unified layout is [image, caption] for every batch item.
            mask_out_alignment_padding(unified_attn_mask, x_inner_pad_mask, [0] * bsz)
            mask_out_alignment_padding(
                unified_attn_mask, cap_inner_pad_mask, x_item_seqlens
            )

        for layer in self.layers:
            unified = gradient_checkpoint_forward(
                layer, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                unified, unified_attn_mask, unified_freqs_cis, adaln_input,
            )

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        return (x,)
