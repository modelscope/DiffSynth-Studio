import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.attention.flex_attention import create_block_mask
from .wan_video_dit import sinusoidal_embedding_1d, RMSNorm, MLP
from ..core.attention.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
from functools import lru_cache, partial


def _score_mod_impl(score, b_idx, h_idx, q_idx, kv_idx, hw: int, log_scale: float):
    condition = (kv_idx >= hw) & (kv_idx < 2 * hw)
    return torch.where(condition, score + log_scale, score)


@lru_cache(maxsize=32)
def _get_score_mod(hw: int, log_scale: float = -1.0):
    return partial(_score_mod_impl, hw=hw, log_scale=log_scale)


def rope_params(max_seq_len, dim, theta=10000, offset=0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len)+offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs, time_stride=1):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f*time_stride:time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


class LayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


class SelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def pre_attention(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        return q, k, v

    def post_attention(self, x):
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, use_img_emb=True):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.use_img_emb = use_img_emb
        if use_img_emb:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, counter=0):
        if self.use_img_emb:
            context_img = context[:, :257]
            context = context[:, 257:]

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if self.use_img_emb:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            img_x = attention_forward(
                q, k_img, v_img,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            )
        # compute attention
        x = attention_forward(
            q, k, v,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )

        # output
        x = x.flatten(2)
        if self.use_img_emb:
            img_x = img_x.flatten(2)
            x = x + img_x
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_img_emb=True
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)

        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)

        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()

        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, use_img_emb=use_img_emb)

        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def pre_self_attention(self, x, e):
        e = (self.modulation.to(x.device) + e).chunk(6, dim=1)
        q, k, v = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], method="pre_attention")
        return q, k, v, e

    def post_self_attention(self, x):
        x = self.self_attn(x, method="post_attention")
        return x

    def cross_attention(self, x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Incontext_AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        refer_stride=1,
        use_img_emb=True,
        use_context_parallel=False,
        sparse_type=0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.refer_stride = refer_stride
        self.use_context_parallel = use_context_parallel
        self.sparse_type = sparse_type

        self.block = AttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, use_img_emb=use_img_emb
        )

    def forward(self, *args, method, **kwargs):
        return getattr(self, method)(*args, **kwargs)

    def forward_ref(
        self,
        x_ref,
        index,
        k_cache,
        v_cache,
        context_ref,
        freqs_ref,
        grid_sizes_ref,
        e_ref,
        context_lens,
        animate2_offload_kv=False,
        use_context_parallel=False,
    ):
        q_ref, k_ref, v_ref, e_ref = self.block(x_ref, e_ref, method='pre_self_attention')

        if use_context_parallel:
            from ..utils.xfuser import all_to_all_4d, is_evenly_divisible, get_sequence_parallel_world_size
            assert is_evenly_divisible(q_ref.shape[2]), (
                f"num_heads ({q_ref.shape[2]}) must be divisible by sequence parallel world size ({get_sequence_parallel_world_size()}) "
            )
            qkv_ref = torch.cat([q_ref, k_ref, v_ref], dim=0)
            qkv_ref = all_to_all_4d(qkv_ref, scatter_dim=2, gather_dim=1)
            q_ref, k_ref, v_ref = qkv_ref.chunk(3, dim=0)

        k_cache[index] = k_ref if not animate2_offload_kv else k_ref.to('cpu')
        v_cache[index] = v_ref if not animate2_offload_kv else v_ref.to('cpu')
        q_ref_add_rope = rope_apply(q_ref, grid_sizes_ref, freqs_ref, self.refer_stride)
        k_ref_add_rope = rope_apply(k_ref, grid_sizes_ref, freqs_ref, self.refer_stride)

        ref_f, ref_h, ref_w = grid_sizes_ref[0].tolist()
        ref_vail_len = ref_f * ref_h * ref_w

        xout_ref = attention_forward(
                q_ref_add_rope.to(v_ref.dtype),
                k_ref_add_rope[:, :ref_vail_len].to(v_ref.dtype),
                v_ref[:, :ref_vail_len],
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        ).to(q_ref_add_rope.dtype)

        if use_context_parallel:
            xout_ref = all_to_all_4d(xout_ref, scatter_dim=1, gather_dim=2)

        y_ref = self.block(xout_ref, method='post_self_attention')

        x_ref = x_ref + y_ref * e_ref[2]

        x_ref = self.block(x_ref, context_ref, context_lens, e_ref, method='cross_attention')

        return x_ref

    def forward_gen(
        self,
        x,
        index,
        k_cache,
        v_cache,
        attn_mask,
        context,
        freqs,
        freqs_ref,
        grid_sizes,
        grid_sizes_ref,
        origin_len,
        origin_area,
        e,
        context_lens,
        use_context_parallel=False,
        log_scale=0.0,
    ):

        origin_latent_f    = origin_len // 4 + 1
        origin_latent_hw   = origin_area[0] * origin_area[1] // 256
        origin_max_len     = (origin_latent_f + 1) * origin_latent_hw
        origin_ref_max_len = origin_latent_f * origin_latent_hw

        f, h, w = grid_sizes[0].tolist()
        vail_len = f * h * w
        hw = h * w

        ref_f, ref_h, ref_w = grid_sizes_ref[0].tolist()
        ref_vail_len = ref_f * ref_h * ref_w
        ref_hw = ref_h * ref_w

        q, k, v, e = self.block(x, e, method='pre_self_attention')
        if use_context_parallel:
            from ..utils.xfuser import all_to_all_4d, is_evenly_divisible, get_sequence_parallel_world_size
            assert is_evenly_divisible(q.shape[2]), (
                f"num_heads ({q.shape[2]}) must be divisible by sequence parallel world size ({get_sequence_parallel_world_size()}) "
            )
            qkv = torch.cat([q, k, v], dim=0)
            qkv = all_to_all_4d(qkv, scatter_dim=2, gather_dim=1)
            q, k, v = qkv.chunk(3, dim=0)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        k_ref, v_ref = k_cache[index].to(x.device), v_cache[index].to(x.device)
        k_ref = rope_apply(k_ref, grid_sizes_ref, freqs_ref, self.refer_stride)

        B, _, N, C = q.shape
        device, dtype = q.device, q.dtype

        target_q_len   = math.ceil(origin_max_len     / 128) * 128
        target_ref_len = math.ceil(origin_ref_max_len / 128) * 128
        target_kv_len  = target_q_len + target_ref_len

        q_padding = q[:, vail_len:].clone()

        q_incontext = torch.zeros(B, target_q_len,  N, C, device=device, dtype=dtype)
        k_incontext = torch.zeros(B, target_kv_len, N, C, device=device, dtype=dtype)
        v_incontext = torch.zeros(B, target_kv_len, N, C, device=device, dtype=dtype)

        q_src = q[:, :vail_len].view(B, f, hw, N, C)
        k_src = k[:, :vail_len].view(B, f, hw, N, C)
        v_src = v[:, :vail_len].view(B, f, hw, N, C)

        q_incontext[:, :f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = q_src
        k_incontext[:, :f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = k_src
        v_incontext[:, :f * origin_latent_hw].view(B, f, origin_latent_hw, N, C)[:, :, :hw] = v_src

        k_ref_src = k_ref[:, :ref_vail_len].view(B, ref_f, ref_hw, N, C)
        v_ref_src = v_ref[:, :ref_vail_len].view(B, ref_f, ref_hw, N, C)

        k_incontext[:, target_q_len : target_q_len + ref_f * origin_latent_hw]\
            .view(B, ref_f, origin_latent_hw, N, C)[:, :, :ref_hw] = k_ref_src
        v_incontext[:, target_q_len : target_q_len + ref_f * origin_latent_hw]\
            .view(B, ref_f, origin_latent_hw, N, C)[:, :, :ref_hw] = v_ref_src

        score_mod = _get_score_mod(hw=int(origin_latent_hw), log_scale=log_scale)

        xout_full = attention_forward(
            q_incontext, k_incontext, v_incontext,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            use_flex=True, attn_mask=attn_mask, score_mod=score_mod
        )

        xout_valid = xout_full[:, :f * origin_latent_hw]
        xout_valid = xout_valid.view(B, f, origin_latent_hw, N, C)
        xout_vail = xout_valid[:, :, :hw]
        xout_vail = xout_vail.reshape(B, f * hw, N, C)  # [B, f*hw, N, C]
        xout = torch.cat([xout_vail, q_padding], dim=1)
        if use_context_parallel:
            xout = all_to_all_4d(xout, scatter_dim=1, gather_dim=2)

        y = self.block(xout, method='post_self_attention')

        x = x + y * e[2]

        x = self.block(x, context, context_lens, e, method='cross_attention')
        return x

    def forward_origin(self, x, x_ref, ref_args, gen_args):
        k_cache, v_cache = {}, {}
        x_ref = self.forward_ref(x_ref, 0, k_cache, v_cache, **ref_args)
        x = self.forward_gen(x, 0, k_cache, v_cache, **gen_args)
        return x, x_ref


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanAnimate2Transformer(nn.Module):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_img_emb=True,
        refer_offset_t=1,
        refer_offset_h=0,
        refer_offset_w=-1,
        refer_stride=1,
        sparse_type=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_img_emb = use_img_emb
        self.refer_offset_t = refer_offset_t
        self.refer_offset_h = refer_offset_h
        self.refer_offset_w = refer_offset_w
        self.refer_stride = refer_stride
        self.sparse_type = sparse_type

        # [Denoising Transformer]
        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # blocks
        self.blocks = nn.ModuleList([Incontext_AttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, refer_stride, use_img_emb=use_img_emb, sparse_type=self.sparse_type
        ) for _ in range(num_layers)])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        if use_img_emb:
            self.img_emb = MLP(1280, dim, has_pos_emb=False)

        self.block_masks = dict()
        self.block_mask_grid_sizes = dict()

    def create_mask(self, origin_len, origin_area, device):
        origin_latent_f = origin_len // 4 + 1
        hw = int(np.prod(origin_area).item() // 256)

        q_len = (origin_latent_f + 1) * hw
        k_len = origin_latent_f * hw

        q_len_total = math.ceil(q_len / 128) * 128
        k_extra_len_total = math.ceil(k_len / 128) * 128
        k_len_total = q_len_total + k_extra_len_total

        q_limit = q_len
        k_limit = k_len
        q_total = q_len_total

        def attention_mask_logic(b, h, q_idx, kv_idx):
            q_valid = q_idx < q_limit
            is_base_attention = kv_idx < q_limit

            q_frame = q_idx // hw
            is_first_part = kv_idx < q_total

            kv_frame_1 = kv_idx // hw
            kv_is_valid_1 = kv_idx < q_limit

            rel_kv_idx = kv_idx - q_total
            kv_frame_2 = (rel_kv_idx // hw) + 1
            kv_is_valid_2 = rel_kv_idx < k_limit

            kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
            kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)

            is_cond_attention = (q_frame == kv_frame) & kv_is_valid

            return q_valid & (is_base_attention | is_cond_attention)

        block_mask = create_block_mask(
            attention_mask_logic,
            B=None,
            H=None,
            Q_LEN=q_len_total,
            KV_LEN=k_len_total,
            device=device,
            _compile=True
        )
        return block_mask

    def forward_ref(
        self,
        x_ref,
        grid_sizes,
        k_cache,
        v_cache,
        clip_fea_ref,
        y_ref,
        context_ref,
        seq_len_ref,
        t,
        animate2_offload_kv=False,
        use_unified_sequence_parallel: bool = False,
    ):
        # [reference]
        x_ref = [torch.cat([u, v], dim=0) for u, v in zip(x_ref, y_ref)]
        # embeddings
        x_ref = [self.patch_embedding(u.unsqueeze(0)) for u in x_ref]
        grid_sizes_ref = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in x_ref
        ])
        x_ref = [u.flatten(2).transpose(1, 2) for u in x_ref]
        seq_lens_ref = torch.tensor([u.size(1) for u in x_ref], dtype=torch.long)
        assert seq_lens_ref.max() <= seq_len_ref
        x_ref = torch.cat([torch.cat([
            u, u.new_zeros(1, seq_len_ref - u.size(1), u.size(2))
        ], dim=1) for u in x_ref])

        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads

        if self.refer_offset_t < 0:
            self.refer_offset_t = grid_sizes[0][0].item()
        if self.refer_offset_h < 0:
            self.refer_offset_h = grid_sizes[0][1].item()
        if self.refer_offset_w < 0:
            self.refer_offset_w = grid_sizes[0][2].item()

        self.freqs_ref = torch.cat([
            rope_params(512, d - 4 * (d // 6), offset=self.refer_offset_t),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_h),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_w)
        ], dim=1).to(x_ref.device)

        # time embeddings ref
        e_ref = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t*0+1).to(x_ref.dtype))
        e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim))

        # [context_ref]
        context_ref = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context_ref]))

        if self.use_img_emb:
            context_clip_ref = self.img_emb(clip_fea_ref) # bs x 257 x dim
            context_ref = torch.concat([context_clip_ref, context_ref], dim=1)

        context_lens = None
        # arguments
        kwargs = dict(
            e_ref=e0_ref,
            grid_sizes_ref=grid_sizes_ref,
            freqs_ref=self.freqs_ref,
            context_ref=context_ref,
            context_lens=context_lens,
            animate2_offload_kv=animate2_offload_kv,
            use_context_parallel=use_unified_sequence_parallel,
        )
        if use_unified_sequence_parallel:
            from ..utils.xfuser import get_current_chunk, is_evenly_divisible, get_sequence_parallel_world_size
            assert is_evenly_divisible(x_ref.shape[1]), (
                f"x_ref sequence length ({x_ref.shape[1]}) must be divisible by sequence parallel world size ({get_sequence_parallel_world_size()}) "
            )
            x_ref = get_current_chunk(x_ref, dim=1)
        for idx, block in enumerate(self.blocks):
            x_ref = block(x_ref, idx, k_cache, v_cache, method='forward_ref', **kwargs)

    def forward_gen(
        self,
        x,
        k_cache,
        v_cache,
        clip_fea,
        y,
        context,
        seq_len,
        t,
        grid_sizes_ref,
        origin_len,
        origin_area,
        is_uncondtion=False,
        use_unified_sequence_parallel: bool = False,
        log_scale=0.0,
    ):
        # params
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in x
        ])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([
            u, u.new_zeros(1, seq_len - u.size(1), u.size(2))
        ], dim=1) for u in x])

        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(512, d - 4 * (d // 6)),
            rope_params(512, 2 * (d // 6)),
            rope_params(512, 2 * (d // 6))
        ], dim=1).to(x.device)

        if self.refer_offset_t < 0:
            self.refer_offset_t = grid_sizes[0][0].item()
        if self.refer_offset_h < 0:
            self.refer_offset_h = grid_sizes[0][1].item()
        if self.refer_offset_w < 0:
            self.refer_offset_w = grid_sizes[0][2].item()

        self.freqs_ref = torch.cat([
            rope_params(512, d - 4 * (d // 6), offset=self.refer_offset_t),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_h),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_w)
        ], dim=1).to(x.device)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # [context]
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context]))

        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea) # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        block_mask_id = (origin_len, origin_area[0], origin_area[1])
        if block_mask_id not in self.block_masks:
            self.block_masks[block_mask_id] = self.create_mask(origin_len, origin_area, x.device)
        attn_mask = self.block_masks[block_mask_id]

        # arguments
        kwargs = dict(
            e=e0,
            attn_mask=attn_mask,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            grid_sizes_ref=grid_sizes_ref,
            freqs_ref=self.freqs_ref,
            context_lens=context_lens,
            origin_area=origin_area,
            origin_len=origin_len,
            use_context_parallel=use_unified_sequence_parallel,
            log_scale=log_scale,
        )
        if use_unified_sequence_parallel:
            from ..utils.xfuser import get_current_chunk, is_evenly_divisible, get_sequence_parallel_world_size
            assert is_evenly_divisible(x.shape[1]), (
                f"sequence length ({x.shape[1]}) must be divisible by sequence parallel world size ({get_sequence_parallel_world_size()}) "
            )
            x = get_current_chunk(x, dim=1)

        for idx, block in enumerate(self.blocks):
            if is_uncondtion and idx==9:
                continue
            x = block(x, idx, k_cache, v_cache, method='forward_gen', **kwargs)

        # head
        x = self.head(x, e)
        if use_unified_sequence_parallel:
            from ..utils.xfuser import gather_all_chunks
            x = gather_all_chunks(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u for u in x]

    def forward_origin(
        self,
        x,
        clip_fea,
        y,
        context,
        seq_len,
        x_ref,
        clip_fea_ref,
        y_ref,
        context_ref,
        seq_len_ref,
        t,
        origin_len,
        origin_area,
        log_scale=0.0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # params
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in x
        ])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([
            u, u.new_zeros(1, seq_len - u.size(1), u.size(2))
        ], dim=1) for u in x])

        # [reference]
        # params
        x_ref = [torch.cat([u, v], dim=0) for u, v in zip(x_ref, y_ref)]
        # embeddings
        x_ref = [self.patch_embedding(u.unsqueeze(0)) for u in x_ref]
        grid_sizes_ref = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in x_ref
        ])
        x_ref = [u.flatten(2).transpose(1, 2) for u in x_ref]
        seq_lens_ref = torch.tensor([u.size(1) for u in x_ref], dtype=torch.long)
        assert seq_lens_ref.max() <= seq_len_ref
        x_ref = torch.cat([torch.cat([
            u, u.new_zeros(1, seq_len_ref - u.size(1), u.size(2))
        ], dim=1) for u in x_ref])

        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(512, d - 4 * (d // 6)),
            rope_params(512, 2 * (d // 6)),
            rope_params(512, 2 * (d // 6))
        ], dim=1).to(x.device)

        if self.refer_offset_t < 0:
            self.refer_offset_t = grid_sizes[0][0].item()
        if self.refer_offset_h < 0:
            self.refer_offset_h = grid_sizes[0][1].item()
        if self.refer_offset_w < 0:
            self.refer_offset_w = grid_sizes[0][2].item()

        self.freqs_ref = torch.cat([
            rope_params(512, d - 4 * (d // 6), offset=self.refer_offset_t),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_h),
            rope_params(512, 2 * (d // 6), offset=self.refer_offset_w)
        ], dim=1).to(x.device)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # time embeddings ref
        e_ref = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t*0+1).to(x.dtype))
        e0_ref = self.time_projection(e_ref).unflatten(1, (6, self.dim))

        # [context]
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context]))

        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea) # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # [context_ref]
        context_ref = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context_ref]))

        if self.use_img_emb:
            context_clip_ref = self.img_emb(clip_fea_ref) # bs x 257 x dim
            context_ref = torch.concat([context_clip_ref, context_ref], dim=1)

        block_mask_id = (origin_len, origin_area[0], origin_area[1])
        if block_mask_id not in self.block_masks:
            self.block_masks[block_mask_id] = self.create_mask(origin_len, origin_area, x.device)
        attn_mask = self.block_masks[block_mask_id]

        # arguments
        ref_args = dict(
            e_ref=e0_ref,
            grid_sizes_ref=grid_sizes_ref,
            freqs_ref=self.freqs_ref,
            context_ref=context_ref,
            context_lens=context_lens,
        )
        gen_args = dict(
            e=e0,
            attn_mask=attn_mask,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            grid_sizes_ref=grid_sizes_ref,
            freqs_ref=self.freqs_ref,
            context_lens=context_lens,
            origin_area=origin_area,
            origin_len=origin_len,
            log_scale=log_scale,
        )
        for idx, block in enumerate(self.blocks):
            x, x_ref = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, x_ref, ref_args, gen_args, method='forward_origin'
            )

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
