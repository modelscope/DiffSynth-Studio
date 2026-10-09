from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.attention import attention_forward
from .minimax_h3_dit import MiniMaxH3DiT, _apply_rope, _sdpa_varlen_attention

MINIMAX_H3_VDN_TEXT_STATE_SCALE = 0.5
MINIMAX_H3_VDN_ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")
MINIMAX_H3_VDN_SHORT_CONV_TARGETS = ("q", "k", "v")
MINIMAX_H3_VDN_BRIDGE_MODES = ("alpha", "none")
MINIMAX_H3_VDN_DELTA_RULES = ("vdn_solve",)
MINIMAX_H3_VDN_SOFTMAX_IMPLS = ("auto", "ref", "flex", "fa4", "decomposed")
# The cards FA4 ships kernels for: sm90 (Hopper), sm100 and sm110 (data-center
# Blackwell). Ampere, Ada and consumer Blackwell are not among them, whatever their
# capability number, so the fa4 gate tests membership here, never `>=`.
MINIMAX_H3_VDN_FA4_MAJORS = (9, 10, 11)

_MINIMAX_H3_VDN_FLASH_GC_DONE = [False]


def mini_max_h3_vdn_has_fa4_kernels(device):
    if not torch.cuda.is_available():
        return False
    resolved = torch.device(device)
    return resolved.type == "cuda" and torch.cuda.get_device_capability(resolved)[0] in MINIMAX_H3_VDN_FA4_MAJORS


def _mini_max_h3_vdn_collect_after_first_flash():
    # The first FLASH/flex kernel leaves the block-mask build's garbage behind; one
    # process-wide gc right after it reclaims it before the linear branch peaks.
    if not _MINIMAX_H3_VDN_FLASH_GC_DONE[0]:
        _MINIMAX_H3_VDN_FLASH_GC_DONE[0] = True
        import gc
        gc.collect()


@dataclass(frozen=True)
class MiniMaxH3VDNLayout:
    seq_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int = 0
    frame_width: int = 0
    text_start: int = 0
    text_len: int = 0

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def text_range(self):
        if not self.text_len:
            raise ValueError("this layout carries no text rows; pass text_start/text_len")
        return self.text_start, self.text_start + self.text_len

    @property
    def frame_size(self):
        if not (self.frame_height and self.frame_width):
            raise ValueError("this layout carries no spatial grid; pass frame_size=(H, W)")
        if self.frame_height * self.frame_width != self.tokens_per_frame:
            raise ValueError(f"grid {self.frame_height}x{self.frame_width} != {self.tokens_per_frame} tokens/frame")
        return self.frame_height, self.frame_width

    def global_index(self, device):
        idx = torch.arange(self.seq_len, device=device)
        return torch.cat([idx[: self.video_start], idx[self.video_end:]])


def mini_max_h3_vdn_layout_from_indices(video_indices, num_latent_frames, tokens_per_frame, seq_len, frame_size=None, text_indices=None):
    video_start = int(video_indices[0].item())
    if int(video_indices[-1].item()) - video_start + 1 != video_indices.numel():
        raise ValueError("video rows are not contiguous in the packed sequence")
    if video_indices.numel() != num_latent_frames * tokens_per_frame:
        raise ValueError(f"{video_indices.numel()} video rows != {num_latent_frames} frames x {tokens_per_frame} tokens/frame")
    height, width = frame_size or (0, 0)
    if frame_size and height * width != tokens_per_frame:
        raise ValueError(f"frame_size {height}x{width} != {tokens_per_frame} tokens/frame")
    text_start, text_len = 0, 0
    if text_indices is not None and text_indices.numel():
        text_start = int(text_indices[0].item())
        if int(text_indices[-1].item()) - text_start + 1 != text_indices.numel():
            raise ValueError("text rows are not contiguous in the packed sequence")
        text_len = int(text_indices.numel())
    return MiniMaxH3VDNLayout(seq_len=seq_len, video_start=video_start, num_frames=num_latent_frames,
                              tokens_per_frame=tokens_per_frame, frame_height=height, frame_width=width,
                              text_start=text_start, text_len=text_len)


def mini_max_h3_vdn_window_bounds(num_frames, radius, chunk=0):
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1) for t in range(num_frames)]


class MiniMaxH3VDNOutputGate(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim=None, bottleneck=None, init_value=0.9, init="constant"):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.init_value = init_value
        out_features = num_heads * (head_dim or 1)
        self.down = None if bottleneck is None else nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck or hidden_size, out_features, bias=True)
        if init == "constant":
            nn.init.zeros_(self.up.weight)
            nn.init.constant_(self.up.bias, math.log(init_value / (1.0 - init_value)))
        elif init != "random":
            raise ValueError(f"OutputGate init must be 'constant' or 'random', got {init!r}")

    def forward(self, x):
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.num_heads, self.head_dim or 1)


class MiniMaxH3VDNRMSNorm(nn.Module):
    # fp32 second-moment accumulation; weight is cast DOWN to x's dtype, never the input up.
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
        return x * torch.rsqrt(ms + self.eps).to(x.dtype) * self.weight.to(x.dtype)


class MiniMaxH3VDNAlpha(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, bottleneck=None):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        bottleneck = bottleneck or head_dim
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, num_heads * head_dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(torch.rand(num_heads * head_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, frame_mean_x):
        # autocast OFF and weights promoted to fp32: alpha is multiplied across every
        # frame, so a bf16 alpha's tail error compounds over the clip.
        with torch.autocast(device_type=frame_mean_x.device.type, enabled=False):
            delta = F.linear(frame_mean_x.float(), self.down.weight.float())
            delta = F.linear(delta, self.up.weight.float())
            delta = delta + self.dt_bias.float()
            scale = torch.exp(self.A_log.float())[:, None]
            delta = delta.view(-1, self.num_heads, self.head_dim)
            return torch.exp(-scale * F.softplus(delta.float()))


def _mini_max_h3_vdn_temporal_shift(x, w, k, pad):
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(k):
        part = xp[dt:dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


_MINIMAX_H3_VDN_TCONV = {"fn": None}


def _mini_max_h3_vdn_temporal_conv(x, w, k, pad):
    if _MINIMAX_H3_VDN_TCONV["fn"] is None:
        _MINIMAX_H3_VDN_TCONV["fn"] = torch.compile(_mini_max_h3_vdn_temporal_shift) if x.is_cuda else _mini_max_h3_vdn_temporal_shift
    return _MINIMAX_H3_VDN_TCONV["fn"](x, w, k, pad)


def _mini_max_h3_vdn_activate(tokens, l2norm):
    x = F.silu(tokens)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if l2norm else x


# Fused temporal-conv + SiLU + L2Norm. The compiled shift spelling still READS EACH
# ELEMENT FIVE TIMES (one full-tensor expression per tap); this kernel tiles over frames
# so the five overlapping taps are served from cache and HBM sees each frame about
# (BLOCK_T + 4) / BLOCK_T times -- roughly 2x over the compiled chain at the real shape.
# Accumulates in fp32 and rounds once, so it is NOT bitwise vs the eager chain (about one
# bf16 ulp, closer to an fp32 reference). Inference only; the 5x5 spatial half stays on
# cudnn, whose NHWC depthwise kernel beats every alternative including a compiled stencil.
_MINIMAX_H3_VDN_TCONV_BLOCK_T = 16


def mini_max_h3_vdn_temporal_conv_activate(x, w, kernel, pad, heads, head_dim, l2norm):
    """x [T, S, C] (CUDA, contiguous), w [C, 5] -> [T*S, heads, head_dim]."""
    import triton
    import triton.language as tl

    if not x.is_cuda:
        raise ValueError("mini_max_h3_vdn_temporal_conv_activate is a Triton kernel; x must be on CUDA")
    if kernel != 5 or pad != 2:
        raise ValueError(f"the kernel unrolls exactly 5 symmetric taps, got kernel={kernel} pad={pad}")
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(f"head_dim must be a power of two >= 16, got {head_dim}")
    T, S_, C_ = x.shape
    if C_ != heads * head_dim:
        raise ValueError(f"C={C_} != heads*head_dim={heads * head_dim}")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("x and w must be contiguous")

    block_t = _MINIMAX_H3_VDN_TCONV_BLOCK_T
    if "kernel" not in _MINIMAX_H3_VDN_TCONV:
        @triton.jit
        def _tconv_act_kernel(X, W, OUT, T, S_, C_, BLOCK_T: tl.constexpr,
                              D_: tl.constexpr, L2: tl.constexpr):
            pid_t = tl.program_id(0)
            pid_s = tl.program_id(1)
            pid_h = tl.program_id(2)
            chan = pid_h * D_ + tl.arange(0, D_)
            rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = rows < T
            acc = tl.zeros((BLOCK_T, D_), dtype=tl.float32)
            for dt in tl.static_range(5):
                r = rows + dt - 2
                ok = valid & (r >= 0) & (r < T)
                v = tl.load(X + (r[:, None] * S_ + pid_s) * C_ + chan[None, :],
                            mask=ok[:, None], other=0.0).to(tl.float32)
                wd = tl.load(W + chan * 5 + dt).to(tl.float32)
                acc += v * wd[None, :]
            y = acc * tl.sigmoid(acc)
            if L2:
                inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
                y = y * inv[:, None]
            tl.store(OUT + (rows[:, None] * S_ + pid_s) * C_ + chan[None, :],
                     y.to(OUT.dtype.element_ty), mask=valid[:, None])

        _MINIMAX_H3_VDN_TCONV["kernel"] = _tconv_act_kernel
    out = torch.empty_like(x)
    _MINIMAX_H3_VDN_TCONV["kernel"][(triton.cdiv(T, block_t), S_, heads)](
        x, w, out, T, S_, C_, BLOCK_T=block_t, D_=head_dim, L2=l2norm,
        num_warps=4, num_stages=2)
    return out.reshape(-1, heads, head_dim)


class MiniMaxH3VDNSepConv(nn.Module):
    KERNEL = 5

    def __init__(self, channels, projs=("k", "v")):
        super().__init__()
        self.projs = tuple(projs)
        k = self.KERNEL
        for name in self.projs:
            sp = nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
            nn.init.normal_(sp.weight, std=(k * k) ** -0.5)
            tm = nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
            nn.init.normal_(tm.weight, std=k ** -0.5)
            setattr(self, f"{name}_sp", sp)
            setattr(self, f"{name}_tm", tm)

    def spatial(self, proj, tokens, num_frames, frame_size):
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        w_sp = getattr(self, f"{proj}_sp").weight
        volume = F.conv2d(volume, w_sp, padding=self.KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        w_tm = getattr(self, f"{proj}_tm").weight.squeeze(1).to(x.dtype)
        return x, w_tm

    def apply(self, proj, tokens, num_frames, frame_size):
        if proj not in self.projs:
            return tokens
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = self.spatial(proj, tokens, num_frames, frame_size)
        out = _mini_max_h3_vdn_temporal_conv(x, w_tm, self.KERNEL, self.KERNEL // 2)
        return out.reshape(-1, heads, head_dim)


def _mini_max_h3_vdn_feature_one(conv, tokens, proj, num_frames, frame_size, use_conv=True,
                                 inference=False):
    conv = conv if use_conv else None
    l2norm = proj != "v"
    if conv is not None and proj in conv.projs:
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size)
        if inference and x.is_cuda:
            return mini_max_h3_vdn_temporal_conv_activate(
                x, w_tm, conv.KERNEL, conv.KERNEL // 2, heads, head_dim, l2norm)
        out = _mini_max_h3_vdn_temporal_conv(x, w_tm, conv.KERNEL, conv.KERNEL // 2)
        return _mini_max_h3_vdn_activate(out.reshape(-1, heads, head_dim), l2norm)
    return _mini_max_h3_vdn_activate(tokens, l2norm)


def _mini_max_h3_vdn_factor_apply(alpha, A_raw, B_raw):
    # vdn_solve: S_out = (S_in Diag(alpha) + B)(I + A)^-1, exact Cholesky joint solve.
    A32 = A_raw.float()
    eye = torch.eye(A32.shape[-1], device=A32.device, dtype=torch.float32).expand_as(A32)
    chol = torch.linalg.cholesky(A32 + eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inv = linv.transpose(-1, -2) @ linv
    transition = alpha.unsqueeze(-1) * inv
    injection = B_raw.float() @ inv
    return transition.to(A_raw.dtype), injection.to(B_raw.dtype)


MINIMAX_H3_VDN_STATS_CHUNK_FRAMES = 16


def mini_max_h3_vdn_frame_statistics(kf, vf, beta, a_fp32=True, inference=False):
    # Inference takes the frames in chunks into preallocated A/B so the four prologue
    # operands exist for one chunk rather than the whole clip (8 GiB -> 1.3 GiB at 345
    # frames); the per-frame GEMMs are unchanged either way.
    num_frames = kf.shape[0]
    if not inference or num_frames <= MINIMAX_H3_VDN_STATS_CHUNK_FRAMES:
        return _mini_max_h3_vdn_frame_statistics_pass(kf, vf, beta, a_fp32)
    heads, dk, dv = kf.shape[1], kf.shape[-1], vf.shape[-1]
    A = kf.new_empty((num_frames, heads, dk, dk), dtype=torch.float32)
    B = kf.new_empty((num_frames, heads, dv, dk), dtype=torch.float32)
    for start in range(0, num_frames, MINIMAX_H3_VDN_STATS_CHUNK_FRAMES):
        stop = min(start + MINIMAX_H3_VDN_STATS_CHUNK_FRAMES, num_frames)
        A[start:stop], B[start:stop] = _mini_max_h3_vdn_frame_statistics_pass(
            kf[start:stop], vf[start:stop], beta[start:stop], a_fp32)
    return A, B


def _mini_max_h3_vdn_frame_statistics_pass(kf, vf, beta, a_fp32):
    # autocast OFF at the OP level: an ambient bf16 autocast would re-downcast the
    # operands and silently turn the fp32 A back into a bf16 one.
    with torch.autocast(device_type=kf.device.type, enabled=False):
        kf16 = kf.contiguous()
        kf32 = kf16.float()
        scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
        vb = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
        if a_fp32:
            A = torch.matmul(scaled32.transpose(-1, -2), kf32)
        else:
            A = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous().transpose(-1, -2), kf).float()
        # bf16 rounding makes A symmetric only to bf16 precision; on real activations that
        # pushes lambda_min(I+A) below 1 and cholesky (lower triangle only) then fails.
        A = 0.5 * (A + A.transpose(-1, -2))
        B = torch.matmul(vb.transpose(-1, -2), kf).float()
        return A, B


def _mini_max_h3_vdn_run_scans(alpha, A_raw, B_raw, text_state=None):
    # autocast OFF: bf16 would reach cuSOLVER, which has no bf16 Cholesky kernel.
    with torch.autocast(device_type=A_raw.device.type, enabled=False):
        transitions, injections = _mini_max_h3_vdn_factor_apply(alpha, A_raw, B_raw)
        num_frames = transitions.shape[0]
        start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
        prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype, device=injections.device)
        suffix = torch.empty_like(prefix)
        state = start
        for frame in range(num_frames):
            torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]
        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


def mini_max_h3_vdn_gather_linear_state(prefix_states, suffix_states, alpha, bounds, bridge="alpha", text_state=None, out_dtype=None):
    if bridge not in MINIMAX_H3_VDN_BRIDGE_MODES:
        raise ValueError(f"bridge={bridge!r}; expected one of {MINIMAX_H3_VDN_BRIDGE_MODES}")
    num_frames = prefix_states.shape[0]
    device = prefix_states.device
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    before_idx = last_before.clamp(min=0)
    after_idx = first_after.clamp(max=num_frames - 1)
    has_before = last_before >= 0
    has_after = first_after < num_frames
    bridge_before = (last_before + 1).clamp(min=0)
    bridge_after = first_after.clamp(max=num_frames)
    frames = torch.arange(num_frames, device=device)

    state_before = prefix_states[before_idx]
    state_after = suffix_states[after_idx]
    if text_state is not None:
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, text_state)
    if bridge == "alpha":
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_alpha_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        alpha_from_before = torch.exp(log_alpha_prefix[frames + 1] - log_alpha_prefix[bridge_before])
        alpha_from_after = torch.exp(log_alpha_prefix[bridge_after] - log_alpha_prefix[frames])
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    if text_state is not None:
        out = state_before + state_after
    else:
        out = state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(-1, 1, 1, 1)
    return out if out_dtype is None else out.to(out_dtype)


def mini_max_h3_vdn_window_softmax_reference(query, key, value, layout, bounds, scale, anchor_frames="none"):
    heads, head_dim = query.shape[1], query.shape[2]
    video_start, video_end = layout.video_start, layout.video_end
    num_frames, tokens_per_frame = layout.num_frames, layout.tokens_per_frame
    global_idx = layout.global_index(query.device)
    out = torch.empty_like(query)

    def sdpa(q_rows, k_rows, v_rows):
        attended = F.scaled_dot_product_attention(
            q_rows.permute(1, 0, 2).unsqueeze(0), k_rows.permute(1, 0, 2).unsqueeze(0),
            v_rows.permute(1, 0, 2).unsqueeze(0), scale=scale)
        return attended.squeeze(0).permute(1, 0, 2)

    if global_idx.numel():
        out[global_idx] = sdpa(query[global_idx], key, value)

    global_key, global_value = key[global_idx], value[global_idx]
    frame_shape = (num_frames, tokens_per_frame, heads, head_dim)
    video_query = query[video_start:video_end].view(frame_shape)
    video_key = key[video_start:video_end].view(frame_shape)
    video_value = value[video_start:video_end].view(frame_shape)
    for frame in range(num_frames):
        lo = max(bounds[frame][0], 0)
        hi = min(bounds[frame][1], num_frames - 1)
        if anchor_frames in ("rows", "both") and frame in (0, num_frames - 1):
            lo, hi = 0, num_frames - 1
        extra = [f for f in ((0, num_frames - 1) if anchor_frames in ("columns", "both") else ()) if not lo <= f <= hi]
        extra_key = [video_key[f].reshape(-1, heads, head_dim) for f in extra]
        extra_value = [video_value[f].reshape(-1, heads, head_dim) for f in extra]
        window_key = torch.cat([global_key, video_key[lo:hi + 1].reshape(-1, heads, head_dim)] + extra_key)
        window_value = torch.cat([global_value, video_value[lo:hi + 1].reshape(-1, heads, head_dim)] + extra_value)
        row_start = video_start + frame * tokens_per_frame
        out[row_start:row_start + tokens_per_frame] = sdpa(video_query[frame], window_key, window_value)
    return out


# ---------------------------------------------------------------------------------------
# Decomposed window softmax: the window as a UNION OF DENSE attentions -- no BlockMask,
# no sparse kernel. The c1 mask is not arbitrary sparsity: every kept pair lies in one of
# a few dense rectangles, so splitting the query rows into groups whose kept KV set is
# identical turns each group into a plain dense attention.
#
# This is what the target library's `auto` resolves to on every CUDA device: the window
# leg needs only SOME varlen kernel, not FA4's -- `mini_max_h3_vdn_varlen_kernel` picks
# FA4's CuTe kernel where it has one and torch's own `varlen_attn` elsewhere. It is also
# the only backend that scales past the BlockMask: create_block_mask needs an O(S^2)
# intermediate, which at 345 frames (S ~ 105k) is ~80 GiB. Inference only -- training
# keeps the flex path.
# ---------------------------------------------------------------------------------------
_MINIMAX_H3_VDN_PLAN_CACHE = {}
_MINIMAX_H3_VDN_MAX_CACHED_PLANS = 4


_MINIMAX_H3_VDN_VARLEN_CACHE = {}


def _mini_max_h3_vdn_fa4_varlen():
    from flash_attn.cute.interface import flash_attn_varlen_func

    def fa4(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                     max_seqlen_q=max_q, max_seqlen_k=max_k, softmax_scale=scale)
        return out[0] if isinstance(out, tuple) else out

    return fa4


def _mini_max_h3_vdn_torch_varlen():
    from torch.nn.attention.varlen import varlen_attn

    def torch_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        return varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale)

    return torch_varlen


def _mini_max_h3_vdn_fa4_installed():
    import importlib.util
    return (importlib.util.find_spec("flash_attn") is not None
            and importlib.util.find_spec("flash_attn.cute") is not None)


def mini_max_h3_vdn_varlen_kernel(device):
    """The varlen kernel the window leg runs on, resolved once per process: FA4's CuTe
    kernel where both the card (sm90/sm100/sm110) and the install qualify, torch's own
    `varlen_attn` (the FA2 lineage, sm80 and up) otherwise. An ImportError here is the
    install to fix: flash-attn-4 or torch >= 2.13.

    The target library gates on the card alone, so on an FA4 card without flash-attn-4 it
    raises; here the check also covers the install, which is what its own commit message
    describes ("on sm8x or without flash-attn-4, torch's own varlen_attn")."""
    if "varlen" not in _MINIMAX_H3_VDN_VARLEN_CACHE:
        _MINIMAX_H3_VDN_VARLEN_CACHE["varlen"] = (
            _mini_max_h3_vdn_fa4_varlen()
            if mini_max_h3_vdn_has_fa4_kernels(device) and _mini_max_h3_vdn_fa4_installed()
            else _mini_max_h3_vdn_torch_varlen())
    return _MINIMAX_H3_VDN_VARLEN_CACHE["varlen"]


class _MiniMaxH3VDNPlan:
    __slots__ = ("dense_q", "win_q", "kv_gather", "cu_q", "cu_k", "max_q", "max_k",
                 "has_windows")

    def __init__(self, layout, bounds, anchor_frames, device):
        S = layout.seq_len
        F, TPF = layout.num_frames, layout.tokens_per_frame
        vs, ve = layout.video_start, layout.video_end
        anchor_set = {0, F - 1} if anchor_frames in ("columns", "rows", "both") else set()
        dense_row_frames = anchor_set if anchor_frames in ("rows", "both") else set()
        dense_col_frames = anchor_set if anchor_frames in ("columns", "both") else set()

        def frame_rows(f):
            return (vs + f * TPF, vs + (f + 1) * TPF)

        global_ranges = [r for r in ((0, vs), (ve, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            return torch.cat([torch.arange(a, b, device=device) for a, b in ranges])

        dense_ranges = merge(global_ranges + [frame_rows(f) for f in sorted(dense_row_frames)])
        self.dense_q = (cat_ranges(dense_ranges) if dense_ranges
                        else torch.empty(0, dtype=torch.long, device=device))

        groups = []
        for f in range(F):
            if f in dense_row_frames:
                continue
            if groups and bounds[groups[-1][-1]] == bounds[f] and groups[-1][-1] == f - 1:
                groups[-1].append(f)
            else:
                groups.append([f])

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(max(lo, 0), min(hi + 1, F))) | dense_col_frames)
            q_r = merge([frame_rows(f) for f in frames])
            kv_r = merge(global_ranges + [frame_rows(f) for f in kv_frames])
            qi, ki = cat_ranges(q_r), cat_ranges(kv_r)
            q_idx.append(qi)
            kv_idx.append(ki)
            q_lens.append(len(qi))
            k_lens.append(len(ki))

        self.has_windows = bool(groups)
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(device, torch.int32)
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(device, torch.int32)
            self.max_q, self.max_k = max(q_lens), max(k_lens)
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)

        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")


def _mini_max_h3_vdn_plan(layout, bounds, anchor_frames, device):
    key = (layout.seq_len, layout.video_start, layout.num_frames,
           layout.tokens_per_frame, tuple(bounds), anchor_frames, str(device))
    if key not in _MINIMAX_H3_VDN_PLAN_CACHE:
        _MINIMAX_H3_VDN_PLAN_CACHE[key] = _MiniMaxH3VDNPlan(layout, bounds, anchor_frames, device)
        while len(_MINIMAX_H3_VDN_PLAN_CACHE) > _MINIMAX_H3_VDN_MAX_CACHED_PLANS:
            _MINIMAX_H3_VDN_PLAN_CACHE.pop(next(iter(_MINIMAX_H3_VDN_PLAN_CACHE)))
    return _MINIMAX_H3_VDN_PLAN_CACHE[key]


def mini_max_h3_vdn_window_softmax_decomposed(query, key, value, layout, bounds, scale,
                                              anchor_frames="none"):
    """[T, H, d] q/k/v -> [T, H, d], exactly the window+anchors+globals mask, as dense
    calls scatter-written straight into a contiguous output (the row sets are disjoint and
    cover [0, T)). Strided k/v are copied contiguous up front: FA4 mis-addresses
    slice-strided operands on sm100, and gathers from a strided source are slower anyway.
    The dense-q leg runs on cuDNN SDPA, which is faster than the varlen kernel at this
    shape."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    varlen = mini_max_h3_vdn_varlen_kernel(query.device)
    plan = _mini_max_h3_vdn_plan(layout, bounds, anchor_frames, query.device)
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if len(plan.dense_q):
        qd = query[plan.dense_q]
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            od = F.scaled_dot_product_attention(
                qd.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0), scale=scale)
        out[plan.dense_q] = od[0].transpose(0, 1)
    if plan.has_windows:
        ow = varlen(query[plan.win_q], key[plan.kv_gather], value[plan.kv_gather],
                    plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, scale)
        out[plan.win_q] = ow
    return out


_MINIMAX_H3_VDN_BLOCK_MASK_CACHE = {}

# ---------------------------------------------------------------------------------------
# Fused inference kernels for the VDN attention path. Same pattern as the block/FF fusions
# in minimax_h3_dit.py: torch.compile(dynamic=False) + module-level cache. NOT bitwise.
# ---------------------------------------------------------------------------------------
_VDN_FUSED_CACHE: dict = {}


def _compiled_vdn(name: str, body):
    if name not in _VDN_FUSED_CACHE:
        _VDN_FUSED_CACHE[name] = torch.compile(body, dynamic=False)
    return _VDN_FUSED_CACHE[name]


def _qk_prep_body(t, weight, eps, freqs):
    """RMSNorm + RoPE as one expression. t: [T, H, d], freqs: [T, rot_dim]."""
    t = nn.functional.rms_norm(t, (t.shape[-1],), weight, eps)
    rot_dim = freqs.shape[-1]
    x_rot, x_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos = torch.cos(freqs).to(t.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(t.dtype).unsqueeze(1)
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)


def _qk_prep_fused(t, weight, eps, freqs):
    return _compiled_vdn("qk_prep", _qk_prep_body)(t, weight, eps, freqs)


def _softmax_gate_body(softmax_out, gate):
    return (softmax_out * gate.to(softmax_out.dtype)).reshape(softmax_out.shape[0], -1)


def _softmax_gate_fused(softmax_out, gate):
    return _compiled_vdn("softmax_gate", _softmax_gate_body)(softmax_out, gate)


def _linear_epilogue_body(readout, weight, eps, gate):
    """RMSNorm + output gate + flatten, one pass. readout: [F, S, H, d] (our einsum's
    output order, already token-major inside each frame)."""
    ms = torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * weight.to(readout.dtype)
    rows = normed.shape[0] * normed.shape[1]
    return normed.reshape(rows, -1) * gate.reshape(rows, -1)


def _linear_epilogue_fused(readout, weight, eps, gate):
    return _compiled_vdn("linear_epilogue", _linear_epilogue_body)(readout, weight, eps, gate)


def mini_max_h3_vdn_build_window_block_mask(layout, bounds, device, anchor_frames="none"):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (layout.seq_len, layout.video_start, layout.num_frames, layout.tokens_per_frame,
           tuple(bounds), anchor_frames, str(device))
    cached = _MINIMAX_H3_VDN_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    vs, ve = layout.video_start, layout.video_end
    num_frames = layout.num_frames
    tokens_per_frame = layout.tokens_per_frame
    lo_t = torch.tensor([max(lo, 0) for lo, _ in bounds], device=device)
    hi_t = torch.tensor([min(hi, num_frames - 1) for _, hi in bounds], device=device)
    anchor_rows = anchor_frames in ("rows", "both")
    anchor_cols = anchor_frames in ("columns", "both")

    def mask_mod(b, h, q_idx, kv_idx):
        q_video = (q_idx >= vs) & (q_idx < ve)
        k_video = (kv_idx >= vs) & (kv_idx < ve)
        dense = ~(q_video & k_video)
        # mask_mod is also evaluated on global rows, where the query-side division runs out
        # of range; those rows are dense anyway, so the clamp only keeps the gather legal.
        qf = torch.clamp(torch.div(q_idx - vs, tokens_per_frame, rounding_mode="floor"), 0, num_frames - 1)
        kf = torch.div(kv_idx - vs, tokens_per_frame, rounding_mode="floor")
        in_window = (kf >= lo_t[qf]) & (kf <= hi_t[qf])
        anchor_row = anchor_rows & ((qf == 0) | (qf == num_frames - 1))
        anchor_col = anchor_cols & ((kf == 0) | (kf == num_frames - 1)) & ~in_window
        return dense | anchor_row | in_window | anchor_col

    mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=layout.seq_len, KV_LEN=layout.seq_len, device=device)
    if len(_MINIMAX_H3_VDN_BLOCK_MASK_CACHE) >= 64:
        _MINIMAX_H3_VDN_BLOCK_MASK_CACHE.clear()
    _MINIMAX_H3_VDN_BLOCK_MASK_CACHE[key] = mask
    return mask


class MiniMaxH3VDNLinearBranch(nn.Module):
    TEXT_STATE_SCALE = MINIMAX_H3_VDN_TEXT_STATE_SCALE

    def __init__(self, hidden_size, num_heads, head_dim, delta_rule="vdn_solve", gate_bottleneck=None,
                 bridge="alpha", a_fp32=True, short_conv=()):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        if bridge not in MINIMAX_H3_VDN_BRIDGE_MODES:
            raise ValueError(f"bridge={bridge!r}; expected one of {MINIMAX_H3_VDN_BRIDGE_MODES}")
        if delta_rule not in MINIMAX_H3_VDN_DELTA_RULES:
            raise ValueError(f"delta_rule={delta_rule!r}; expected one of {MINIMAX_H3_VDN_DELTA_RULES}")
        self.bridge = bridge
        self.delta_rule = delta_rule
        self.a_fp32 = a_fp32
        targets = tuple(short_conv) if isinstance(short_conv, (list, tuple)) else None
        if targets is None or any(t not in MINIMAX_H3_VDN_SHORT_CONV_TARGETS for t in targets) or len(set(targets)) != len(targets):
            raise ValueError(f"short_conv={short_conv!r}; expected a distinct subset of {MINIMAX_H3_VDN_SHORT_CONV_TARGETS}")
        self.short_conv = MiniMaxH3VDNSepConv(num_heads * head_dim, targets) if targets else None
        self.alpha = MiniMaxH3VDNAlpha(hidden_size, num_heads, head_dim)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.output_gate = MiniMaxH3VDNOutputGate(hidden_size, num_heads, head_dim, bottleneck=gate_bottleneck or head_dim, init="random")
        self.norm = MiniMaxH3VDNRMSNorm(head_dim)
        self.inference_mode = False

    def _features(self, qkv_raw, num_frames, frame_size):
        if self.short_conv is not None and frame_size is None:
            raise ValueError("short_conv needs the spatial grid; pass frame_size=(H, W)")
        return tuple(_mini_max_h3_vdn_feature_one(self.short_conv, tokens, proj, num_frames, frame_size,
                                                  inference=self.inference_mode)
                     for proj, tokens in zip(("q", "k", "v"), qkv_raw))

    def _text_chunk_state(self, text_x, text_qkv_raw):
        head_dim = self.head_dim
        length = text_qkv_raw[1].shape[0]
        key = _mini_max_h3_vdn_feature_one(None, text_qkv_raw[1], "k", None, None, use_conv=False)
        value = _mini_max_h3_vdn_feature_one(None, text_qkv_raw[2], "v", None, None, use_conv=False)
        key = key.view(1, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x)).view(1, length, self.num_heads).permute(0, 2, 1)
        A, B = mini_max_h3_vdn_frame_statistics(key, value, beta, a_fp32=self.a_fp32)
        with torch.autocast(device_type=A.device.type, enabled=False):
            ones = torch.ones(1, self.num_heads, head_dim, device=A.device, dtype=A.dtype)
            _, injection = _mini_max_h3_vdn_factor_apply(ones, A, B)
        return injection[0]

    def forward(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size=None,
                skip_ends=False, text_x=None, text_qkv_raw=None):
        ref = xv
        if not skip_ends:
            return self._readout(xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size, text_x, text_qkv_raw)
        if num_frames <= 2:
            return ref.new_zeros(num_frames * tokens_per_frame, self.num_heads * self.head_dim)
        inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
        readout = self._readout(xv[inner], num_frames - 2, tokens_per_frame,
                                [(lo - 1, hi - 1) for lo, hi in bounds[1:num_frames - 1]],
                                tuple(t[inner] for t in qkv_raw), frame_size, text_x, text_qkv_raw)
        out = readout.new_empty(num_frames * tokens_per_frame, readout.shape[-1])
        out[:tokens_per_frame].zero_()
        out[(num_frames - 1) * tokens_per_frame:].zero_()
        out[inner] = readout
        return out

    def _readout(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size=None, text_x=None, text_qkv_raw=None):
        num_tokens = num_frames * tokens_per_frame
        heads, head_dim = self.num_heads, self.head_dim
        shape_per_frame = (num_frames, tokens_per_frame, heads, head_dim)

        query, key, value = self._features(qkv_raw, num_frames, frame_size)
        query_by_frame = query.view(shape_per_frame)
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(xv)).view(num_frames, tokens_per_frame, heads).permute(0, 2, 1)

        A, B = mini_max_h3_vdn_frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32, inference=True)
        del key, value, key_by_frame, value_by_frame     # summarised into A and B
        # dtype=fp32 on the mean: xv is bf16, so a bf16 mean loses precision before the
        # fp32 island can recover it.
        alpha = self.alpha(xv.view(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32))

        text_state = None
        if text_x is not None:
            text_state = self.TEXT_STATE_SCALE * self._text_chunk_state(text_x, text_qkv_raw)
        prefix_states, suffix_states = _mini_max_h3_vdn_run_scans(alpha, A, B, text_state=text_state)

        linear_state = mini_max_h3_vdn_gather_linear_state(prefix_states, suffix_states, alpha, bounds,
                                                           bridge=self.bridge, text_state=text_state).to(xv.dtype)
        readout = torch.einsum("fhvk,fshk->fshv", linear_state, query_by_frame)
        gate = self.output_gate(xv)
        if self.inference_mode:
            nw = self.norm.weight.to(device=readout.device, dtype=readout.dtype)
            return _linear_epilogue_fused(readout, nw, self.norm.eps, gate)
        readout = self.norm(readout.reshape(num_tokens, heads, head_dim))
        return (readout * gate).reshape(num_tokens, heads * head_dim)


class MiniMaxH3VDNAttention(nn.Module):
    def __init__(self, orig_attn, hidden_size, delta_rule="vdn_solve", radius=1, chunk=5,
                 enable_softmax_gate=True, linear_head_dim=128, softmax_impl="auto",
                 anchor_frames="both", enable_text_state=True, bridge="alpha", a_fp32=True,
                 short_conv=("k", "v")):
        super().__init__()
        if anchor_frames not in MINIMAX_H3_VDN_ANCHOR_FRAME_MODES:
            raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {MINIMAX_H3_VDN_ANCHOR_FRAME_MODES}")
        if softmax_impl not in MINIMAX_H3_VDN_SOFTMAX_IMPLS:
            raise ValueError(f"softmax_impl={softmax_impl!r}; expected one of {MINIMAX_H3_VDN_SOFTMAX_IMPLS}")
        self.orig = orig_attn
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.radius = radius
        self.chunk = chunk
        self.softmax_impl = softmax_impl
        self.anchor_frames = anchor_frames
        self.enable_text_state = enable_text_state
        self.layout = None
        d_linear = linear_head_dim or self.head_dim
        self.linear_attention = MiniMaxH3VDNLinearBranch(hidden_size, self.num_heads, d_linear, delta_rule=delta_rule,
                                                         bridge=bridge, a_fp32=a_fp32, short_conv=short_conv)
        self.to_out_linear = nn.Linear(self.num_heads * d_linear, hidden_size, bias=False)
        self.enable_softmax_gate = enable_softmax_gate
        if enable_softmax_gate:
            self.softmax_gate = MiniMaxH3VDNOutputGate(hidden_size, self.num_heads, init_value=0.99)
        self.inference_mode = False

    def _qkv(self, x, rope_freqs):
        orig = self.orig
        total = x.shape[0]
        qkv = orig.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim)
        query_raw, key_raw, value = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
        if self.inference_mode and rope_freqs is not None:
            # .to() for the same reason as _fused_block_forward: under VRAM management the
            # norm weights sit on the offload device and the compiled body reads them raw.
            qw = orig.q_norm.weight.to(device=query_raw.device, dtype=query_raw.dtype)
            kw = orig.k_norm.weight.to(device=key_raw.device, dtype=key_raw.dtype)
            query = _qk_prep_fused(query_raw, qw, orig.q_norm.eps, rope_freqs)
            key = _qk_prep_fused(key_raw, kw, orig.k_norm.eps, rope_freqs)
        else:
            query = orig.q_norm(query_raw)
            key = orig.k_norm(key_raw)
            if rope_freqs is not None:
                query = _apply_rope(query, rope_freqs)
                key = _apply_rope(key, rope_freqs)
        return query, key, value, (query_raw, key_raw, value)

    def _resolved_impl(self, device):
        """The backend that will actually run: `auto` follows the target library's
        resolution -- decomposed on any CUDA device that has a varlen kernel, flex on a
        machine without CUDA."""
        if self.softmax_impl != "auto":
            return self.softmax_impl
        return "decomposed" if torch.cuda.is_available() else "flex"

    def _window_softmax(self, query, key, value, layout, bounds, scale):
        impl = self._resolved_impl(value.device)
        if impl == "ref":
            return mini_max_h3_vdn_window_softmax_reference(query, key, value, layout, bounds, scale,
                                                            anchor_frames=self.anchor_frames)
        if impl == "decomposed":
            return mini_max_h3_vdn_window_softmax_decomposed(query, key, value, layout, bounds, scale,
                                                             anchor_frames=self.anchor_frames)
        block_mask = mini_max_h3_vdn_build_window_block_mask(layout, bounds, value.device, anchor_frames=self.anchor_frames)
        q = query.permute(1, 0, 2).unsqueeze(0)
        k = key.permute(1, 0, 2).unsqueeze(0)
        v = value.permute(1, 0, 2).unsqueeze(0)
        if impl == "fa4":
            import importlib.util
            if importlib.util.find_spec("flash_attn.cute") is None:
                raise ImportError("softmax_impl='fa4' requires flash-attn-4 (flash_attn.cute); it is not installed")
            if not mini_max_h3_vdn_has_fa4_kernels(value.device):
                raise ImportError("softmax_impl='fa4' needs a card FA4 ships kernels for (sm90/sm100/sm110)")
            from torch.nn.attention.flex_attention import flex_attention
            out = torch.compile(flex_attention, dynamic=False)(q, k, v, block_mask=block_mask, scale=scale,
                                                               kernel_options={"BACKEND": "FLASH"})
        else:
            out = attention_forward(q, k, v, attn_mask=block_mask, scale=scale, use_flex=True)
        _mini_max_h3_vdn_collect_after_first_flash()
        return out.squeeze(0).permute(1, 0, 2)

    def forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
        layout = self.layout
        if layout is None:
            return self.orig(x, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        content = layout.seq_len
        xc = x[:content]
        rope_c = rope_freqs[:content] if rope_freqs is not None else None
        bounds = mini_max_h3_vdn_window_bounds(layout.num_frames, self.radius, self.chunk)
        full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
        scale = self.head_dim ** -0.5

        if not full_cover and self._resolved_impl(x.device) in ("flex", "fa4"):
            # Built (once; cached) BEFORE the projections exist: create_block_mask
            # materialises ~10 GiB at 89k rows, and here the layer's live set is x alone
            # rather than x plus the raw and roped q/k/v.
            mini_max_h3_vdn_build_window_block_mask(layout, bounds, x.device, anchor_frames=self.anchor_frames)

        query, key, value, qkv_raw = self._qkv(xc, rope_c)
        if full_cover:
            softmax_out = _sdpa_varlen_attention(query, key, value, cu_seqlens=torch.tensor([0, content], dtype=torch.int32, device=x.device),
                                                 softmax_scale=self.orig.softmax_scale)
            linear_active = False
        else:
            softmax_out = self._window_softmax(query, key, value, layout, bounds, scale)
            linear_active = True
        del query, key, value

        if self.enable_softmax_gate:
            gate = self.softmax_gate(xc)
            flat = _softmax_gate_fused(softmax_out, gate) if self.inference_mode else (softmax_out * gate).reshape(content, -1)
        else:
            flat = softmax_out.reshape(content, -1)
        out = self.orig.out_proj(flat.type_as(xc))
        del softmax_out

        if linear_active:
            video_start, video_end = layout.video_start, layout.video_end
            video_x = xc[video_start:video_end]
            video_qkv_raw = tuple(t[video_start:video_end] for t in qkv_raw)
            text_x = text_qkv_raw = None
            if self.enable_text_state:
                text_start, text_end = layout.text_range
                text_x = xc[text_start:text_end]
                text_qkv_raw = tuple(t[text_start:text_end] for t in qkv_raw)
            linear_readout = self.linear_attention(
                video_x, layout.num_frames, layout.tokens_per_frame, bounds, qkv_raw=video_qkv_raw,
                frame_size=(layout.frame_size if self.linear_attention.short_conv is not None else None),
                skip_ends=self.anchor_frames == "both", text_x=text_x, text_qkv_raw=text_qkv_raw)
            out = out.clone()
            out[video_start:video_end] += self.to_out_linear(linear_readout.type_as(xc))

        full_out = torch.zeros_like(x)
        full_out[:content] = out
        return full_out


class MiniMaxH3VDNBranchBlockAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, delta_rule, bridge, a_fp32, short_conv,
                 enable_softmax_gate, linear_head_dim):
        super().__init__()
        self.to_out_linear = nn.Linear(num_heads * (linear_head_dim or head_dim), hidden_size, bias=False)
        if enable_softmax_gate:
            self.softmax_gate = MiniMaxH3VDNOutputGate(hidden_size, num_heads, init_value=0.99)
        self.linear_attention = MiniMaxH3VDNLinearBranch(hidden_size, num_heads, linear_head_dim or head_dim,
                                                         delta_rule=delta_rule, bridge=bridge, a_fp32=a_fp32,
                                                         short_conv=short_conv)


class MiniMaxH3VDNBranchBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, delta_rule, bridge, a_fp32, short_conv,
                 enable_softmax_gate, linear_head_dim):
        super().__init__()
        self.attn = MiniMaxH3VDNBranchBlockAttention(hidden_size, num_heads, head_dim, delta_rule, bridge, a_fp32,
                                                     short_conv, enable_softmax_gate, linear_head_dim)


class MiniMaxH3VDNBranch(nn.Module):
    def __init__(self, num_layers=50, hidden_size=5376, num_attention_heads=56, attention_head_dim=128,
                 delta_rule="vdn_solve", bridge="alpha", a_fp32=True, short_conv=("k", "v"),
                 enable_softmax_gate=True, linear_head_dim=128):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3VDNBranchBlock(hidden_size, num_attention_heads, attention_head_dim, delta_rule, bridge,
                                     a_fp32, short_conv, enable_softmax_gate, linear_head_dim)
             for _ in range(num_layers)]
        )


class MiniMaxH3DiTVDN(MiniMaxH3DiT):
    _repeated_blocks = ["MiniMaxH3DiTBlock"]

    def __init__(self, vdn_radius=1, vdn_chunk=5, vdn_anchor_frames="both", vdn_delta_rule="vdn_solve",
                 vdn_bridge="alpha", vdn_a_fp32=True, vdn_linear_head_dim=128, vdn_short_conv=("k", "v"),
                 vdn_enable_text_state=True, vdn_enable_softmax_gate=True, vdn_softmax_impl="auto", **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.attn = MiniMaxH3VDNAttention(
                block.attn, hidden_size=self.hidden_size, delta_rule=vdn_delta_rule, radius=vdn_radius,
                chunk=vdn_chunk, enable_softmax_gate=vdn_enable_softmax_gate, linear_head_dim=vdn_linear_head_dim,
                softmax_impl=vdn_softmax_impl, anchor_frames=vdn_anchor_frames, enable_text_state=vdn_enable_text_state,
                bridge=vdn_bridge, a_fp32=vdn_a_fp32, short_conv=vdn_short_conv)

    def forward(self, *args, vdn_layout=None, control_hints=None, **kwargs):
        if control_hints is not None:
            raise ValueError("MiniMaxH3DiTVDN does not support ControlNet; control_hints must be None")
        if vdn_layout is not None:
            for block in self.blocks:
                block.attn.layout = vdn_layout
        return super().forward(*args, control_hints=control_hints, **kwargs)


def load_minimax_h3_vdn_branch(dit, branch):
    if len(branch.blocks) != len(dit.blocks):
        raise ValueError(f"vdn branch has {len(branch.blocks)} layers, dit has {len(dit.blocks)}")

    def unwrapped_state_dict(module):
        # VRAM-managed branches wrap Conv/RMSNorm in AutoWrappedModule, which nests the
        # original module under `.module` and therefore prefixes its state_dict keys.
        return {k.replace(".module.", "."): v for k, v in module.state_dict().items()}

    for dit_block, branch_block in zip(dit.blocks, branch.blocks):
        src = branch_block.attn
        dst = dit_block.attn
        if not isinstance(dst, MiniMaxH3VDNAttention):
            raise ValueError("dit block attention is not a MiniMaxH3VDNAttention; call enable_minimax_h3_vdn first")
        dst.to_out_linear.load_state_dict(unwrapped_state_dict(src.to_out_linear), assign=True)
        if dst.enable_softmax_gate:
            dst.softmax_gate.load_state_dict(unwrapped_state_dict(src.softmax_gate), assign=True)
        dst.linear_attention.load_state_dict(unwrapped_state_dict(src.linear_attention), assign=True)


def enable_minimax_h3_vdn(dit, vdn_branch=None, device=None, **vdn_kwargs):
    for block in dit.blocks:
        block.attn = MiniMaxH3VDNAttention(block.attn, hidden_size=dit.hidden_size, **vdn_kwargs)
    dit.__class__ = MiniMaxH3DiTVDN
    if vdn_branch is not None:
        load_minimax_h3_vdn_branch(dit, vdn_branch)
    if device is not None:
        # VRAM-managed base DiTs only move AutoWrapped modules, so the modules created
        # here (after load) must be pinned to the computation device explicitly.
        for block in dit.blocks:
            block.attn.to_out_linear.to(device)
            if block.attn.enable_softmax_gate:
                block.attn.softmax_gate.to(device)
            block.attn.linear_attention.to(device)
    return dit


def enable_minimax_h3_vdn_fused(dit):
    """Turn on every fused inference kernel: block pointwise, FF SwiGLU, QK-norm+RoPE,
    softmax gate. Idempotent. NOT bitwise vs eager — one rounding at the store instead
    of one per op, in the accurate direction. Default off so the eager path stays the
    correctness reference."""
    from .minimax_h3_dit import enable_minimax_h3_fused_kernels
    enable_minimax_h3_fused_kernels(dit)
    for block in dit.blocks:
        attn = block.attn
        if isinstance(attn, MiniMaxH3VDNAttention):
            attn.inference_mode = True
            attn.linear_attention.inference_mode = True


def disable_minimax_h3_vdn_fused(dit):
    """Revert to the eager path."""
    from .minimax_h3_dit import disable_minimax_h3_fused_kernels
    disable_minimax_h3_fused_kernels(dit)
    for block in dit.blocks:
        attn = block.attn
        if isinstance(attn, MiniMaxH3VDNAttention):
            attn.inference_mode = False
            attn.linear_attention.inference_mode = False
