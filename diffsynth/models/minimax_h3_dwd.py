import math, torch
from torch import nn
from torch.nn import functional as F

from ..core.attention import attention_forward


NUM_HEADS = 32
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
FFN_MULT = 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)
WINDOW = 16
BLOCK = 5
T_SHIFT = 2
TIME_NORM = 5
PATCH, PATCH_T, OUT_CH = 16, 4, 3
FRAMES_TO_TRIM = 3


def _rope(T, H, W, s, st, device):
    """cos/sin (1, T*H*W, 1, ROPE_DIM // 2) for tokens in (T, H, W) order with window-local time and space ids
    for shifts (st, s); both halves of the rotary dims share them."""
    lt = ((torch.arange(T, device=device) + st) % BLOCK).float()
    ly = ((torch.arange(H, device=device) + s) % WINDOW).float()
    lx = ((torch.arange(W, device=device) + s) % WINDOW).float()
    t = (lt + 0.5) / TIME_NORM * 2 - 1
    hy = (ly + 0.5) / WINDOW * 2 - 1
    wx = (lx + 0.5) / WINDOW * 2 - 1
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing='ij'), -1).reshape(1, -1, 3)
    inv_freq = 1.0 / ROPE_THETA ** torch.arange(0, 1, 6 / ROPE_DIM, dtype=torch.float32, device=device)
    angles = 2.0 * math.pi * grid[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).unsqueeze(2)
    return angles.cos(), angles.sin()


def _rotate_(x, emb):
    cos, sin = (t.to(x.dtype) for t in emb)
    half = cos.size(-1)
    first, second = x[..., :half], x[..., half:2 * half]
    first_sin, second_sin = first * sin, second * sin
    first.mul_(cos).sub_(second_sin)
    second.mul_(cos).add_(first_sin)


def _partition(x, s, st):
    """x: (B, T, H, W, ...) -> windows (B*nW, BLOCK*WINDOW*WINDOW, ...), valid (nW, BLOCK*WINDOW*WINDOW), info.
    Window order is (time window, row, col)."""
    B, T, H, W = x.shape[:4]
    ws, tb = WINDOW, BLOCK
    rest = x.shape[4:]
    Tp = ((T + st + tb - 1) // tb) * tb
    Hp = ((H + s + ws - 1) // ws) * ws
    Wp = ((W + s + ws - 1) // ws) * ws
    xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
    xp[:, st:st + T, s:s + H, s:s + W] = x
    valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
    valid[st:st + T, s:s + H, s:s + W] = True
    nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
    r = len(rest)
    xw = xp.view(B, nt, tb, nh, ws, nw, ws, *rest).permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r))
    xw = xw.reshape(B * nt * nh * nw, tb * ws * ws, *rest)
    vw = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, tb * ws * ws)
    return xw, vw, (B, T, H, W, Tp, Hp, Wp, nt, nh, nw)


def _pack(rows, valid, total):
    valid = valid.expand(rows.shape).flatten(0, 1)
    order = valid.logical_not().flatten().to(torch.int8).argsort(stable=True)[:total]
    return rows.flatten()[order], F.pad(valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))


def _window_rows(B, T, H, W, nb, S, s, st, device, padded=False):
    N = T * H * W
    tokens, valid, info = _partition(torch.arange(1, N + 1, device=device).view(1, T, H, W), s, st)
    Tp, Hp, Wp, nt, nh, nw = info[4:]
    blocks = [(max(j * BLOCK - st, 0) // BLOCK, (min((j + 1) * BLOCK - st, T) - 1) // BLOCK) for j in range(nt)]
    global_rows = torch.tensor([[b * S + i for b in pair for i in range(S)] for pair in blocks], device=device)
    global_valid = torch.tensor([[True] * S + [b1 != b0] * S for b0, b1 in blocks], device=device)
    global_rows, global_valid = (
        t[:, None, None].expand(nt, nh, nw, 2 * S).reshape(-1, 2 * S)
        for t in (global_rows, global_valid)
    )
    batch = torch.arange(B, device=device)[:, None, None]
    token_rows = batch * N + tokens - 1
    key_rows = torch.cat([token_rows, B * N + batch * nb * S + global_rows], -1)
    key_valid = torch.cat([valid, global_valid], -1)
    if padded:
        output_rows = torch.arange(token_rows.numel(), device=device).view(
            B, nt, nh, nw, BLOCK, WINDOW, WINDOW).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Tp, Hp, Wp)
        output_rows = output_rows[:, st:st + T, s:s + H, s:s + W].reshape(-1)
        return (
            token_rows.masked_fill(~valid, 0).flatten(0, 1),
            key_rows.masked_fill(~key_valid, 0).flatten(0, 1),
            key_valid.expand_as(key_rows).flatten(0, 1).contiguous(), output_rows,
        )
    q_rows, cu_seqlens_q = _pack(token_rows, valid, B * N)
    k_rows, cu_seqlens_k = _pack(
        key_rows, key_valid, B * (N + nh * nw * S * sum(1 + (b1 != b0) for b0, b1 in blocks)))
    return q_rows, k_rows, cu_seqlens_q, cu_seqlens_k


def _global_rows(B, T, H, W, nb, S, device, padded=False):
    N = T * H * W
    tokens = torch.arange(nb * BLOCK * H * W, device=device).view(nb, -1)
    batch = torch.arange(B, device=device)[:, None, None]
    globals_ = torch.arange(nb * S, device=device).view(nb, S)
    rows = torch.cat([batch * N + tokens, B * N + batch * nb * S + globals_], -1)
    valid = torch.cat([tokens < N, torch.ones(nb, S, dtype=torch.bool, device=device)], -1)
    if padded:
        return rows.masked_fill(~valid, 0).flatten(0, 1), valid.expand_as(rows).flatten(0, 1).contiguous()
    rows, cu_seqlens_k = _pack(rows, valid, B * (N + nb * S))
    cu_seqlens_q = torch.arange(B * nb + 1, dtype=torch.int32, device=device) * S
    return rows, cu_seqlens_q, cu_seqlens_k


def _padded(device, dtype):
    return not (
        device.type == "cuda"
        and dtype in (torch.bfloat16, torch.float16)
        and torch.backends.cuda.is_flash_attention_available()
        and torch.cuda.get_device_capability(device) >= (8, 0)
    )


class SolH3Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.to_out = nn.Linear(DIM, DIM)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=NORM_EPS, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=NORM_EPS, elementwise_affine=False)

    def forward(
        self, query, key, value,
        cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None,
        attn_mask=None,
    ):
        if attn_mask is not None:
            return attention_forward(
                query, key, value, q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d",
                out_pattern="b s n d", attn_mask=attn_mask, compatibility_mode=True,
            )
        return torch.ops.aten._flash_attention_forward(
            query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, False, False,
        )[0]


class SolH3SwiGLU(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2((F.silu(gate.float()) * up).to(x.dtype))


class SolH3Block(nn.Module):
    """One Sol decoder layer on the tokens x (B, T, H, W, D) and the per-block global tokens g (B, nb, S, D)."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.norm2 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.scale1 = nn.Parameter(torch.empty(DIM))
        self.scale2 = nn.Parameter(torch.empty(DIM))
        self.attn = SolH3Attention()
        self.ff = SolH3SwiGLU(DIM, DIM * FFN_MULT)

    def forward(self, x, g, shifted, global_indices=None, window_indices=None):
        x, g = self._attention(x, g, shifted, global_indices, window_indices)
        x = x + self.ff(self.norm2(x.float()).to(x.dtype)) * self.scale2
        g = g + self.ff(self.norm2(g.float()).to(g.dtype)) * self.scale2
        return x, g

    def _attention(self, x, g, shifted, global_indices, window_indices):
        B, T, H, W, _ = x.shape
        nb, S = g.shape[1:3]
        N = B * T * H * W
        heads, dh = NUM_HEADS, HEAD_DIM
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        qkv = self.attn.to_qkv(self.norm1(torch.cat([x.flatten(0, 3), g.flatten(0, 2)]).float()).to(x.dtype))
        qkv = qkv.view(-1, heads, 3, dh)
        q = self.attn.norm_q(qkv[:, :, 0].float()).to(x.dtype)
        k = self.attn.norm_k(qkv[:, :, 1].float()).to(x.dtype)
        v = qkv[:, :, 2]
        rope = _rope(T, H, W, s, st, x.device)
        _rotate_(q[:N].view(B, -1, heads, dh), rope)
        _rotate_(k[:N].view(B, -1, heads, dh), rope)
        padded = _padded(x.device, x.dtype)
        if global_indices is None:
            global_indices = _global_rows(B, T, H, W, nb, S, x.device, padded=padded)
        if padded:
            rows, key_mask = global_indices
            og = self.attn(q[N:].unflatten(0, (B * nb, S)), k[rows], v[rows], attn_mask=key_mask[:, None, None, :])
            og = og.flatten(0, 1)
        else:
            rows, cu_seqlens_q, cu_seqlens_k = global_indices
            og = self.attn(q[N:], k[rows], v[rows], cu_seqlens_q, cu_seqlens_k, S, BLOCK * H * W + S)
        if window_indices is None:
            window_indices = _window_rows(B, T, H, W, nb, S, s, st, x.device, padded=padded)
        q_rows, k_rows, *_ = window_indices
        q = q[q_rows]
        k = k[k_rows]
        v = v[k_rows]
        del qkv
        if padded:
            _, _, key_mask, output_rows = window_indices
            o = self.attn(q, k, v, attn_mask=key_mask[:, None, None, :]).flatten(0, 1)
        else:
            _, _, cu_seqlens_q, cu_seqlens_k = window_indices
            o = self.attn(q, k, v, cu_seqlens_q, cu_seqlens_k, BLOCK * WINDOW ** 2, BLOCK * WINDOW ** 2 + 2 * S)
        del q, k, v
        if padded:
            o = o[output_rows]
        else:
            o = torch.empty_like(o).index_copy_(0, q_rows, o)
        x = x + self.attn.to_out(o.flatten(1)).view_as(x) * self.scale1
        g = g + self.attn.to_out(og.flatten(1)).view_as(g) * self.scale1
        return x, g


class SolH3DWDHead(nn.Module):

    def __init__(
        self,
        in_channels,
        src_patch_size,
        first_frame_src_patch_size,
        first_frame_period,
        residual=False,
        pixel_channels=OUT_CH,
        num_layers=8,
    ):
        super().__init__()
        assert not residual, 'SolH3DWDHead requires linear_head=False on the DWD head.'
        _, src_h, src_w = src_patch_size
        assert (
            tuple(src_patch_size) == (PATCH_T, src_h, src_w)
            and src_h % PATCH == 0
            and src_w % PATCH == 0
        ), f"tokens must cover one latent frame and whole Sol tokens, got src_patch_size {src_patch_size}."
        assert (
            first_frame_src_patch_size is not None
            and tuple(first_frame_src_patch_size) == (PATCH_T - FRAMES_TO_TRIM, src_h, src_w)
            and first_frame_period == BLOCK
            and pixel_channels == OUT_CH
        ), "the pixel geometry must follow Sol's frame trimming at every 5-latent block."
        self.expansion = (src_h // PATCH, src_w // PATCH)
        self.feature_expand = nn.Linear(in_channels, DIM * self.expansion[0] * self.expansion[1])
        self.register_tokens = nn.Parameter(torch.empty(1, NUM_REGISTER_TOKENS, DIM))
        self.blocks = nn.ModuleList([SolH3Block() for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(DIM, eps=NORM_EPS)
        self.proj_out = nn.Linear(DIM, OUT_CH * PATCH_T * PATCH * PATCH)
        self.first_frame_proj_out = nn.Linear(DIM, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH)

    def forward(self, features, token_grid_size, global_tokens=None):
        B = features.size(0)
        t, h, w = token_grid_size
        e_h, e_w = self.expansion
        x = self.feature_expand(features.to(self.register_tokens.dtype))
        x = x.view(B, t, h, w, e_h, e_w, DIM).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, t, h * e_h, w * e_w, DIM)
        return self.decode(x, global_tokens)

    def decode(self, x, global_tokens=None):
        """Run the Sol layers on (B, T, H, W, D) activations; returns (B, F, 3, 16H, 16W)."""
        B, T, H, W, _ = x.shape
        if global_tokens is None:
            global_tokens = torch.cat([self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, DIM)], 1)
            global_tokens = global_tokens.unsqueeze(1).expand(B, (T + BLOCK - 1) // BLOCK, -1, -1).contiguous()
        g = global_tokens.to(x.dtype)
        for block, args in self.block_args(x.shape, g.shape, x.device, x.dtype):
            x, g = block(x, g, *args)
        x = self.norm_out(x)
        num_chunks, tail = divmod(T, BLOCK)
        if num_chunks:
            out = self._project_pixels(x[:, :num_chunks * BLOCK].unflatten(1, (num_chunks, BLOCK))).flatten(1, 2)
        if tail:
            end = self._project_pixels(x[:, num_chunks * BLOCK:].unsqueeze(1)).flatten(1, 2)
            out = torch.cat((out, end), dim=1) if num_chunks else end
        return out

    def block_args(self, x_shape, g_shape, device, dtype=torch.bfloat16):
        """Yield each Sol layer and its arguments after (x, g), for (B, T, H, W, D) tokens and (B, nb, S, D) global
        tokens of these shapes and dtype; the attention indices are built once and shared by the layers."""
        B, T, H, W, _ = x_shape
        nb, S = g_shape[1:3]
        global_indices = {}
        window_indices = {}
        for index, block in enumerate(self.blocks, NUM_LAYERS - len(self.blocks)):
            padded = _padded(device, dtype)
            if padded not in global_indices:
                global_indices[padded] = _global_rows(B, T, H, W, nb, S, device, padded=padded)
            shifted = index % 2 == 1
            if (padded, shifted) not in window_indices:
                s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
                window_indices[padded, shifted] = _window_rows(B, T, H, W, nb, S, s, st, device, padded=padded)
            yield block, (shifted, global_indices[padded], window_indices[padded, shifted])

    def _project_pixels(self, x):
        B, n, t, H, W, _ = x.shape
        first = self.first_frame_proj_out(x[:, :, :1])
        first = first.view(B, n, H, W, OUT_CH, PATCH_T - FRAMES_TO_TRIM, PATCH, PATCH).permute(
            0, 1, 5, 4, 2, 6, 3, 7).reshape(B, n, PATCH_T - FRAMES_TO_TRIM, OUT_CH, H * PATCH, W * PATCH)
        if t > 1:
            rest = self.proj_out(x[:, :, 1:])
            rest = rest.view(B, n, t - 1, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(
                0, 1, 2, 6, 5, 3, 7, 4, 8).reshape(B, n, (t - 1) * PATCH_T, OUT_CH, H * PATCH, W * PATCH)
            first = torch.cat((first, rest), dim=2)
        return first


class MiniMaxH3DWDAdaLayerNormOut(nn.Module):

    def __init__(self, hidden_size, adaln_rank, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.linear = nn.Linear(adaln_rank, 2 * hidden_size)

    def forward(self, features, timestep_embedding):
        scale, shift = self.linear(F.silu(timestep_embedding)).chunk(2, dim=-1)
        return self.norm(features) * (1 + scale) + shift


class MiniMaxH3DWD(nn.Module):

    def __init__(self, hidden_size=5376, adaln_rank=8):
        super().__init__()
        self.dwd_norm_out = MiniMaxH3DWDAdaLayerNormOut(hidden_size, adaln_rank)
        self.dwd_pixel_refiner = SolH3DWDHead(hidden_size, (4, 32, 32), (1, 32, 32), 5)

    def forward(self, features, timestep_embedding, token_grid_size):
        features = self.dwd_norm_out(features, timestep_embedding)
        pixels = self.dwd_pixel_refiner(features.unsqueeze(0), token_grid_size).transpose(1, 2).float()
        mean = pixels.new_tensor((-0.03, -0.088, -0.188)).view(1, -1, 1, 1, 1)
        std = pixels.new_tensor((0.458, 0.448, 0.45)).view(1, -1, 1, 1, 1)
        return (pixels.mul_(std).add_(mean).clamp_(-1, 1) + 1) * 0.5
