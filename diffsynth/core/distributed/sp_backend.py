"""Sequence-parallel attention backends.

Backends are registered in `SEQUENCE_PARALLEL_BACKENDS`; the group and its
size/rank are looked up on every call, never cached.
"""

from __future__ import annotations

import torch

from ..attention.common import rearrange_out, rearrange_qkv
from .collectives import all_to_all_4d
from .context import get_parallel_group


class UlyssesBackend:
    """Exchange sequence shards for head shards around a local attention call.

    Inside a sharded block stack each rank holds S/P tokens with all H heads. Self-
    attention needs every key, but attention heads are independent of one another,
    so the work can be redistributed by head instead of by token:

        q, k, v  [B, S/P, H, D]
          all-to-all (scatter heads, gather tokens)
                 [B, S, H/P, D]      every token, a subset of heads
          local attention kernel     unchanged; it just sees a full sequence
                 [B, S, H/P, D]
          all-to-all (scatter tokens, gather heads)
        out      [B, S/P, H, D]      back to this rank's tokens, all heads

    The output lines up with the local residual, so the block continues on its
    slice, and the result matches single-device attention mathematically.

    Exchange and local attention both use `"b s n d"`, the layout the SP-enabled
    models call with; callers in another layout are converted on the way in and
    restored on the way out.

    Q, K and V ride in one collective when their shapes match. Concatenating along
    `head_dim` keeps each head's q/k/v adjacent, so the exchange still scatters
    whole heads and the result is bit-identical to three separate exchanges. The
    concatenation costs a few percent while removing two collectives. Tensors whose
    shapes differ are exchanged separately. Both head counts must be divisible by P.
    """

    def _exchange_qkv(self, q, k, v):
        group = get_parallel_group()
        if q.shape == k.shape == v.shape:
            head_size = q.shape[-1]
            merged = torch.cat((q, k, v), dim=-1)
            merged = all_to_all_4d(merged, scatter_dim=2, gather_dim=1, group=group)
            return merged.split(head_size, dim=-1)
        return (
            all_to_all_4d(q, scatter_dim=2, gather_dim=1, group=group),
            all_to_all_4d(k, scatter_dim=2, gather_dim=1, group=group),
            all_to_all_4d(v, scatter_dim=2, gather_dim=1, group=group),
        )

    def forward(self, q, k, v, *, local_attention, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, **attn_kwargs):
        required = "b s n d"
        q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required, dims)
        q, k, v = self._exchange_qkv(q, k, v)

        out = local_attention(q, k, v, required, required, required, required, **attn_kwargs)

        out = all_to_all_4d(out, scatter_dim=1, gather_dim=2, group=get_parallel_group())
        return rearrange_out(out, out_pattern, required, dims)


SEQUENCE_PARALLEL_BACKENDS = {"ulysses": UlyssesBackend}
