"""Ulysses collectives for H3 training with a replicated loss and averaged gradients.

All ranks in the group process the same sample. DDP/ZeRO must average parameter
gradients over that group. This module does not implement data parallel groups.
"""
import torch
import torch.distributed as dist


def _exchange(x, group, to_sequence):
    size = dist.get_world_size(group)
    if to_sequence:
        length, heads, width = x.shape
        send = x.reshape(length, size, heads // size, width).permute(1, 0, 2, 3).contiguous()
    else:
        length, heads, width = x.shape
        send = x.reshape(size, length // size, heads, width).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    if to_sequence:
        return recv.reshape(size * length, heads // size, width)
    return recv.permute(1, 0, 2, 3).reshape(length // size, size * heads, width)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, to_sequence):
        ctx.group, ctx.to_sequence = group, to_sequence
        return _exchange(x, group, to_sequence)

    @staticmethod
    def backward(ctx, grad):
        return _exchange(grad, ctx.group, not ctx.to_sequence), None, None


def _gather(x, group):
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=0)


class _SplitReplicated(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        # A contiguous slice can still retain the full input's storage.
        return x.chunk(dist.get_world_size(group), dim=0)[dist.get_rank(group)].clone()

    @staticmethod
    def backward(ctx, grad):
        # The embedding was evaluated on every rank. Undo the replicated-loss
        # scaling here so its parameter gradients are not multiplied by SP size.
        return _gather(grad, ctx.group) / dist.get_world_size(ctx.group), None


class _GatherReplicatedLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _gather(x, group)

    @staticmethod
    def backward(ctx, grad):
        size = dist.get_world_size(ctx.group)
        # Each rank computes the same full loss. Compensate for the parameter
        # gradient average performed later by DDP/ZeRO.
        return grad.chunk(size, dim=0)[dist.get_rank(ctx.group)].contiguous() * size, None


class MiniMaxH3SequenceParallel:
    def __init__(self, size):
        if not dist.is_initialized() or size < 2 or size != dist.get_world_size():
            raise ValueError("H3 sequence parallel size must equal the initialized world size (>=2).")
        self.group = dist.group.WORLD
        self.size = size
        self.rank = dist.get_rank()

    def split(self, x):
        padding = (-x.shape[0]) % self.size
        if padding:
            x = torch.cat([x, x.new_zeros((padding, *x.shape[1:]))], dim=0)
        return _SplitReplicated.apply(x, self.group)

    def gather(self, x, length):
        return _GatherReplicatedLoss.apply(x, self.group)[:length]

    def attention(self, q, k, v, cu_seqlens, scale, attention_fn):
        q, k, v = [_AllToAll.apply(x, self.group, True) for x in (q, k, v)]
        # Preserve all original packed segments, including the pipeline's pad
        # segment. Extra SP alignment tokens are excluded from attention.
        out = torch.zeros_like(q)
        bounds = cu_seqlens.tolist()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop > start:
                seg = [x[start:stop].transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
                result = attention_fn(*seg, scale=scale)
                out[start:stop] = result.squeeze(0).transpose(0, 1)
        return _AllToAll.apply(out, self.group, False)
