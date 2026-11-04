import torch
import torch.distributed as dist


def all_to_all_4d(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Shard `scatter_dim` across the group and gather `gather_dim` back."""
    return _AllToAll4D.apply(tensor, scatter_dim, gather_dim, group)


def _all_to_all_4d(tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"all_to_all_4d expects a 4-D tensor, got {tensor.dim()}-D")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    if tensor.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"cannot scatter dim {scatter_dim} of size {tensor.shape[scatter_dim]} "
            f"across {world_size} ranks"
        )

    if (scatter_dim, gather_dim) == (2, 1):
        # [B, S/P, H, D] -> [B, S, H/P, D]
        batch, shard_seq, heads, head_dim = tensor.shape
        seq, shard_heads = shard_seq * world_size, heads // world_size
        send = (
            tensor.unflatten(2, (world_size, shard_heads))
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return (
            recv.reshape(seq, batch, shard_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .reshape(batch, seq, shard_heads, head_dim)
        )

    elif (scatter_dim, gather_dim) == (1, 2):
        # [B, S, H/P, D] -> [B, S/P, H, D]
        batch, seq, shard_heads, head_dim = tensor.shape
        heads, shard_seq = shard_heads * world_size, seq // world_size
        send = (
            tensor.unflatten(1, (world_size, shard_seq))
            .permute(1, 3, 2, 0, 4)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return (
            recv.reshape(heads, shard_seq, batch, head_dim)
            .permute(2, 1, 0, 3)
            .contiguous()
            .reshape(batch, shard_seq, heads, head_dim)
        )

    else:
        raise ValueError(f"unsupported exchange ({scatter_dim}, {gather_dim}); expected (2, 1) or (1, 2)")


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scatter_dim, gather_dim, group):
        ctx.group, ctx.scatter_dim, ctx.gather_dim = group, scatter_dim, gather_dim
        return _all_to_all_4d(tensor, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad):
        # The adjoint is the same exchange with the dims swapped.
        return _all_to_all_4d(grad, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None, None


def all_gather_sequence(tensor: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Concatenate sequence shards in group-rank order."""
    return _AllGatherSequence.apply(tensor, dim, group)


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, group):
        ctx.dim = dim if dim >= 0 else dim + tensor.dim()
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        if ctx.world_size == 1:
            return tensor
        tensor = tensor.contiguous()
        buffers = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(buffers, tensor, group=group)
        return torch.cat(buffers, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        # Every rank holds a copy of the output, so all their gradients contribute:
        # reduce_scatter sums the copies and returns this rank's own slice.
        if ctx.world_size == 1:
            return grad, None, None
        perm = list(range(grad.dim()))
        perm[0], perm[ctx.dim] = perm[ctx.dim], perm[0]
        grad = grad.permute(perm).contiguous()
        size = grad.shape[0] // ctx.world_size
        out = torch.empty((size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter_tensor(out, grad, group=ctx.group)
        return out.permute(perm).contiguous(), None, None


def broadcast_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Broadcast a tensor from group rank 0."""
    if dist.get_world_size(group) == 1:
        return tensor
    tensor = tensor.contiguous()
    dist.broadcast(tensor, src=dist.get_global_rank(group, 0), group=group)
    return tensor
