"""
Basic distributed utilities for multi-GPU support.

Provides initialization and communication primitives.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Any


# Global state
_DISTRIBUTED_INITIALIZED = False
_RANK = 0
_WORLD_SIZE = 1
_LOCAL_RANK = 0
_DEVICE = None


def get_backend_for_device(device_type: str) -> str:
    """Get the appropriate distributed backend for the device type."""
    if device_type == "cuda":
        return "nccl"
    elif device_type == "npu":
        return "hccl"
    else:
        return "gloo"


def init_distributed(
    backend: Optional[str] = None,
    init_method: str = "env://",
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    device_type: str = "cuda",
) -> bool:
    """
    Initialize distributed process group.

    Args:
        backend: Distributed backend ("nccl", "gloo", "hccl"). Auto-detected if None.
        init_method: URL for process group initialization.
        world_size: Total number of processes. Read from env if None.
        rank: Global rank of this process. Read from env if None.
        local_rank: Local rank on this node. Read from env if None.
        device_type: Device type for computation ("cuda", "npu", "cpu").

    Returns:
        True if distributed is successfully initialized, False otherwise.
    """
    global _DISTRIBUTED_INITIALIZED, _RANK, _WORLD_SIZE, _LOCAL_RANK, _DEVICE

    if _DISTRIBUTED_INITIALIZED:
        return True

    # Check if we're in a distributed environment
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Single GPU - no distributed needed
    if world_size == 1:
        _RANK = 0
        _WORLD_SIZE = 1
        _LOCAL_RANK = 0
        if device_type == "cuda" and torch.cuda.is_available():
            _DEVICE = torch.device("cuda:0")
        elif device_type == "mps" and torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
        return False

    # Multi-GPU - initialize distributed
    if backend is None:
        backend = get_backend_for_device(device_type)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )

    _RANK = rank
    _WORLD_SIZE = world_size
    _LOCAL_RANK = local_rank
    _DISTRIBUTED_INITIALIZED = True

    # Set device for this process
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        _DEVICE = torch.device(f"cuda:{local_rank}")
    elif device_type == "npu":
        import torch_npu
        torch.npu.set_device(local_rank)
        _DEVICE = torch.device(f"npu:{local_rank}")
    else:
        _DEVICE = torch.device("cpu")

    return True


def cleanup_distributed():
    """Clean up distributed process group."""
    global _DISTRIBUTED_INITIALIZED
    if dist.is_initialized():
        dist.destroy_process_group()
    _DISTRIBUTED_INITIALIZED = False


def get_rank() -> int:
    """Get the global rank of this process."""
    if dist.is_initialized():
        return dist.get_rank()
    return _RANK


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return _WORLD_SIZE


def get_local_rank() -> int:
    """Get the local rank on this node."""
    return _LOCAL_RANK


def get_device() -> torch.device:
    """Get the device for this process."""
    return _DEVICE


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return _DISTRIBUTED_INITIALIZED and _WORLD_SIZE > 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all ranks."""
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """Reduce tensor across all ranks."""
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all ranks."""
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def all_gather_into_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather tensors from all ranks and concatenate along dim."""
    if not is_distributed():
        return tensor

    gathered = all_gather(tensor)
    return torch.cat(gathered, dim=dim)


def reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Reduce and scatter tensor across ranks."""
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    rank = get_rank()

    # Split input tensor
    chunks = tensor.chunk(world_size, dim=dim)

    # Create output tensor
    output = torch.zeros_like(chunks[rank])

    # Reduce-scatter
    dist.reduce_scatter(output, list(chunks))

    return output


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
