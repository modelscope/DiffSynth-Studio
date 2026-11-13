from datetime import timedelta
import os

import torch
import torch.distributed as dist

from ..device.npu_compatible_device import get_device_type, get_nccl_backend, get_torch_device


_PARALLEL_GROUP: dist.ProcessGroup | None = None
_OWNS_PROCESS_GROUP = False
_GENERATION = 0


def init_parallel(timeout_seconds: int = 1800) -> dist.ProcessGroup:
    """Initialize or reuse the process-wide communication group."""
    global _PARALLEL_GROUP, _OWNS_PROCESS_GROUP, _GENERATION
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
    else:
        timeout = timedelta(seconds=timeout_seconds)
        store, rank, world_size = next(dist.rendezvous("env://", timeout=timeout))
        store = dist.PrefixStore(f"diffsynth/{_GENERATION}", store)

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    get_torch_device().set_device(local_rank)

    if not initialized:
        dist.init_process_group(
            backend=get_nccl_backend(), store=store, rank=rank, world_size=world_size,
            timeout=timeout, device_id=torch.device(get_device_type(), local_rank),
        )
        _OWNS_PROCESS_GROUP = True
        _GENERATION += 1

    _PARALLEL_GROUP = dist.group.WORLD
    return _PARALLEL_GROUP


def get_parallel_group() -> dist.ProcessGroup | None:
    """Return the process-wide group, or None before initialization."""
    return _PARALLEL_GROUP


def destroy_parallel() -> None:
    """Clear the group and release the communicator if this module created it."""
    global _PARALLEL_GROUP, _OWNS_PROCESS_GROUP
    if _OWNS_PROCESS_GROUP and dist.is_initialized():
        dist.destroy_process_group()
    _PARALLEL_GROUP = None
    _OWNS_PROCESS_GROUP = False
