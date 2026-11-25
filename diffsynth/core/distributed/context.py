from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from ..device.npu_compatible_device import get_device_type, get_nccl_backend, get_torch_device


class ParallelContext:
    """The process group and one group per parallel axis."""

    def __init__(self):
        self.world: dist.ProcessGroup | None = None
        self.sequence: dist.ProcessGroup | None = None
        self.cfg: dist.ProcessGroup | None = None
        self.mesh = None
        self.owns_process_group = False
        self.generation = 0

    def get_world_group(self) -> dist.ProcessGroup | None:
        """Every rank, or None before initialization."""
        return self.world

    def get_sequence_group(self) -> dist.ProcessGroup | None:
        """The axis sequence parallelism exchanges on; the world group when CFG is off."""
        return self.sequence

    def get_cfg_group(self) -> dist.ProcessGroup | None:
        """The classifier-free-guidance axis, or None when it is off."""
        return self.cfg

    def get_sequence_rank(self) -> int:
        return 0 if self.sequence is None else dist.get_rank(self.sequence)

    def get_sequence_world_size(self) -> int:
        return 1 if self.sequence is None else dist.get_world_size(self.sequence)

    def get_cfg_rank(self) -> int:
        return 0 if self.cfg is None else dist.get_rank(self.cfg)

    def get_cfg_world_size(self) -> int:
        return 1 if self.cfg is None else dist.get_world_size(self.cfg)

    def build_axes(self, parallel_config):
        self.world = dist.group.WORLD
        self.sequence = self.cfg = self.mesh = None
        world_size = dist.get_world_size(self.world)
        if world_size == 1:
            return

        cfg_size = 2 if parallel_config is not None and parallel_config.cfg_enabled else 1
        sequence_enabled = parallel_config is not None and parallel_config.sequence_enabled
        if world_size % cfg_size != 0:
            raise ValueError(f"world size {world_size} must be divisible by cfg size {cfg_size}")
        sp_size = world_size // cfg_size

        if cfg_size > 1 and sp_size > 1:
            # Outermost axis first, so the sequence axis gets adjacent ranks.
            self.mesh = init_device_mesh(
                get_device_type(), (cfg_size, sp_size), mesh_dim_names=("cfg", "sequence"),
            )
        if cfg_size > 1:
            self.cfg = self.mesh["cfg"].get_group() if self.mesh is not None else self.world
        if sequence_enabled and sp_size > 1:
            self.sequence = self.mesh["sequence"].get_group() if self.mesh is not None else self.world

    def clear(self):
        self.world = self.sequence = self.cfg = self.mesh = None
        self.owns_process_group = False


parallel_context = ParallelContext()


def init_parallel(timeout_seconds: int = 1800, parallel_config=None) -> dist.ProcessGroup:
    """Initialize or reuse the process group and lay out the parallel axes."""
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
    else:
        timeout = timedelta(seconds=timeout_seconds)
        store, rank, world_size = next(dist.rendezvous("env://", timeout=timeout))
        store = dist.PrefixStore(f"diffsynth/{parallel_context.generation}", store)

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    get_torch_device().set_device(local_rank)

    if not initialized:
        dist.init_process_group(
            backend=get_nccl_backend(), store=store, rank=rank, world_size=world_size,
            timeout=timeout, device_id=torch.device(get_device_type(), local_rank),
        )
        parallel_context.owns_process_group = True
        parallel_context.generation += 1

    parallel_context.build_axes(parallel_config)
    return parallel_context.world


def destroy_parallel() -> None:
    """Clear the groups and release the communicator if this module created it."""
    if parallel_context.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()
    parallel_context.clear()