"""
Distributed computing support for DiffSynth-Studio.

Provides multi-GPU support through:
- Data Parallel (DP): Batch-level parallelism
- Tensor Parallel (TP): Layer-level parallelism
- Pipeline utilities for distributed inference
"""

from .parallel import (
    init_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    barrier,
    broadcast,
    all_reduce,
    all_gather,
)

from .tensor_parallel import (
    TensorParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    split_tensor_along_dim,
    gather_tensor_along_dim,
)

from .data_parallel import (
    DataParallelPipeline,
    scatter_batch,
    gather_outputs,
)

from .multi_gpu import (
    MultiGPUPipeline,
    auto_device_map,
    get_optimal_device_map,
)

__all__ = [
    # Initialization
    "init_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    # Communication
    "barrier",
    "broadcast",
    "all_reduce",
    "all_gather",
    # Tensor Parallel
    "TensorParallelLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "split_tensor_along_dim",
    "gather_tensor_along_dim",
    # Data Parallel
    "DataParallelPipeline",
    "scatter_batch",
    "gather_outputs",
    # Multi-GPU
    "MultiGPUPipeline",
    "auto_device_map",
    "get_optimal_device_map",
]
