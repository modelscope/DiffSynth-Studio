"""
Multi-GPU Pipeline Support.

Provides utilities for distributing models across multiple GPUs
and running inference efficiently.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass
import math

from .parallel import (
    init_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    barrier,
    get_device,
)
from .tensor_parallel import apply_tensor_parallelism


@dataclass
class DeviceMapEntry:
    """Entry in a device map."""
    device: str
    dtype: torch.dtype = torch.bfloat16
    offload: bool = False


def get_gpu_memory_info() -> List[Tuple[int, int, int]]:
    """
    Get memory info for all available GPUs.

    Returns:
        List of (device_id, free_memory_bytes, total_memory_bytes)
    """
    if not torch.cuda.is_available():
        return []

    info = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        info.append((i, free, total))
    return info


def estimate_model_size(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> int:
    """
    Estimate the memory size of a model in bytes.

    Args:
        model: PyTorch model
        dtype: Data type for estimation

    Returns:
        Estimated size in bytes
    """
    dtype_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    total_params = sum(p.numel() for p in model.parameters())
    return total_params * dtype_size


def auto_device_map(
    models: Dict[str, nn.Module],
    max_memory_per_gpu: Optional[Dict[int, int]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, str]:
    """
    Automatically create a device map for models.

    Distributes models across available GPUs based on their size
    and available memory.

    Args:
        models: Dictionary of model name -> model
        max_memory_per_gpu: Maximum memory to use per GPU (bytes)
        dtype: Data type for memory estimation

    Returns:
        Dictionary of model name -> device string
    """
    if not torch.cuda.is_available():
        return {name: "cpu" for name in models}

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return {name: "cpu" for name in models}

    if num_gpus == 1:
        return {name: "cuda:0" for name in models}

    # Get available memory per GPU
    gpu_memory = get_gpu_memory_info()

    if max_memory_per_gpu is None:
        max_memory_per_gpu = {i: int(free * 0.9) for i, free, _ in gpu_memory}

    # Estimate model sizes
    model_sizes = {name: estimate_model_size(model, dtype) for name, model in models.items()}

    # Sort models by size (largest first)
    sorted_models = sorted(model_sizes.items(), key=lambda x: -x[1])

    # Greedy assignment
    device_map = {}
    gpu_usage = {i: 0 for i in range(num_gpus)}

    for name, size in sorted_models:
        # Find GPU with most available space
        best_gpu = min(
            range(num_gpus),
            key=lambda i: gpu_usage[i] if gpu_usage[i] + size <= max_memory_per_gpu.get(i, float('inf')) else float('inf')
        )

        if gpu_usage[best_gpu] + size <= max_memory_per_gpu.get(best_gpu, float('inf')):
            device_map[name] = f"cuda:{best_gpu}"
            gpu_usage[best_gpu] += size
        else:
            # No GPU has enough space, use CPU offload
            device_map[name] = "cpu"

    return device_map


def get_optimal_device_map(
    model_configs: List[Dict[str, Any]],
    strategy: str = "balanced",
) -> Dict[str, str]:
    """
    Get optimal device mapping for model configurations.

    Args:
        model_configs: List of model configurations with estimated sizes
        strategy: Distribution strategy ("balanced", "sequential", "largest_first")

    Returns:
        Device map dictionary
    """
    if not torch.cuda.is_available():
        return {cfg.get("name", f"model_{i}"): "cpu" for i, cfg in enumerate(model_configs)}

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return {cfg.get("name", f"model_{i}"): "cpu" for i, cfg in enumerate(model_configs)}

    device_map = {}

    if strategy == "balanced":
        # Distribute models evenly across GPUs
        for i, cfg in enumerate(model_configs):
            name = cfg.get("name", f"model_{i}")
            device_map[name] = f"cuda:{i % num_gpus}"

    elif strategy == "sequential":
        # Fill GPUs sequentially
        gpu_idx = 0
        gpu_memory = get_gpu_memory_info()
        current_usage = 0

        for i, cfg in enumerate(model_configs):
            name = cfg.get("name", f"model_{i}")
            size = cfg.get("size_bytes", 0)

            _, free, _ = gpu_memory[gpu_idx]
            if current_usage + size > free * 0.9:
                gpu_idx = min(gpu_idx + 1, num_gpus - 1)
                current_usage = 0

            device_map[name] = f"cuda:{gpu_idx}"
            current_usage += size

    elif strategy == "largest_first":
        # Sort by size and assign largest to first GPU
        sorted_configs = sorted(
            enumerate(model_configs),
            key=lambda x: x[1].get("size_bytes", 0),
            reverse=True
        )

        gpu_usage = [0] * num_gpus

        for i, cfg in sorted_configs:
            name = cfg.get("name", f"model_{i}")
            size = cfg.get("size_bytes", 0)

            # Assign to GPU with least usage
            best_gpu = min(range(num_gpus), key=lambda g: gpu_usage[g])
            device_map[name] = f"cuda:{best_gpu}"
            gpu_usage[best_gpu] += size

    return device_map


class MultiGPUPipeline:
    """
    Pipeline wrapper that distributes model components across multiple GPUs.

    Supports:
    - Model parallel: Different models on different GPUs
    - Tensor parallel: Single model split across GPUs
    - Data parallel: Same model on all GPUs processing different data
    """

    def __init__(
        self,
        pipeline: nn.Module,
        parallel_mode: str = "model",
        tensor_parallel_layers: Optional[List[str]] = None,
        device_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            pipeline: The diffusion pipeline
            parallel_mode: Parallelism mode ("model", "tensor", "data", "hybrid")
            tensor_parallel_layers: Layers to apply tensor parallelism to
            device_map: Manual device mapping for model parallel
        """
        self.pipeline = pipeline
        self.parallel_mode = parallel_mode
        self.tensor_parallel_layers = tensor_parallel_layers
        self.device_map = device_map

        self._setup_parallelism()

    def _setup_parallelism(self):
        """Set up the parallelism strategy."""
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if num_gpus <= 1:
            # Single GPU or CPU - no parallelism needed
            self.parallel_mode = "none"
            return

        if self.parallel_mode == "model":
            self._setup_model_parallel()
        elif self.parallel_mode == "tensor":
            self._setup_tensor_parallel()
        elif self.parallel_mode == "data":
            self._setup_data_parallel()
        elif self.parallel_mode == "hybrid":
            self._setup_hybrid_parallel()

    def _setup_model_parallel(self):
        """Set up model parallelism - different models on different GPUs."""
        # Get all model components
        models = {}
        for name, module in self.pipeline.named_children():
            if isinstance(module, nn.Module) and sum(p.numel() for p in module.parameters()) > 0:
                models[name] = module

        # Create device map if not provided
        if self.device_map is None:
            self.device_map = auto_device_map(models)

        # Move models to assigned devices
        for name, device in self.device_map.items():
            if hasattr(self.pipeline, name):
                module = getattr(self.pipeline, name)
                if module is not None:
                    module.to(device)

    def _setup_tensor_parallel(self):
        """Set up tensor parallelism - layers split across GPUs."""
        # Initialize distributed
        init_distributed()

        # Apply tensor parallelism to specified layers
        for name, module in self.pipeline.named_children():
            if isinstance(module, nn.Module):
                apply_tensor_parallelism(
                    module,
                    tp_layers=self.tensor_parallel_layers,
                )

    def _setup_data_parallel(self):
        """Set up data parallelism - same model on all GPUs."""
        from .data_parallel import DataParallelPipeline

        # Wrap pipeline in data parallel wrapper
        self._data_parallel = DataParallelPipeline(self.pipeline)

    def _setup_hybrid_parallel(self):
        """Set up hybrid parallelism - combination of strategies."""
        # Use model parallel for large components
        self._setup_model_parallel()

        # Use tensor parallel for transformer blocks
        if self.tensor_parallel_layers:
            self._setup_tensor_parallel()

    def __call__(self, **kwargs):
        """Run the pipeline."""
        if self.parallel_mode == "data":
            return self._data_parallel(**kwargs)
        else:
            return self.pipeline(**kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped pipeline."""
        if name in ("pipeline", "parallel_mode", "tensor_parallel_layers", "device_map", "_data_parallel"):
            return object.__getattribute__(self, name)
        return getattr(self.pipeline, name)

    def to(self, device):
        """Move pipeline to device (respects device map if set)."""
        if self.device_map:
            # Already distributed, ignore
            return self
        self.pipeline.to(device)
        return self

    @property
    def device(self):
        """Get the primary device of the pipeline."""
        if self.device_map:
            # Return the device of the main model (usually the transformer)
            for key in ["dit", "transformer", "unet"]:
                if key in self.device_map:
                    return torch.device(self.device_map[key])
            # Return first device
            return torch.device(list(self.device_map.values())[0])
        return get_device()


def enable_multi_gpu(
    pipeline: nn.Module,
    mode: str = "auto",
    tensor_parallel_layers: Optional[List[str]] = None,
) -> MultiGPUPipeline:
    """
    Enable multi-GPU support for a pipeline.

    Args:
        pipeline: The diffusion pipeline
        mode: Parallelism mode ("auto", "model", "tensor", "data", "hybrid")
        tensor_parallel_layers: Layers to apply tensor parallelism to

    Returns:
        MultiGPUPipeline wrapper
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus <= 1:
        # No multi-GPU needed
        return MultiGPUPipeline(pipeline, parallel_mode="none")

    if mode == "auto":
        # Auto-select based on model size and GPU count
        total_params = sum(p.numel() for p in pipeline.parameters())

        if total_params > 10e9:  # > 10B parameters
            # Large model - use tensor or model parallel
            mode = "tensor" if num_gpus >= 4 else "model"
        else:
            # Smaller model - use data parallel for throughput
            mode = "data"

    return MultiGPUPipeline(
        pipeline,
        parallel_mode=mode,
        tensor_parallel_layers=tensor_parallel_layers,
    )


def setup_multi_gpu_training(
    pipeline: nn.Module,
    use_ddp: bool = True,
    use_fsdp: bool = False,
    mixed_precision: bool = True,
) -> nn.Module:
    """
    Set up multi-GPU training for a pipeline.

    Args:
        pipeline: The model/pipeline to train
        use_ddp: Whether to use DistributedDataParallel
        use_fsdp: Whether to use FullyShardedDataParallel
        mixed_precision: Whether to use mixed precision

    Returns:
        Wrapped model ready for distributed training
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized():
        init_distributed()

    rank = get_rank()
    device = torch.device(f"cuda:{rank}")

    # Move to device
    pipeline = pipeline.to(device)

    if use_fsdp:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision

            mp_policy = None
            if mixed_precision:
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            pipeline = FSDP(
                pipeline,
                mixed_precision=mp_policy,
                device_id=rank,
            )
        except ImportError:
            print("FSDP not available, falling back to DDP")
            use_ddp = True
            use_fsdp = False

    if use_ddp and not use_fsdp:
        pipeline = DDP(
            pipeline,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

    return pipeline
