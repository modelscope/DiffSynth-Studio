"""
Data Parallelism utilities for batch-level parallel processing.

Enables processing multiple inputs across GPUs simultaneously.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Union, Tuple
from PIL import Image
import numpy as np

from .parallel import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    barrier,
    all_gather,
)


def scatter_batch(
    batch: Union[torch.Tensor, List, Dict],
    dim: int = 0,
) -> Union[torch.Tensor, List, Dict]:
    """
    Scatter a batch across all ranks.

    Args:
        batch: Input batch (tensor, list, or dict)
        dim: Dimension to scatter along for tensors

    Returns:
        Local portion of the batch for this rank
    """
    if not is_distributed():
        return batch

    rank = get_rank()
    world_size = get_world_size()

    if isinstance(batch, torch.Tensor):
        # Split tensor
        chunks = batch.chunk(world_size, dim=dim)
        if rank < len(chunks):
            return chunks[rank]
        else:
            # Edge case: more ranks than batch items
            return chunks[-1][:0]  # Empty tensor

    elif isinstance(batch, (list, tuple)):
        # Split list
        batch_size = len(batch)
        per_rank = batch_size // world_size
        remainder = batch_size % world_size

        start = rank * per_rank + min(rank, remainder)
        end = start + per_rank + (1 if rank < remainder else 0)

        result = batch[start:end]
        return type(batch)(result) if isinstance(batch, tuple) else result

    elif isinstance(batch, dict):
        # Scatter each value
        return {k: scatter_batch(v, dim) for k, v in batch.items()}

    else:
        # Cannot scatter, return as-is
        return batch


def gather_outputs(
    output: Union[torch.Tensor, List, Dict],
    dim: int = 0,
    dst: int = 0,
) -> Union[torch.Tensor, List, Dict]:
    """
    Gather outputs from all ranks.

    Args:
        output: Local output from this rank
        dim: Dimension to gather along for tensors
        dst: Destination rank (only this rank gets the full output)

    Returns:
        Gathered output on dst rank, original on others
    """
    if not is_distributed():
        return output

    rank = get_rank()
    world_size = get_world_size()

    if isinstance(output, torch.Tensor):
        # Gather tensor
        gathered = all_gather(output)
        if rank == dst:
            return torch.cat(gathered, dim=dim)
        return output

    elif isinstance(output, list):
        # Gather lists
        gathered = all_gather_object(output)
        if rank == dst:
            result = []
            for sublist in gathered:
                result.extend(sublist)
            return result
        return output

    elif isinstance(output, dict):
        # Gather each value
        if rank == dst:
            return {k: gather_outputs(v, dim, dst) for k, v in output.items()}
        return output

    elif isinstance(output, Image.Image):
        # Convert PIL to tensor, gather, convert back
        output_tensor = torch.from_numpy(np.array(output))
        gathered = all_gather(output_tensor)
        if rank == dst:
            return [Image.fromarray(t.numpy()) for t in gathered]
        return output

    else:
        # Cannot gather, return as-is
        return output


def all_gather_object(obj: Any) -> List[Any]:
    """Gather arbitrary Python objects from all ranks."""
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    output = [None] * world_size
    dist.all_gather_object(output, obj)
    return output


class DataParallelPipeline:
    """
    Wrapper to enable data parallelism for diffusion pipelines.

    Distributes batch processing across multiple GPUs.
    """

    def __init__(
        self,
        pipeline: nn.Module,
        gather_on_main: bool = True,
    ):
        """
        Args:
            pipeline: The diffusion pipeline to parallelize
            gather_on_main: Whether to gather results on main process
        """
        self.pipeline = pipeline
        self.gather_on_main = gather_on_main
        self.rank = get_rank()
        self.world_size = get_world_size()

    def __call__(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> Union[List[Image.Image], torch.Tensor]:
        """
        Run pipeline with data parallel batch processing.

        Args:
            batch_size: Total batch size across all ranks
            **kwargs: Pipeline arguments (prompt, negative_prompt, etc.)

        Returns:
            Generated images (gathered on main process if gather_on_main=True)
        """
        # Calculate local batch size
        local_batch_size = batch_size // self.world_size
        remainder = batch_size % self.world_size
        if self.rank < remainder:
            local_batch_size += 1

        if local_batch_size == 0:
            # This rank has no work
            barrier()
            if self.gather_on_main and not is_main_process():
                return []
            return []

        # Scatter prompt if it's a list
        prompt = kwargs.get("prompt", "")
        if isinstance(prompt, list):
            kwargs["prompt"] = scatter_batch(prompt)

        negative_prompt = kwargs.get("negative_prompt", "")
        if isinstance(negative_prompt, list):
            kwargs["negative_prompt"] = scatter_batch(negative_prompt)

        # Scatter seed if provided as list
        seed = kwargs.get("seed", None)
        if isinstance(seed, list):
            kwargs["seed"] = scatter_batch(seed)

        # Run local inference
        local_output = self.pipeline(**kwargs)

        # Gather results
        if self.gather_on_main:
            output = gather_outputs(local_output, dst=0)
            barrier()
            return output if is_main_process() else local_output
        else:
            barrier()
            return local_output

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped pipeline."""
        if name in ("pipeline", "gather_on_main", "rank", "world_size"):
            return object.__getattribute__(self, name)
        return getattr(self.pipeline, name)

    def to(self, device):
        """Move pipeline to device."""
        self.pipeline.to(device)
        return self


class BatchDistributor:
    """
    Utility class for distributing work across GPUs.

    Handles batch creation and result collection for multi-GPU inference.
    """

    def __init__(self, world_size: Optional[int] = None):
        self.world_size = world_size or get_world_size()
        self.rank = get_rank()

    def distribute_prompts(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[int]]:
        """
        Distribute prompts across ranks.

        Args:
            prompts: List of prompts

        Returns:
            Tuple of (local_prompts, original_indices)
        """
        indices = list(range(len(prompts)))
        local_prompts = scatter_batch(prompts)
        local_indices = scatter_batch(indices)
        return local_prompts, local_indices

    def collect_results(
        self,
        local_results: List[Any],
        local_indices: List[int],
    ) -> List[Any]:
        """
        Collect results from all ranks and reorder.

        Args:
            local_results: Results from this rank
            local_indices: Original indices of local results

        Returns:
            All results in original order
        """
        # Gather results and indices
        all_results = all_gather_object(local_results)
        all_indices = all_gather_object(local_indices)

        if not is_main_process():
            return local_results

        # Flatten and reorder
        flat_results = []
        flat_indices = []
        for results, indices in zip(all_results, all_indices):
            flat_results.extend(results)
            flat_indices.extend(indices)

        # Sort by original index
        sorted_pairs = sorted(zip(flat_indices, flat_results), key=lambda x: x[0])
        return [result for _, result in sorted_pairs]

    def get_local_batch_size(self, total_batch_size: int) -> int:
        """
        Calculate local batch size for this rank.

        Args:
            total_batch_size: Total batch size across all ranks

        Returns:
            Batch size for this rank
        """
        base = total_batch_size // self.world_size
        remainder = total_batch_size % self.world_size
        return base + (1 if self.rank < remainder else 0)
