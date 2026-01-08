"""
Tensor Parallelism for distributing large layers across GPUs.

Implements column-parallel and row-parallel linear layers following
Megatron-LM style tensor parallelism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from .parallel import get_rank, get_world_size, is_distributed


def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """
    Split a tensor along a dimension and return the chunk for this rank.

    Args:
        tensor: Input tensor to split
        dim: Dimension to split along
        world_size: Total number of ranks (auto-detected if None)
        rank: This process rank (auto-detected if None)

    Returns:
        The chunk of tensor for this rank
    """
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()

    if world_size == 1:
        return tensor

    # Ensure tensor is divisible
    size = tensor.size(dim)
    assert size % world_size == 0, f"Tensor size {size} not divisible by world_size {world_size}"

    return tensor.chunk(world_size, dim=dim)[rank].contiguous()


def gather_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Gather tensor chunks from all ranks along a dimension.

    Args:
        tensor: Local tensor chunk
        dim: Dimension to gather along
        world_size: Total number of ranks (auto-detected if None)

    Returns:
        Full gathered tensor
    """
    if world_size is None:
        world_size = get_world_size()

    if world_size == 1:
        return tensor

    # Prepare gather list
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    return torch.cat(gathered, dim=dim)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Copy input to model parallel region (identity forward, all-reduce backward)."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if is_distributed():
            dist.all_reduce(grad_output)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Reduce output from model parallel region (all-reduce forward, identity backward)."""

    @staticmethod
    def forward(ctx, input_):
        if is_distributed():
            dist.all_reduce(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather output from model parallel region."""

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        if not is_distributed():
            return input_
        return gather_tensor_along_dim(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if not is_distributed():
            return grad_output, None
        return split_tensor_along_dim(grad_output, ctx.dim), None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Scatter input to model parallel region."""

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        if not is_distributed():
            return input_
        return split_tensor_along_dim(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if not is_distributed():
            return grad_output, None
        return gather_tensor_along_dim(grad_output, ctx.dim), None


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Copy input to tensor parallel region."""
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce input from tensor parallel region."""
    return _ReduceFromModelParallelRegion.apply(input_)


def gather_from_tensor_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather input from tensor parallel region."""
    return _GatherFromModelParallelRegion.apply(input_, dim)


def scatter_to_tensor_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Scatter input to tensor parallel region."""
    return _ScatterToModelParallelRegion.apply(input_, dim)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix is split along the output dimension (columns).
    Y = XA where A is split column-wise across GPUs.

    Each GPU computes a portion of the output features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[callable] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Total output feature dimension
            bias: Whether to use bias
            gather_output: Whether to gather output from all ranks
            init_method: Weight initialization function
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        world_size = get_world_size()
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"

        self.out_features_per_partition = out_features // world_size

        # Create local weight
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method: Optional[callable] = None):
        """Initialize weights."""
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features] if gather_output,
            else [..., out_features_per_partition]
        """
        # Copy input to parallel region
        input_parallel = copy_to_tensor_parallel_region(input_)

        # Local linear
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # Gather if needed
        if self.gather_output:
            output = gather_from_tensor_parallel_region(output_parallel, dim=-1)
        else:
            output = output_parallel

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        gather_output: bool = True,
    ) -> "ColumnParallelLinear":
        """
        Create ColumnParallelLinear from existing nn.Linear.

        Splits the weight matrix along output dimension.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            gather_output=gather_output,
        )

        # Split weights
        rank = get_rank()
        world_size = get_world_size()
        weight_chunks = linear.weight.data.chunk(world_size, dim=0)
        layer.weight.data.copy_(weight_chunks[rank])

        if linear.bias is not None:
            bias_chunks = linear.bias.data.chunk(world_size, dim=0)
            layer.bias.data.copy_(bias_chunks[rank])

        return layer


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The weight matrix is split along the input dimension (rows).
    Y = XA where A is split row-wise across GPUs.

    Each GPU receives a portion of the input features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[callable] = None,
    ):
        """
        Args:
            in_features: Total input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            input_is_parallel: Whether input is already split across ranks
            init_method: Weight initialization function
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        world_size = get_world_size()
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"

        self.in_features_per_partition = in_features // world_size

        # Create local weight
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        # Bias is NOT split - only rank 0 has full bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method: Optional[callable] = None):
        """Initialize weights."""
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_: Input tensor of shape [..., in_features] or
                   [..., in_features_per_partition] if input_is_parallel

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Split input if not already parallel
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_parallel_region(input_, dim=-1)

        # Local linear (no bias yet)
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce across ranks
        output = reduce_from_tensor_parallel_region(output_parallel)

        # Add bias (only on full output after reduce)
        if self.bias is not None:
            output = output + self.bias

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        input_is_parallel: bool = False,
    ) -> "RowParallelLinear":
        """
        Create RowParallelLinear from existing nn.Linear.

        Splits the weight matrix along input dimension.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            input_is_parallel=input_is_parallel,
        )

        # Split weights
        rank = get_rank()
        world_size = get_world_size()
        weight_chunks = linear.weight.data.chunk(world_size, dim=1)
        layer.weight.data.copy_(weight_chunks[rank])

        # Bias is not split
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer


class TensorParallelLinear(nn.Module):
    """
    Wrapper that automatically chooses Column or Row parallel based on dimensions.

    For expanding layers (out > in): Use ColumnParallel
    For contracting layers (in > out): Use RowParallel
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallel_mode: Optional[str] = None,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            parallel_mode: "column", "row", or None for auto
        """
        super().__init__()

        if parallel_mode is None:
            # Auto-select based on dimensions
            parallel_mode = "column" if out_features >= in_features else "row"

        self.parallel_mode = parallel_mode

        if parallel_mode == "column":
            self.linear = ColumnParallelLinear(
                in_features, out_features, bias=bias, gather_output=True
            )
        else:
            self.linear = RowParallelLinear(
                in_features, out_features, bias=bias, input_is_parallel=False
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.linear(input_)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        parallel_mode: Optional[str] = None,
    ) -> "TensorParallelLinear":
        """Create TensorParallelLinear from existing nn.Linear."""
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)

        if parallel_mode is None:
            parallel_mode = "column" if linear.out_features >= linear.in_features else "row"

        layer.parallel_mode = parallel_mode

        if parallel_mode == "column":
            layer.linear = ColumnParallelLinear.from_linear(linear, gather_output=True)
        else:
            layer.linear = RowParallelLinear.from_linear(linear, input_is_parallel=False)

        return layer


def apply_tensor_parallelism(
    module: nn.Module,
    tp_layers: Optional[list] = None,
    min_features: int = 1024,
) -> nn.Module:
    """
    Apply tensor parallelism to a module by replacing Linear layers.

    Args:
        module: Module to parallelize
        tp_layers: List of layer names to parallelize. If None, auto-detect large layers.
        min_features: Minimum feature size to consider for parallelism

    Returns:
        Module with tensor parallel layers
    """
    world_size = get_world_size()
    if world_size == 1:
        return module

    def should_parallelize(name: str, layer: nn.Module) -> bool:
        if not isinstance(layer, nn.Linear):
            return False
        if tp_layers is not None:
            return any(tp_name in name for tp_name in tp_layers)
        # Auto-detect: parallelize large layers
        return (
            layer.in_features >= min_features or
            layer.out_features >= min_features
        ) and (
            layer.in_features % world_size == 0 and
            layer.out_features % world_size == 0
        )

    def replace_linear(parent: nn.Module, name: str, layer: nn.Linear):
        tp_layer = TensorParallelLinear.from_linear(layer)
        setattr(parent, name, tp_layer)

    # Traverse and replace
    for name, child in list(module.named_modules()):
        if should_parallelize(name, child):
            # Find parent
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = module
                child_name = name
            else:
                parent = module.get_submodule(parts[0])
                child_name = parts[1]
            replace_linear(parent, child_name, child)

    return module
