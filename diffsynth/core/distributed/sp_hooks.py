"""Sequence parallelism installed as forward hooks on a DiT's repeated blocks.

Design
------
Every rank holds the full model weights and receives the same full inputs. A DiT
spends its time in a stack of identical blocks (`layers.0 ... layers.N`), so that
stack is the only region that is parallelized: on entry each rank keeps a
contiguous 1/P slice of the tokens, every block runs on that slice, and on exit
the slices are all-gathered back into the full sequence. Code before and after
the stack never sees a sharded tensor.

Token-wise work (projections, norms, modulation, MLP) is correct on a slice as
is. Self-attention is not, because a query needs every key; that part is solved
inside `attention_forward` by the backend in `sp_backend.py`, which this
file only enables by opening a scope.

Lifecycle of one stack forward
------------------------------
    install_sequence_parallel(model, parallel_config)
        find stacks of blocks declared with `sequence_parallel_recipe`, group them by
        parent container, and register hooks on the block objects

    first block, pre-hook   `_open`: record this rank's local token count from the
                            primary input length, set the current SP state
    every block, pre-hook   `_narrow` each declared input to the local slice
    attention calls         read the scope, exchange through the backend
    last block, post-hook   `_exit`: close the scope, all-gather the output

Hooks sit on the blocks rather than around a loop because the loop may live in
the model's `forward` or in a pipeline that reimplements it; the block objects
are the same in both. Every block, not just the first, narrows its inputs,
because loops re-pass the same full-length tensors (RoPE tables, per-token
masks) to each block while `x` is already sharded. `_narrow` therefore cuts a
global-length tensor, passes a local-length one, and rejects anything else.

What gets sharded
-----------------
A block class declares its token-aligned inputs with `sequence_parallel_recipe`.
Shapes cannot decide this: a `(B, S)` tensor may be a per-token modulation mask
(shard it) or a key mask read by the attention kernel after the all-to-all (keep
it global). Only declared arguments are narrowed. The declaration also names
`no_exchange_modules` submodules, whose K/V is already replicated (cross-attention
to text), so their attention runs on the local query without exchanging;
`no_exchange_depth` in the state marks when execution is inside one.

A sequence not divisible by P is refused: padding would enter the softmax and
change the result.
"""

from __future__ import annotations

import functools
import inspect

from typing import Callable, TypeVar

import torch
import torch.distributed as dist

from .collectives import all_gather_sequence
from .context import get_parallel_group
from .sp_backend import SEQUENCE_PARALLEL_BACKENDS
from .sp_state import SequenceParallelState


_ClassT = TypeVar("_ClassT", bound=type)


def sequence_parallel_recipe(
    split_dimensions_for_inputs: dict[str, int],
    gather_dimensions_for_outputs: tuple[int, ...] = (),
    no_exchange_modules: tuple[str, ...] = (),
) -> Callable[[_ClassT], _ClassT]:
    """Declare how one block participates in sequence parallelism.

    `split_dimensions_for_inputs` maps token-aligned arguments of `forward` to the
    dimension that carries their tokens. The first entry is the block's token
    stream: its length defines the layout, and it is the input the stack splits and
    gathers. The others are per-token companions narrowed together with it.

    `gather_dimensions_for_outputs` lists, in order, the token dimension of each
    value the last block returns. A block returning one tensor declares `(1,)`; the
    tuple is positional, so a block returning several values declares one dimension
    per value.

    `no_exchange_modules` lists submodules whose K/V is already replicated (for
    example cross-attention to text). Their attention runs on the local query
    without an exchange.
    """
    def decorate(cls: _ClassT) -> _ClassT:
        cls._sequence_parallel_recipe = {
            "split_dimensions_for_inputs": dict(split_dimensions_for_inputs),
            "gather_dimensions_for_outputs": tuple(gather_dimensions_for_outputs),
            "no_exchange_modules": tuple(no_exchange_modules),
        }
        return cls
    return decorate


class SequenceParallelStack:
    """Hooks one stack: the consecutive blocks of one container (`layers.0 ... layers.N`).

    The stack splits and gathers only at its boundary, dropping the gather and
    re-split between adjacent blocks:

        full x -> [split] -> block 0 -> block 1 -> ... -> block N -> [gather] -> full x
                             |---------- local tokens throughout ----------|

    Each block's pre-hook also cuts the global-length inputs its caller re-passes,
    such as RoPE tables. Blocks must run in order, each feeding the next.
    """

    def __init__(self, block_class, declaration, backend="ulysses"):
        self.backend = SEQUENCE_PARALLEL_BACKENDS[backend]()
        self.block_class = block_class
        self.token_inputs = dict(declaration["split_dimensions_for_inputs"])
        self.output_dims = tuple(declaration.get("gather_dimensions_for_outputs", ()))
        self.no_exchange_modules = tuple(declaration.get("no_exchange_modules", ()))
        self._state = None
        # Positional parameter names of the declared forward, in declaration order.
        self.arg_order = tuple(
            name for name, parameter in inspect.signature(block_class.forward).parameters.items()
            if name != "self" and parameter.kind in (
                parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD
            )
        )

    def install(self, blocks):
        """Hook every block, not just the ends.

        Loops that drive a block stack routinely re-supply an invariant tensor on
        every iteration — Z-Image passes the same `freqs_cis` variable each time —
        so the caller's copy is never the sharded one. A hook on the first block
        alone would leave later blocks with a sharded `x` and a full-length
        `freqs_cis`.
        """
        for index, block in enumerate(blocks):
            block.register_forward_pre_hook(
                self._make_pre_hook(is_first=index == 0), with_kwargs=True
            )
        blocks[-1].register_forward_hook(self._exit, with_kwargs=True)
        for block in blocks:
            for name in self.no_exchange_modules:
                module = self._submodule(block, name)
                if module is not None:
                    module.register_forward_pre_hook(self._enter_no_exchange, with_kwargs=True)
                    module.register_forward_hook(self._leave_no_exchange, with_kwargs=True)

    @staticmethod
    def _submodule(block, name):
        """The submodule at a dotted path, or None when this instance lacks it."""
        return functools.reduce(lambda module, part: getattr(module, part, None), name.split("."), block)

    @staticmethod
    def _enter_no_exchange(_module, _args, _kwargs):
        state = SequenceParallelState.current()
        if state is not None:
            state.no_exchange_depth += 1

    @staticmethod
    def _leave_no_exchange(_module, _args, _kwargs, _output):
        state = SequenceParallelState.current()
        if state is not None:
            state.no_exchange_depth -= 1

    def _fetch(self, name, args, kwargs):
        if name in kwargs:
            return kwargs[name]
        if name in self.arg_order:
            index = self.arg_order.index(name)
            if index < len(args):
                return args[index]
        return None

    def _narrow(self, name, tensor):
        """Shard one declared input along the dimension it declares.

        Three outcomes, all explicit: a global-length tensor is cut, an already-cut
        one passes (loops re-supply the previous iteration's value), and anything
        else is a bug in the declaration rather than something to guess about.
        """
        dim = self.token_inputs[name]
        if isinstance(tensor, (tuple, list)):
            # Narrow every element of the container.
            items = [self._narrow(name, item) for item in tensor]
            # namedtuples take positional arguments, other containers take an iterable.
            return type(tensor)(*items) if hasattr(tensor, "_fields") else type(tensor)(items)
        if not torch.is_tensor(tensor):
            return tensor
        local_length = self._state.local_length
        length = tensor.shape[dim]
        group = get_parallel_group()
        sp_size = dist.get_world_size(group)
        if length == local_length:
            return tensor
        if length == local_length * sp_size:
            return tensor.narrow(dim, dist.get_rank(group) * local_length, local_length)
        raise ValueError(f"{self.block_class.__name__}.{name}: token length {length} on dim {dim}, "
                         f"expected {local_length * sp_size} (global) or {local_length} (local)")

    def _make_pre_hook(self, is_first):
        def pre_hook(module, args, kwargs):
            if is_first and not self._open(args, kwargs):
                return None
            if self._state is None:
                return None
            new_args = tuple(
                self._narrow(name, value) if name in self.token_inputs else value
                for name, value in zip(self.arg_order, args)
            )
            new_kwargs = {
                name: (self._narrow(name, value) if name in self.token_inputs else value)
                for name, value in kwargs.items()
            }
            return new_args, new_kwargs

        return pre_hook

    def _open(self, args, kwargs):
        sequence_group = get_parallel_group()
        if SequenceParallelState.is_disabled() or sequence_group is None or dist.get_world_size(sequence_group) == 1:
            return False
        for name, dim in self.token_inputs.items():
            tensor = self._fetch(name, args, kwargs)
            if torch.is_tensor(tensor):
                break
        else:
            raise ValueError(f"{self.block_class.__name__} was called without any of "
                             f"its declared token inputs {tuple(self.token_inputs)}")
        length = tensor.shape[dim]
        sp_size = dist.get_world_size(sequence_group)
        if length % sp_size != 0:
            raise ValueError(f"{self.block_class.__name__} got {length} tokens, not "
                             f"divisible by the {sp_size} sequence-parallel ranks")
        self._state = SequenceParallelState.set_state(self.backend, length // sp_size)
        return True

    def _exit(self, module, args, kwargs, output):
        if self._state is None:
            return output
        SequenceParallelState.deactivate()
        self._state = None
        return self._gather_outputs(output)

    def _gather_outputs(self, output):
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        if len(outputs) != len(self.output_dims):
            raise ValueError(f"{self.block_class.__name__} returned {len(outputs)} values but declared {len(self.output_dims)} gather dimensions")
        group = get_parallel_group()
        gathered = []
        for item, dim in zip(outputs, self.output_dims):
            if not torch.is_tensor(item):
                raise TypeError(f"sequence-parallel stacks expect the last block to return tensors, got {type(item).__name__}")
            gathered.append(all_gather_sequence(item, dim=dim, group=group))
        if not isinstance(output, (tuple, list)):
            return gathered[0]
        return type(output)(*gathered) if hasattr(output, "_fields") else type(output)(gathered)


def _declared_class(module):
    """The sequence-parallel class of `module`, unwrapping framework wrappers.

    A wrapper (`AutoWrappedNonRecurseModule`, `FSDP[Original]`) keeps the real class
    in `.module`. Only the class's own declaration counts, not an inherited one.
    """
    cls = type(module)
    if "_sequence_parallel_recipe" in cls.__dict__:
        return cls
    inner = getattr(module, "module", None)
    if isinstance(inner, torch.nn.Module):
        return _declared_class(inner)
    return None


def _find_block_stacks(model):
    """Group the model's declared blocks into stacks.

    Stacks are discovered from structure, never declared by the model. A module is a
    block when its class, or the class inside a residency wrapper, carries a
    `sequence_parallel_recipe` declaration. A stack is the maximal run of such blocks
    under one container with consecutive indices and one class; that run is what the
    hooks open and close once, so the token stream is split and gathered at the run,
    not per block.

    A gap in the indices means an undeclared module sits in between and needs the
    full sequence; a different class means a different declaration applies. Either
    ends the run. A declared block outside a numbered list is a stack of its own.

    Z-Image is the typical case, three containers, each one class:

        noise_refiner.0..1    -> [noise_refiner.0, noise_refiner.1]
        context_refiner.0..1  -> [context_refiner.0, context_refiner.1]
        layers.0..29          -> [layers.0, ..., layers.29]

    Returns (blocks, declared class, its declaration) for each stack.
    """
    # Find every declared module, keyed by its path in the model.
    hits = {}
    for path, module in model.named_modules():
        if path and _declared_class(module) is not None:
            hits[path] = module
    # Group numbered members by their container; a stand-alone block is its own stack.
    containers = {}
    for path, module in hits.items():
        # Skip hits nested inside another hit.
        if any(path.startswith(other + ".") for other in hits):
            continue  # keep the outer module
        # Split off the last path component: "layers.3" -> ("layers", ".", "3"), "single" -> ("", "", "single").
        parent, _, index = path.rpartition(".")
        if index.isdigit():
            # Numbered member: joins the blocks of its container.
            containers.setdefault(parent, []).append((int(index), module))
        else:
            # Stand-alone block forms its own stack.
            containers[path] = [(0, module)]
    # Cut each container into runs of consecutive indices that share one class.
    runs = []
    for entries in containers.values():
        # The current run, as (position, module, class) triples in index order.
        run = []
        for position, module in sorted(entries):
            cls = _declared_class(module)
            # A gap or a class change starts a new stack.
            if run and (position != run[-1][0] + 1 or cls is not run[-1][2]):
                runs.append(run)
                run = []
            run.append((position, module, cls))
        runs.append(run)
    # Pair every run with the class and declaration that its hooks will use.
    stacks = []
    for run in runs:
        blocks = [module for _, module, _ in run]
        block_class = run[0][2]
        stacks.append((blocks, block_class, block_class._sequence_parallel_recipe))
    return stacks


def install_sequence_parallel(model, parallel_config):
    """Install sequence-parallel hooks on every declared block stack of `model`.

    Does nothing when `parallel_config` is absent or disables SP, or the group is
    missing or has one rank.

    Stacks are refused when their sequence is not divisible by the number of ranks.
    """
    if parallel_config is None or not parallel_config.sp_enabled:
        return
    sequence_group = get_parallel_group()
    if sequence_group is None or dist.get_world_size(sequence_group) == 1:
        return

    for blocks, block_class, declaration in _find_block_stacks(model):
        SequenceParallelStack(block_class, declaration, parallel_config.sequence.backend).install(blocks)
