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

    first block, pre-hook   `_open`: record local lengths and padding per input,
                            set the current SP state
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

Each declared input is independently padded to a multiple of P. An ordered output
map associates each returned tensor with the input whose padding is trimmed at
stack exit. Dense attention can change when padded tokens participate; varlen
streams exclude them by keeping the original segment boundaries.
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
    gather_dimensions_for_outputs: dict[str, int] | None = None,
    no_exchange_modules: tuple[str, ...] = (),
) -> Callable[[_ClassT], _ClassT]:
    """Declare the input shards and output gathers of a block stack.

    Inputs
    ------
    `split_dimensions_for_inputs` maps forward-argument names to their token
    dimensions. Every tensor-bearing argument is padded and split independently:
    its own length determines its local length and padding, not the first input's
    length. Undeclared arguments remain untouched. None and other non-tensors pass
    through without length entries and must remain tensor-free throughout the stack.
    Tuple/list inputs, including namedtuples, are traversed recursively; all tensor
    leaves under one argument must share its declared token dimension and length.
    Container types and non-tensor elements are preserved.

    The first block receives full inputs. Later blocks may receive local tensors
    from the previous block and full companion tensors re-passed by the caller.
    Hooks distinguish them by length. A stream of at most one token has equal
    global and local lengths; it is padded and re-cut on every block, which only
    rewrites padded positions that the exit trims.

    Outputs
    -------
    `gather_dimensions_for_outputs` is an ordered mapping of input names to output
    token dimensions. Insertion order matches return-value order, NOT input order
    or namedtuple field names. Each key identifies the input whose padding is
    removed after gathering that output; each value gives the gather dimension,
    which may differ from the input's split dimension. Every key must appear in the
    input map and refer to a tensor-bearing argument at stack entry.

    A single tensor requires one entry. A flat tuple/list of tensors, including a
    namedtuple, requires one entry per returned tensor and preserves its type.
    Dict returns, nested output containers, and repeated outputs associated with
    the same input name are not represented by this mapping. Omitting the output
    map stores an empty map; it does not infer output dimensions.

    For one stream with a companion RoPE table:

        @sequence_parallel_recipe(
            split_dimensions_for_inputs={"x": 1, "rope": 0},
            gather_dimensions_for_outputs={"x": 1},
        )
        class Block(nn.Module):
            def forward(self, x, rope):
                return x

    For two streams returned in the reverse of their input order:

        @sequence_parallel_recipe(
            split_dimensions_for_inputs={"image": 1, "text": 1},
            gather_dimensions_for_outputs={"text": 1, "image": 1},
        )
        class Block(nn.Module):
            def forward(self, image, text):
                return text, image

    With two ranks and image/text lengths 7/10, their local lengths are 4/5 and
    padding counts are 1/0. The first output gathers text with no trimming; the
    second gathers image and removes one trailing token.

    Scope and attention
    -------------------
    The decorator records a declaration on this class only; it is not inherited.
    Installation groups consecutive, same-class blocks into stacks. Splitting
    happens on entry, companion inputs are narrowed per block, and outputs are
    gathered only at the last block. Without enabled SP and a multi-rank group,
    or inside `sequence_parallel_disabled()`, hooks leave the forward unchanged.

    `no_exchange_modules` names submodules with replicated K/V. Their attention
    runs on the local query without exchange. Other attention inside the stack
    uses the SP backend, which requires supported, divisible head counts.

    Padding adds no attention mask: dense attention can differ from an unsharded
    run. Packed varlen attention must retain the original global segment boundaries
    and preserve the padded output length. Multi-stream models must align local
    Q/K/V with their own RoPE tables and masks. This declaration does not split a
    combined RoPE table into its constituent streams, reorder masks, or implement
    replicated-text joint attention.
    """
    output_dims = dict(gather_dimensions_for_outputs or {})
    for name in output_dims:
        if name not in split_dimensions_for_inputs:
            raise ValueError(f"gather output {name!r} is not declared in split_dimensions_for_inputs")

    def decorate(cls: _ClassT) -> _ClassT:
        cls._sequence_parallel_recipe = {
            "split_dimensions_for_inputs": dict(split_dimensions_for_inputs),
            "gather_dimensions_for_outputs": dict(output_dims),
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

    Each declared input has its own length and padding. Each block's pre-hook also
    cuts full companion tensors re-passed by the caller, such as RoPE tables.
    Blocks must run in order, each feeding its local outputs into the corresponding
    named inputs of the next block. The last block's outputs are gathered in output
    declaration order and trimmed with the padding of their associated inputs.
    """

    def __init__(self, block_class, declaration, backend="ulysses"):
        self.backend = SEQUENCE_PARALLEL_BACKENDS[backend]()
        self.block_class = block_class
        self.token_inputs = dict(declaration["split_dimensions_for_inputs"])
        self.output_dims = dict(declaration.get("gather_dimensions_for_outputs", {}))
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

        A global-length tensor is padded if needed and cut; an already-cut one passes
        (loops re-supply the previous iteration's value), except a stream of at most
        one token, whose global and local lengths coincide and is cut from the global
        side. Anything else is a bug in the declaration rather than something to
        guess about.
        """
        dim = self.token_inputs[name]
        if isinstance(tensor, (tuple, list)):
            # Narrow every element of the container.
            items = [self._narrow(name, item) for item in tensor]
            # namedtuples take positional arguments, other containers take an iterable.
            return type(tensor)(*items) if hasattr(tensor, "_fields") else type(tensor)(items)
        if not torch.is_tensor(tensor):
            return tensor
        local_length = self._state.local_lengths[name]
        length = tensor.shape[dim]
        group = get_parallel_group()
        padding = self._state.paddings[name]
        global_length = local_length * dist.get_world_size(group) - padding
        # Local output from the previous block.
        if length == local_length and length != global_length:
            return tensor
        # Full tensor from the caller; pad and cut this rank's slice.
        if length == global_length:
            tensor = self._pad(tensor, dim, padding)
            return tensor.narrow(dim, dist.get_rank(group) * local_length, local_length)
        raise ValueError(f"{self.block_class.__name__}.{name}: token length {length} on dim {dim}, "
                         f"expected {global_length} (global) or {local_length} (local)")

    @staticmethod
    def _pad(tensor, dim, padding):
        if padding == 0:
            return tensor
        shape = tensor.shape[:dim] + (padding,) + tensor.shape[dim + 1:]
        return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)

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

    def _token_length(self, name, value):
        if torch.is_tensor(value):
            return value.shape[self.token_inputs[name]]
        if not isinstance(value, (tuple, list)):
            return None
        lengths = {self._token_length(name, item) for item in value} - {None}
        if len(lengths) > 1:
            raise ValueError(f"{self.block_class.__name__}.{name}: container tensors have different token lengths {sorted(lengths)}")
        return next(iter(lengths), None)

    def _open(self, args, kwargs):
        sequence_group = get_parallel_group()
        if SequenceParallelState.is_disabled() or sequence_group is None or dist.get_world_size(sequence_group) == 1:
            return False
        sp_size = dist.get_world_size(sequence_group)
        local_lengths, paddings = {}, {}
        for name in self.token_inputs:
            length = self._token_length(name, self._fetch(name, args, kwargs))
            if length is not None:
                paddings[name] = (-length) % sp_size
                local_lengths[name] = (length + paddings[name]) // sp_size
        if not local_lengths:
            raise ValueError(f"{self.block_class.__name__} was called without any of its declared token inputs {tuple(self.token_inputs)}")
        for name in self.output_dims:
            if name not in local_lengths:
                raise ValueError(f"{self.block_class.__name__}: gather output {name!r} has no tensor-bearing input at stack entry")
        self._state = SequenceParallelState.set_state(self.backend, local_lengths, paddings)
        return True

    def _exit(self, module, args, kwargs, output):
        if self._state is None:
            return output
        output = self._gather_outputs(output)
        SequenceParallelState.deactivate()
        self._state = None
        return output

    def _gather_outputs(self, output):
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        if len(outputs) != len(self.output_dims):
            raise ValueError(f"{self.block_class.__name__} returned {len(outputs)} values but declared {len(self.output_dims)} gather dimensions")
        group = get_parallel_group()
        gathered = []
        for item, (name, dim) in zip(outputs, self.output_dims.items()):
            if not torch.is_tensor(item):
                raise TypeError(f"sequence-parallel stacks expect the last block to return tensors, got {type(item).__name__}")
            item = all_gather_sequence(item, dim=dim, group=group)
            item = item.narrow(dim, 0, item.shape[dim] - self._state.paddings[name])
            gathered.append(item)
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

    Each input is padded independently when its token length is not divisible by
    the number of ranks; each output is trimmed through its associated input name.
    """
    if parallel_config is None or not parallel_config.sp_enabled:
        return
    sequence_group = get_parallel_group()
    if sequence_group is None or dist.get_world_size(sequence_group) == 1:
        return

    for blocks, block_class, declaration in _find_block_stacks(model):
        SequenceParallelStack(block_class, declaration, parallel_config.sequence.backend).install(blocks)
