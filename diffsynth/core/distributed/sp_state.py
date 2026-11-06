"""Thread-local state that tells `attention_forward` its inputs are sharded.

`attention_forward` learns that its inputs are sharded from a thread-local
`SequenceParallelState` rather than from an argument, so no block or attention
signature in any model has to change. A block stack enters the scope on its first
block and leaves it on its last; outside a scope attention runs exactly as on one
device.
The state carries the backend, per-input `local_lengths` and `paddings`, and
`attention_parallel` for selecting local or distributed attention.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class SequenceParallelState:
    """State of one sharded stack forward, plus the per-thread activation.

    Each thread holds at most one active instance: the stack activates it on its
    first block and deactivates it after gathering its last block's outputs.
    `attention_forward` reads `backend` and `attention_parallel` for dispatch.

    `local_lengths` and `paddings` are keyed by declared forward-input names. Each
    tensor-bearing input has its own local token count and global tail-padding
    count; tensor-free inputs have no entries. Output gathering looks up padding
    through the input name specified in `gather_dimensions_for_outputs`.
    """

    _local = threading.local()

    def __init__(self, backend: "UlyssesBackend", local_lengths: dict[str, int], paddings: dict[str, int]):
        self.backend = backend
        self.local_lengths = local_lengths
        self.paddings = paddings
        # Nesting depth inside `no_exchange_modules` submodules.
        self.no_exchange_depth = 0

    @property
    def attention_parallel(self) -> bool:
        """Whether attention should exchange the current shard."""
        return self.no_exchange_depth == 0

    @classmethod
    def current(cls) -> "SequenceParallelState | None":
        return getattr(cls._local, "current", None)

    @classmethod
    def set_state(cls, backend: "UlyssesBackend", local_lengths: dict[str, int], paddings: dict[str, int]) -> "SequenceParallelState":
        """Activate a state with per-input local lengths and padding counts."""
        if cls.current() is not None:
            raise RuntimeError("sequence parallel state does not support nesting.")
        state = cls(backend, local_lengths, paddings)
        cls._local.current = state
        return state

    @classmethod
    def deactivate(cls) -> None:
        cls._local.current = None

    @classmethod
    def is_disabled(cls) -> bool:
        return getattr(cls._local, "disabled", 0) > 0

    @classmethod
    @contextmanager
    def disabled(cls):
        """Run one forward with the hooked stacks left whole.

        Used for a pass that has to produce full tensors — writing a KV cache that
        later acts as a replicated prefix. The stacks stay installed; they simply do
        not split.
        """
        depth = getattr(cls._local, "disabled", 0)
        cls._local.disabled = depth + 1
        try:
            yield
        finally:
            cls._local.disabled = depth


def current_sequence_parallel_state() -> "SequenceParallelState | None":
    """The state of the sharded stack running on this thread, or None outside one."""
    return SequenceParallelState.current()


@contextmanager
def sequence_parallel_disabled():
    """Run one forward with the hooked stacks left whole.

    Use it for a pass whose output must stay full-length on every rank, such as
    writing a KV cache that later acts as a replicated prefix. It works both as a
    context manager and as a decorator (`@sequence_parallel_disabled()`).
    """
    with SequenceParallelState.disabled():
        yield


