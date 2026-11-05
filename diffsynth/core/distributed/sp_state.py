"""Thread-local state that tells `attention_forward` its inputs are sharded.

`attention_forward` learns that its inputs are sharded from a thread-local
`SequenceParallelState` rather than from an argument, so no block or attention
signature in any model has to change. A block stack enters the scope on its first
block and leaves it on its last; outside a scope attention runs exactly as on one
device.
The state carries the backend, this rank's `local_length`, and `no_exchange_depth`
(inside a submodule whose K/V is replicated).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class SequenceParallelState:
    """State of one sharded stack forward, plus the per-thread activation.

    Each thread holds at most one active instance: the stack activates it on its
    first block and deactivates it on its last. `attention_forward` reads it to know
    it is running on a shard.
    """

    _local = threading.local()

    def __init__(self, backend: "UlyssesBackend", local_length: int):
        self.backend = backend
        self.local_length = local_length
        # Nesting depth inside `no_exchange_modules` submodules.
        self.no_exchange_depth = 0

    @classmethod
    def current(cls) -> "SequenceParallelState | None":
        return getattr(cls._local, "current", None)

    @classmethod
    def set_state(cls, backend: "UlyssesBackend", local_length: int) -> "SequenceParallelState":
        """Build a state and make it visible to `attention_forward` on this thread."""
        if cls.current() is not None:
            raise RuntimeError("sequence parallel state does not support nesting.")
        state = cls(backend, local_length)
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


