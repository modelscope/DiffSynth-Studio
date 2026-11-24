from __future__ import annotations

from dataclasses import dataclass

from .sp_backend import SEQUENCE_PARALLEL_BACKENDS


@dataclass(frozen=True)
class SequenceParallelConfig:
    backend: str = "ulysses"
    # Shard and gather each block independently (full sequences between blocks) instead of one persistent stack.
    gather_between_blocks: bool = False

    def __post_init__(self):
        if self.backend not in SEQUENCE_PARALLEL_BACKENDS:
            choices = ", ".join(repr(name) for name in SEQUENCE_PARALLEL_BACKENDS)
            raise ValueError(f"unknown sequence backend {self.backend!r}; expected one of {choices}")


@dataclass(frozen=True)
class CFGParallelConfig:
    pass


@dataclass(frozen=True)
class ParallelConfig:
    sequence: SequenceParallelConfig | None = None
    cfg: CFGParallelConfig | None = None
    timeout_seconds: int = 1800

    def __post_init__(self):
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")

    @property
    def sequence_enabled(self) -> bool:
        return self.sequence is not None

    @property
    def cfg_enabled(self) -> bool:
        return self.cfg is not None
