from __future__ import annotations

from dataclasses import dataclass, field

from .sequence_parallel import SEQUENCE_PARALLEL_BACKENDS


@dataclass(frozen=True)
class SequenceParallelConfig:
    backend: str | None = "ulysses"

    def __post_init__(self):
        if self.backend is not None and self.backend not in SEQUENCE_PARALLEL_BACKENDS:
            choices = ", ".join(repr(name) for name in SEQUENCE_PARALLEL_BACKENDS)
            raise ValueError(f"unknown sequence backend {self.backend!r}; expected None or one of {choices}")

    @property
    def enabled(self) -> bool:
        return self.backend is not None


@dataclass(frozen=True)
class ParallelConfig:
    sequence: SequenceParallelConfig = field(default_factory=SequenceParallelConfig)
    timeout_seconds: int = 1800

    def __post_init__(self):
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")

    @property
    def sp_enabled(self) -> bool:
        return self.sequence.enabled
