"""Generic tensor compression pipeline.

A pipeline is a list of :class:`Scheme` stages. Compression applies the stages in order,
decompression applies them in reverse, and the stored buffers only belong to the last stage::

    [svd, entropack]   compress: tensor -> (u, s, v) -> coded(u), coded(s), coded(v)
                       decompress: coded -> (u, s, v) -> tensor

New schemes only need to subclass :class:`Scheme` and call :func:`register_scheme`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = [
    "Scheme",
    "CompressedTensor",
    "CompressPipeline",
    "compress_tensor",
    "decompress_tensor",
    "register_scheme",
]

_REGISTRY: dict[str, type["Scheme"]] = {}


def register_scheme(name: str, scheme: type["Scheme"]) -> None:
    _REGISTRY[name] = scheme


def _nbytes(value: Any) -> int:
    # entropack.CompressedTensor exposes storage_nbytes(); plain tensors are dense.
    return int(value.storage_nbytes()) if hasattr(value, "storage_nbytes") else value.numel() * value.element_size()


class Scheme(ABC):
    """One compression stage over a list of tensors."""

    name: str = ""
    #: KV-cache layout this stage expects, or ``None`` when the layout does not matter.
    kv_layout: str | None = None

    @abstractmethod
    def compress(self, tensors: list[torch.Tensor]) -> tuple[list[Any], dict]:
        """Return the stored buffers and the metadata needed to decode them."""

    @abstractmethod
    def decompress(self, buffers: list[Any], meta: dict) -> list[torch.Tensor]:
        """Restore the tensors produced by :meth:`compress`."""

    def nbytes(self, buffers: list[Any]) -> int:
        return sum(_nbytes(buffer) for buffer in buffers)

    def to(self, buffers: list[Any], device: torch.device | str) -> list[Any]:
        return [buffer.to(device) for buffer in buffers]

    def __str__(self) -> str:
        return self.name


class CompressedTensor:
    """A tensor stored as the buffers of a compression pipeline."""

    def __init__(self, stages: list[tuple[Scheme, dict]], buffers: list[Any], shape, dtype):
        self.stages = stages
        self.buffers = buffers
        self.shape = tuple(shape)
        self.dtype = dtype
        self.meta: dict[str, Any] = {}

    @property
    def method(self) -> str:
        return "+".join(str(scheme) for scheme, _ in self.stages)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def original_nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()

    @property
    def nbytes(self) -> int:
        return self.stages[-1][0].nbytes(self.buffers)

    @property
    def compression_ratio(self) -> float:
        return self.original_nbytes / self.nbytes if self.nbytes else float("inf")

    @property
    def bits_per_element(self) -> float:
        return 8.0 * self.nbytes / self.numel if self.numel else 0.0

    def decompress(self) -> torch.Tensor:
        payload = self.buffers
        for scheme, meta in reversed(self.stages):
            payload = scheme.decompress(payload, meta)
        return payload[0].to(self.dtype)

    def to(self, device: torch.device | str) -> "CompressedTensor":
        moved = CompressedTensor(self.stages, self.stages[-1][0].to(self.buffers, device), self.shape, self.dtype)
        moved.meta = dict(self.meta)
        return moved

    def __repr__(self) -> str:
        return f"CompressedTensor({self.method}, shape={self.shape}, {self.nbytes / 2**20:.2f}MiB, {self.compression_ratio:.2f}x)"


class CompressPipeline:
    """A list of schemes applied in order; ``config`` is one dict or a list of dicts."""

    def __init__(self, config: dict | list[dict] | "CompressPipeline"):
        if isinstance(config, CompressPipeline):
            self.schemes = list(config.schemes)
        else:
            specs = [config] if isinstance(config, dict) else list(config)
            if not specs:
                raise ValueError("a compression pipeline needs at least one scheme")
            self.schemes = [self._build(spec) for spec in specs]
        # The KV layout is declared by the first stage that cares (SVD); entropack does not.
        self.kv_layout = next((s.kv_layout for s in self.schemes if s.kv_layout), None)

    @staticmethod
    def _build(spec: dict | Scheme) -> Scheme:
        if isinstance(spec, Scheme):
            return spec
        if not isinstance(spec, dict):
            raise TypeError(f"a scheme config must be a dict, got {type(spec).__name__}")
        spec = dict(spec)
        name = spec.pop("method", None)
        if name not in _REGISTRY:
            raise ValueError(f"unknown compression method {name!r}; available: {sorted(_REGISTRY)}")
        return _REGISTRY[name](**spec)

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        payload = [tensor]
        stages = []
        for scheme in self.schemes:
            payload, meta = scheme.compress(payload)
            stages.append((scheme, meta))
        return CompressedTensor(stages, payload, tensor.shape, tensor.dtype)

    def __str__(self) -> str:
        return "+".join(str(scheme) for scheme in self.schemes)


def compress_tensor(tensor: torch.Tensor, config: dict | list[dict]) -> CompressedTensor:
    return CompressPipeline(config).compress(tensor)


def decompress_tensor(store: CompressedTensor) -> torch.Tensor:
    if not isinstance(store, CompressedTensor):
        raise TypeError(f"expected a CompressedTensor, got {type(store).__name__}")
    return store.decompress()
