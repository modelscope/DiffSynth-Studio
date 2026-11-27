"""Compressed KV-cache storage for ``QwenImage21Pipeline``.

The pipeline writes the conditioning prefix into ``kv_cache["key"] / ["value"]`` on the first
denoising step and reads it back on every later step. A :class:`CompressedKVCacheBlock` is a
``dict`` subclass, so the attention processor is unchanged: writes compress the tensor (the dense
copy is dropped) and reads decode a temporary dense copy that attention releases afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .pipeline import CompressedTensor, CompressPipeline

__all__ = [
    "KVCompressor",
    "KVCompressionStats",
    "CompressedKVCacheBlock",
    "AccumulatingKVCacheBlock",
    "create_kv_cache_blocks",
]


@dataclass
class KVCompressionStats:
    method: str = ""
    compress_calls: int = 0
    decompress_calls: int = 0
    numel: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0
    compress_seconds: float = 0.0
    decompress_seconds: float = 0.0

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.stored_bytes if self.stored_bytes else 1.0

    @property
    def effective_bpp(self) -> float:
        return 8.0 * self.stored_bytes / self.numel if self.numel else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "compress_calls": self.compress_calls,
            "decompress_calls": self.decompress_calls,
            "original_gib": round(self.original_bytes / 2**30, 3),
            "stored_gib": round(self.stored_bytes / 2**30, 3),
            "compression_ratio": round(self.compression_ratio, 3),
            "effective_bpp": round(self.effective_bpp, 3),
            "compress_seconds": round(self.compress_seconds, 4),
            "decompress_seconds": round(self.decompress_seconds, 4),
        }


class KVCompressor:
    """Compress/decompress one ``[batch, seq, heads, head_dim]`` KV tensor per call."""

    def __init__(self, config: dict | list[dict]):
        self.pipeline = CompressPipeline(config)
        self.layout = self.pipeline.kv_layout
        self.stats = KVCompressionStats(method=str(self.pipeline))
        self._events: list[tuple[str, Any, Any]] = []

    def _to_matrices(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            return tensor
        if self.layout == "joint":
            batch, seq, heads, dim = tensor.shape
            return tensor.reshape(batch, seq, heads * dim).contiguous()
        return tensor.transpose(1, 2).contiguous()

    def _from_matrices(self, store: CompressedTensor, dense: torch.Tensor) -> torch.Tensor:
        layout = store.meta.get("kv_layout")
        if layout == "joint":
            return dense.reshape(store.meta["kv_shape"])
        if layout == "head":
            return dense.transpose(1, 2)
        return dense

    def _timer(self, tensor: torch.Tensor | None):
        if tensor is not None and tensor.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _record(self, kind: str, start) -> None:
        if isinstance(start, torch.cuda.Event):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._events.append((kind, start, end))
        else:
            field = f"{kind}_seconds"
            setattr(self.stats, field, getattr(self.stats, field) + time.perf_counter() - start)

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        start = self._timer(tensor)
        working = self._to_matrices(tensor) if self.layout else tensor
        store = self.pipeline.compress(working)
        store.meta.update(kv_shape=tuple(tensor.shape), kv_layout=self.layout or "none")
        self._record("compress", start)

        self.stats.compress_calls += 1
        self.stats.original_bytes += tensor.numel() * tensor.element_size()
        self.stats.stored_bytes += store.nbytes
        self.stats.numel += tensor.numel()
        return store

    def decompress(self, store: CompressedTensor) -> torch.Tensor:
        start = self._timer(store.buffers[0])
        dense = store.decompress()
        self._record("decompress", start)
        self.stats.decompress_calls += 1
        return self._from_matrices(store, dense)

    def report(self) -> dict[str, Any]:
        if self._events:
            torch.cuda.synchronize()
            for kind, start, end in self._events:
                field = f"{kind}_seconds"
                setattr(self.stats, field, getattr(self.stats, field) + start.elapsed_time(end) / 1000.0)
            self._events.clear()
        return self.stats.as_dict()

    def __repr__(self) -> str:
        return f"KVCompressor({self.pipeline})"


class CompressedKVCacheBlock(dict):
    """One transformer block's KV cache, storing entries compressed."""

    def __init__(self, compressor: KVCompressor | None = None):
        super().__init__()
        self.compressor = compressor

    def __setitem__(self, name: str, value: Any) -> None:
        if self.compressor is not None and isinstance(value, torch.Tensor):
            value = self.compressor.compress(value)
        super().__setitem__(name, value)

    def __getitem__(self, name: str):
        value = super().__getitem__(name)
        if isinstance(value, CompressedTensor) and self.compressor is not None:
            return self.compressor.decompress(value)
        return value

    def get(self, name: str, default: Any = None):
        return self[name] if dict.__contains__(self, name) else default


class AccumulatingKVCacheBlock(dict):
    """A write-through view of another cache block that merges writes into its entries.

    Use it when several writers produce the same keys one after another (one ControlNet per
    checkpoint, say) and the cache should still hold *one* entry per key: writing decodes the
    stored value, adds the new one and stores the re-compressed sum. Only ``target`` owns
    tensors, so N writers keep N-independent-of-writers entries.
    """

    def __init__(self, target: dict, compressor: KVCompressor | None = None):
        super().__init__()
        self.target = target
        self.compressor = compressor if compressor is not None else getattr(target, "compressor", None)

    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor) and name in self.target:
            # `target[name]` (not dict.__getitem__) so a CompressedKVCacheBlock decodes for us.
            value = self.target[name] + value
        # `target[name] = ...` compresses again and releases the previous stored value.
        self.target[name] = value

    def __getitem__(self, name: str):
        return self.target[name]

    def get(self, name: str, default: Any = None):
        return self.target[name] if name in self.target else default

    def __contains__(self, name: str) -> bool:
        return name in self.target

    def __iter__(self):
        return iter(self.target)

    def __len__(self) -> int:
        return len(self.target)

    def keys(self):
        return self.target.keys()

    def items(self):
        return self.target.items()

    def values(self):
        return self.target.values()


def create_kv_cache_blocks(num_blocks: int, compressor: KVCompressor | None) -> list[dict]:
    if compressor is None:
        return [{} for _ in range(num_blocks)]
    return [CompressedKVCacheBlock(compressor) for _ in range(num_blocks)]
