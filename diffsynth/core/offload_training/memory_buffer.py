import torch

ALIGNMENT = 64

def _align_up(x: int, alignment: int = ALIGNMENT) -> int:
    return (x + alignment - 1) // alignment * alignment

def _next_power_of_two(x: int) -> int:
    """
    Smallest power of two >= x.
    For power-of-two x=2^k: (x-1) has bit_length=k, so 1<<k = x (unchanged).
    For non-power-of-two: (x-1).bit_length() exceeds floor-log2(x), rounding up to next 2^n.
    """
    return 1 if x <= 1 else 1 << (x - 1).bit_length()

def _prev_power_of_two(x: int) -> int:
    """Largest power-of-two <= x."""
    return 1 if x <= 1 else 1 << (x.bit_length() - 1)

def _tensor_storage_size(tensor: torch.Tensor) -> int:
    return _align_up(tensor.numel() * tensor.element_size())


class BaseBufferPool:
    """Naive per-tensor pin_memory allocation. No pre-allocation, no memory saving."""

    def allocate_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pin_memory()

    @classmethod
    def from_model(cls, model: torch.nn.Module, **kwargs):
        return cls()


class PinnedBuffer:
    """Single pinned uint8 buffer with bump-pointer allocation. Lazy: actual memory allocated on first allocate_like."""

    def __init__(self, size: int):
        self._size = size
        self._buf: torch.Tensor | None = None
        self._offset = 0

    def _ensure_allocated(self):
        if self._buf is None:
            self._buf = torch.empty(self._size, dtype=torch.uint8, device="cpu", pin_memory=True)

    @property
    def capacity(self) -> int:
        return self._size

    @property
    def remaining(self) -> int:
        return self._size if self._buf is None else self._buf.numel() - self._offset

    @property
    def used(self) -> int:
        return self._offset

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, min_size: int = 1 * 1024**3):
        size = max(_tensor_storage_size(tensor) + ALIGNMENT, min_size)
        return cls(_next_power_of_two(size))

    def allocate_like(self, tensor: torch.Tensor, *, copy: bool = True, non_blocking: bool = False) -> torch.Tensor | None:
        """Try to allocate a view for tensor. Returns None if not enough space."""
        num_bytes = tensor.numel() * tensor.element_size()
        if num_bytes > self.remaining:
            return None
        self._ensure_allocated()
        view = self._buf.narrow(0, self._offset, num_bytes).view(tensor.dtype).reshape(tuple(tensor.shape))
        if copy:
            view.copy_(tensor, non_blocking=bool(non_blocking and tensor.device.type == "cuda"))
        self._offset = _align_up(self._offset + num_bytes)
        return view


class PinnedArenaPool(BaseBufferPool):
    """Pinned arena pool — pre-allocate pinned memory, avoid per-tensor cudaHostAlloc overhead.

    Pool strategy:
      1. Sizing: from_model() scans all non-trainable params + buffers, sums their aligned sizes
         as total_bytes. max_chunk_size is raised to fit the largest single tensor.
      2. Decomposition: total_bytes is split into power-of-two chunks (min_chunk_size ~ max_chunk_size).
         Each chunk becomes one PinnedBuffer (lazy — actual pin_memory on first use).
      3. Allocation: allocate_like() sequentially probes each buffer for space (first-fit).
         Each PinnedBuffer uses bump-pointer with ALIGNMENT padding between tensors.
      4. Growth: if all existing buffers are full, _grow() appends a new PinnedBuffer sized
         to the requesting tensor (at least min_chunk_size, power-of-two rounded).
      5. Fallback: on any exception, falls back to per-tensor pin_memory().
    """

    def __init__(self, total_bytes: int, min_chunk_size: int = 1 * 1024**3, max_chunk_size: int = 4 * 1024**3):
        self.min_chunk_size = _next_power_of_two(int(min_chunk_size))
        self.max_chunk_size = _next_power_of_two(int(max_chunk_size))
        self.min_chunk_size = min(self.min_chunk_size, self.max_chunk_size)
        self._buffers = [PinnedBuffer(s) for s in self._decompose(total_bytes, self.min_chunk_size, self.max_chunk_size)]

    @classmethod
    def from_model(cls, model: torch.nn.Module, min_chunk_size: int = 1 * 1024**3, max_chunk_size: int = 4 * 1024**3):
        """Size pool for all non-trainable params + buffers."""
        tensors = [p for p in model.parameters() if not p.requires_grad] + list(model.buffers())
        total = sum(_tensor_storage_size(t) for t in tensors)
        max_tensor_size = max((_tensor_storage_size(t) for t in tensors), default=0)
        max_chunk_size = _next_power_of_two(max_tensor_size) if max_tensor_size > max_chunk_size else max_chunk_size
        return cls(total, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)

    @staticmethod
    def _decompose(total_bytes: int, min_chunk_size: int, max_chunk_size: int) -> list:
        """Decompose total_bytes into power-of-two chunks capped by min/max."""
        if total_bytes <= 0:
            return []
        chunks, remaining = [], total_bytes
        while remaining > 0:
            chunk = max(min(_prev_power_of_two(remaining), max_chunk_size), min_chunk_size)
            chunks.append(chunk)
            remaining -= chunk
        chunks.sort(reverse=True)
        return chunks

    def _grow(self, tensor: torch.Tensor):
        self._buffers.append(PinnedBuffer.from_tensor(tensor, min_size=self.min_chunk_size))

    def allocate_like(self, tensor: torch.Tensor, *, copy: bool = True, require_contiguous: bool = True, non_blocking: bool = False) -> torch.Tensor:
        """Allocate a pinned view. Falls back to per-tensor pin_memory on failure."""
        src = tensor.detach()
        if require_contiguous and not src.is_contiguous():
            src = src.contiguous()
        try:
            for buf in self._buffers:
                view = buf.allocate_like(src, copy=copy, non_blocking=non_blocking)
                if view is not None:
                    return view
            self._grow(src)
            return self._buffers[-1].allocate_like(src, copy=copy, non_blocking=non_blocking)
        except Exception:
            return src.pin_memory()
