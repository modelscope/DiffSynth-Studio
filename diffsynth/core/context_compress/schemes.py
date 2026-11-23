"""Compression schemes: randomized SVD (``torch.svd_lowrank`` / Cholesky-QR) and EntroPack."""

from __future__ import annotations

from abc import abstractmethod

import torch

from .pipeline import Scheme, register_scheme

__all__ = ["SVD", "CholeskyQR", "EntroPack"]


def _parse_dtype(name) -> torch.dtype | None:
    if name is None or isinstance(name, torch.dtype):
        return name
    dtype = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


class _RandomizedSVD(Scheme):
    """Shared rank policy, batching, storage and reconstruction of the randomized SVD schemes.

    ``rank`` / ``rank_ratio`` / ``energy`` select how many singular triplets are kept (default
    half of ``min(m, n)``); ``layout`` selects the KV-cache matrix layout.
    """

    def __init__(self, rank=None, rank_ratio=None, energy=None, oversample=8, niter=2, seed=0,
                 factor_dtype=None, energy_max_rank=128, layout="head"):
        if layout not in ("head", "joint"):
            raise ValueError(f"layout must be 'head' or 'joint', got {layout!r}")
        if rank is not None and rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if rank_ratio is not None and not 0 < rank_ratio <= 1:
            raise ValueError(f"rank_ratio must be in (0, 1], got {rank_ratio}")
        if energy is not None and not 0 < energy <= 1:
            raise ValueError(f"energy must be in (0, 1], got {energy}")
        self.rank = rank
        self.rank_ratio = rank_ratio
        self.energy = energy
        self.oversample = oversample
        self.niter = niter
        self.seed = seed
        self.factor_dtype = _parse_dtype(factor_dtype)
        self.energy_max_rank = energy_max_rank
        self.kv_layout = layout

    @abstractmethod
    def _decompose(self, matrix: torch.Tensor, probe: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the leading ``probe`` singular triplets of a ``[batch, m, n]`` matrix."""

    def _rank(self, m: int, n: int) -> int | None:
        smallest = min(m, n)
        if self.rank is not None:
            return min(self.rank, smallest)
        if self.rank_ratio is not None:
            return max(1, round(self.rank_ratio * smallest))
        if self.energy is not None:
            return None
        return max(1, round(0.5 * smallest))

    def compress(self, tensors):
        tensor = tensors[0]
        shape = tensor.shape
        m, n = (shape[-2], shape[-1]) if tensor.ndim >= 2 else (1, shape[-1])
        matrix = tensor.reshape(-1, m, n).float()

        rank = self._rank(m, n)
        if rank is not None and rank >= min(m, n):  # factors would not be smaller than the tensor
            return [tensor], {"shape": tuple(shape), "raw": True}
        probe = min(min(m, n), (rank if rank is not None else min(min(m, n), self.energy_max_rank)) + self.oversample)
        devices = [matrix.device.index] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if matrix.is_cuda:
                torch.cuda.manual_seed(self.seed)
            else:
                torch.manual_seed(self.seed)
            u, s, v = self._decompose(matrix, probe)

        if rank is None:  # keep the smallest rank whose captured energy reaches the target
            variance = s.square()
            captured = variance.cumsum(-1) / variance.sum(-1, keepdim=True).clamp_min(torch.finfo(s.dtype).tiny)
            rank = int((captured < self.energy).sum(-1).add(1).max().clamp(max=s.shape[-1]))
        dtype = self.factor_dtype or tensor.dtype
        return [x[..., :rank].to(dtype) for x in (u, s, v)], {"shape": tuple(shape), "rank": rank}

    def decompress(self, buffers, meta):
        if meta.get("raw"):
            return [buffers[0]]
        u, s, v = buffers
        return [torch.matmul(u * s.unsqueeze(-2), v.transpose(-2, -1)).reshape(meta["shape"])]

    def __str__(self):
        if self.rank is not None:
            policy = f"rank={self.rank}"
        elif self.rank_ratio is not None:
            policy = f"ratio={self.rank_ratio}"
        elif self.energy is not None:
            policy = f"energy={self.energy}"
        else:
            policy = "ratio=0.5"
        return f"{self.name}({policy},{self.kv_layout})"


class SVD(_RandomizedSVD):
    """Randomized SVD via ``torch.svd_lowrank``."""

    name = "svd"

    def _decompose(self, matrix, probe):
        return torch.svd_lowrank(matrix, q=probe, niter=self.niter)


class CholeskyQR(_RandomizedSVD):
    """Randomized SVD with a Cholesky-QR range finder.

    The same Halko-Martinsson-Tropp algorithm as :class:`SVD`, but the range finder is
    orthonormalized with two Cholesky-QR passes (Gram matrix + triangular solve, both fast
    matmul kernels) instead of ``torch.linalg.qr``, which is very slow for tall, skinny
    batches on some accelerators. The small projected SVD still uses ``torch.svd_lowrank``.
    """

    name = "cholesky_qr"

    @staticmethod
    def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        for _ in range(2):
            gram = matrix.transpose(-2, -1) @ matrix
            scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(torch.finfo(gram.dtype).tiny)
            # A rank-deficient sample makes the Gram matrix singular; load the diagonal until
            # the factorization is finite, and fall back to QR as a last resort.
            for jitter in (1e-6, 1e-4, 1e-2, 1.0):
                try:
                    chol = torch.linalg.cholesky(gram + (jitter * scale)[..., None, None] * eye)
                    if not torch.isfinite(chol).all():
                        continue
                    orth = torch.linalg.solve_triangular(chol, matrix.transpose(-2, -1), upper=False)
                    orth = orth.transpose(-2, -1)
                    if torch.isfinite(orth).all():
                        matrix = orth
                        break
                except RuntimeError:
                    continue
            else:
                return torch.linalg.qr(matrix).Q
        return matrix

    def _decompose(self, matrix, probe):
        batch, cols = matrix.shape[0], matrix.shape[2]
        sample = matrix @ torch.randn(batch, cols, probe, device=matrix.device, dtype=matrix.dtype)
        for _ in range(self.niter):
            sample = matrix @ self._orthonormalize(matrix.transpose(-2, -1) @ self._orthonormalize(sample))
        basis = self._orthonormalize(sample)
        projected = basis.transpose(-2, -1) @ matrix
        u, s, v = torch.svd_lowrank(projected, q=min(projected.shape[-2], projected.shape[-1]), niter=1)
        return basis @ u, s, v


def _import_entropack():
    try:
        import entropack
    except ImportError as error:
        raise ImportError(
            'entropack is required for method="entropack"; install it with pip install "entropack[cuda13]"'
        ) from error
    return entropack


class EntroPack(Scheme):
    """EntroPack entropy coding: ``lossy`` (rate controlled), ``dfloat11`` or ``tile_ans`` (lossless)."""

    name = "entropack"

    def __init__(self, scheme="lossy", target_bpp=4.0, execution_backend="auto", row_width=128, min_numel=4096):
        if scheme not in ("lossy", "dfloat11", "tile_ans"):
            raise ValueError(f"scheme must be lossy/dfloat11/tile_ans, got {scheme!r}")
        if not 0.001 <= target_bpp <= 11:
            raise ValueError(f"target_bpp must be in [0.001, 11], got {target_bpp}")
        self.ep = _import_entropack()
        self.scheme = scheme
        self.target_bpp = target_bpp
        self.execution_backend = execution_backend
        self.row_width = row_width
        self.min_numel = min_numel
        self.config = self._make_config()
        self.raw_config = self.ep.RawConfig(execution_backend=execution_backend)

    def _make_config(self):
        backend = self.execution_backend
        if self.scheme == "lossy":
            return self.ep.EntroPackLossyConfig(target_bpp=self.target_bpp, execution_backend=backend)
        if self.scheme == "dfloat11":
            return self.ep.DFloat11Config(execution_backend=backend)
        return self.ep.TileANSConfig(execution_backend=backend)

    def _to_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        # EntroPack wants a 2-D matrix; narrow rows lose rate-control precision, so group them.
        matrix = tensor.reshape(-1, tensor.shape[-1]) if tensor.ndim >= 2 else tensor.reshape(1, -1)
        width = matrix.shape[-1]
        if 0 < width < self.row_width and matrix.shape[0] > 1:
            group = min(self.row_width // width, matrix.shape[0])
            while group > 1 and matrix.shape[0] % group:
                group -= 1
            if group > 1:
                matrix = matrix.reshape(-1, width * group)
        return matrix.contiguous()

    def compress(self, tensors):
        buffers, shapes = [], []
        for tensor in tensors:
            matrix = self._to_2d(tensor)
            config = self.config if matrix.numel() >= self.min_numel else self.raw_config
            buffers.append(self.ep.compress(matrix, config))
            shapes.append(tuple(tensor.shape))
        return buffers, {"shapes": shapes}

    def decompress(self, buffers, meta):
        return [self.ep.decompress(buffer, self.config).reshape(shape) for buffer, shape in zip(buffers, meta["shapes"])]

    def __str__(self):
        return f"entropack({self.scheme}@{self.target_bpp}bpp)"


register_scheme("svd", SVD)
register_scheme("cholesky_qr", CholeskyQR)
register_scheme("entropack", EntroPack)
