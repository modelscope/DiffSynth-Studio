"""Tensor compression for the pipeline: a list of stages, applied forward and reversed back.

    from diffsynth.core.context_compress import compress_tensor

    store = compress_tensor(tensor, [{"method": "svd", "rank": 32}, {"method": "entropack", "target_bpp": 8.0}])
    tensor_again = store.decompress()

Built-in stages: ``svd`` (``torch.svd_lowrank``), ``cholesky_qr`` (the same randomized SVD with
a fast Cholesky-QR range finder) and ``entropack``. Register more with
:func:`register_scheme`. :mod:`.kv_cache` applies the pipeline to the QwenImage21 KV cache.
"""

from .pipeline import (
    CompressedTensor,
    CompressPipeline,
    Scheme,
    compress_tensor,
    decompress_tensor,
    register_scheme,
)
from .schemes import CholeskyQR, EntroPack, SVD
from .kv_cache import CompressedKVCacheBlock, KVCompressionStats, KVCompressor, AccumulatingKVCacheBlock, create_kv_cache_blocks

__all__ = [
    "CompressedTensor",
    "CompressPipeline",
    "Scheme",
    "compress_tensor",
    "decompress_tensor",
    "register_scheme",
    "SVD",
    "CholeskyQR",
    "EntroPack",
    "KVCompressor",
    "KVCompressionStats",
    "CompressedKVCacheBlock",
    "AccumulatingKVCacheBlock",
    "create_kv_cache_blocks",
]
