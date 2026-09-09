from .eager import na3d


class EagerSdpaAttention:
    def __call__(self, attn, q, k, v):
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q, k = q.to(dtype=v.dtype), k.to(dtype=v.dtype)
        return na3d(q, k, v, kernel_size=attn.kernel_size, is_causal=None, scale=1.0)
