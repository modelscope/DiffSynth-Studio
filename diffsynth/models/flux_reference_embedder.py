from .sd3_dit import TimestepEmbeddings
from .flux_dit import RoPEEmbedding
import torch
from einops import repeat


class FluxReferenceEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.idx_embedder = TimestepEmbeddings(256, 256)
        
    def forward(self, image_ids, idx, dtype):
        pos_emb = self.pos_embedder(image_ids)
        idx_emb = self.idx_embedder(idx, dtype=dtype)
        length = pos_emb.shape[2]
        pos_emb = repeat(pos_emb, "B N L C H W -> 1 N (B L) C H W")
        idx_emb = repeat(idx_emb, "B (C H W) -> 1 1 (B L) C H W", C=64, H=2, W=2, L=length)
        image_rotary_emb = pos_emb + idx_emb
        return image_rotary_emb
