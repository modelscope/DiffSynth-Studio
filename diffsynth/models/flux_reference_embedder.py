from .sd3_dit import TimestepEmbeddings
from .flux_dit import RoPEEmbedding
import torch
from einops import repeat


class FluxReferenceEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.idx_embedder = TimestepEmbeddings(256, 256)
        self.proj = torch.nn.Linear(3072, 3072)
        
    def forward(self, image_ids, idx, dtype, device):
        pos_emb = self.pos_embedder(image_ids, device=device)
        idx_emb = self.idx_embedder(idx, dtype=dtype).to(device)
        length = pos_emb.shape[2]
        pos_emb = repeat(pos_emb, "B N L C H W -> 1 N (B L) C H W")
        idx_emb = repeat(idx_emb, "B (C H W) -> 1 1 (B L) C H W", C=64, H=2, W=2, L=length)
        image_rotary_emb = pos_emb + idx_emb
        return image_rotary_emb
    
    def init(self):
        self.idx_embedder.timestep_embedder[-1].load_state_dict({
            "weight": torch.zeros((256, 256)),
            "bias": torch.zeros((256,))
        }),
        self.proj.load_state_dict({
            "weight": torch.eye(3072),
            "bias": torch.zeros((3072,))
        })
