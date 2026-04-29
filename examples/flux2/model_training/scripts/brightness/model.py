import torch, math
from PIL import Image
import numpy as np


class SingleValueEncoder(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=4096, length=32):
        super().__init__()
        self.length = length
        self.prefer_value_embedder = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out))
        self.positional_embedding = torch.nn.Parameter(torch.randn(self.length, dim_out))

    def get_timestep_embedding(self, timesteps, embedding_dim, max_period=10000):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb

    def forward(self, value, dtype):
        emb = self.get_timestep_embedding(value * 1000, 256).to(dtype)
        emb = self.prefer_value_embedder(emb).squeeze(0)
        base_embeddings = emb.expand(self.length, -1)
        positional_embedding = self.positional_embedding.to(dtype=base_embeddings.dtype, device=base_embeddings.device)
        learned_embeddings = base_embeddings + positional_embedding
        return learned_embeddings


class ValueFormatModel(torch.nn.Module):
    def __init__(self, num_double_blocks=5, num_single_blocks=20, dim=3072, num_heads=24, length=512):
        super().__init__()
        self.block_names = [f"double_{i}" for i in range(num_double_blocks)] + [f"single_{i}" for i in range(num_single_blocks)]
        self.proj_k = torch.nn.ModuleDict({block_name: SingleValueEncoder(dim_out=dim, length=length) for block_name in self.block_names})
        self.proj_v = torch.nn.ModuleDict({block_name: SingleValueEncoder(dim_out=dim, length=length) for block_name in self.block_names})
        self.num_heads = num_heads
        self.length = length

    @torch.no_grad()
    def process_inputs(self, pipe, scale, **kwargs):
        return {"value": torch.Tensor([scale]).to(dtype=pipe.torch_dtype, device=pipe.device)}

    def forward(self, value, **kwargs):
        kv_cache = {}
        for block_name in self.block_names:
            k = self.proj_k[block_name](value, value.dtype)
            k = k.view(1, self.length, self.num_heads, -1)
            v = self.proj_v[block_name](value, value.dtype)
            v = v.view(1, self.length, self.num_heads, -1)
            kv_cache[block_name] = (k, v)
        return {"kv_cache": kv_cache}


class DataAnnotator:
    def __call__(self, image, **kwargs):
        image = Image.open(image)
        image = np.array(image)
        return {"scale": image.astype(np.float32).mean() / 255}


TEMPLATE_MODEL = ValueFormatModel
TEMPLATE_MODEL_PATH = None # You should modify this parameter after training
TEMPLATE_DATA_PROCESSOR = DataAnnotator