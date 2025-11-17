import torch
from diffsynth.models.svd_unet import TemporalTimesteps


class MultiValueEncoder(torch.nn.Module):
    def __init__(self, encoders=()):
        super().__init__()
        self.encoders = torch.nn.ModuleList(encoders)

    def __call__(self, values, dtype):
        emb = []
        for encoder, value in zip(self.encoders, values):
            if value is not None:
                value = value.unsqueeze(0)
                emb.append(encoder(value, dtype))
        emb = torch.concat(emb, dim=0)
        return emb


class SingleValueEncoder(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=4096, prefer_len=32, computation_device=None):
        super().__init__()
        self.prefer_len = prefer_len
        self.prefer_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device)
        self.prefer_value_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(self.prefer_len, dim_out) 
        )
        self._initialize_weights()

    def _initialize_weights(self):
        last_linear = self.prefer_value_embedder[-1]
        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    def forward(self, value, dtype):
        value = value * 1000
        emb = self.prefer_proj(value).to(dtype)
        emb = self.prefer_value_embedder(emb).squeeze(0)
        base_embeddings = emb.expand(self.prefer_len, -1)
        positional_embedding = self.positional_embedding.to(dtype=base_embeddings.dtype, device=base_embeddings.device)
        learned_embeddings = base_embeddings + positional_embedding
        return learned_embeddings

    @staticmethod
    def state_dict_converter():
        return SingleValueEncoderStateDictConverter()


class SingleValueEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict

    def from_civitai(self, state_dict):
        return state_dict
