import torch, warnings, glob, os, types
from diffsynth.models.svd_unet import TemporalTimesteps

# class ValueEncoder(torch.nn.Module):
#     def __init__(self, dim_in, dim_out, prefer_len, computation_device=None):
#         super().__init__()
#         self.value_embedder_color = SingleValueEncoder(dim_in, dim_out, prefer_len, computation_device=None)
#         self.value_embedder_photo = SingleValueEncoder(dim_in, dim_out, prefer_len, computation_device=None)
#         self.value_embedder_detail = SingleValueEncoder(dim_in, dim_out, prefer_len, computation_device=None)
#         self.value_embedder_logic = SingleValueEncoder(dim_in, dim_out, prefer_len, computation_device=None)
#         self.value_embedder = {
#             'color': self.value_embedder_color,
#             'photo': self.value_embedder_photo,
#             'detail': self.value_embedder_detail,
#             'logic': self.value_embedder_logic,
#         }
    

class SingleValueEncoder(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=3072, prefer_len=32, computation_device=None):
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
        emb = self.prefer_proj(value).to(dtype)
        emb = self.prefer_value_embedder(emb).squeeze(0)
        base_embeddings = emb.expand(self.prefer_len, -1)
        learned_embeddings = base_embeddings + self.positional_embedding  # [8, dim_out]
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
