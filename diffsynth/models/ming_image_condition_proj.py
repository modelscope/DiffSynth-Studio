import torch
from torch import nn


class MingImageRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states


class MingImageConditionProj(nn.Module):
    def __init__(
        self,
        llm_hidden_size=2048,
        connector_hidden_size=1536,
        diffusion_c_input_dim=2560,
        diffusion_inner_dim=3840,
        directvlm_dim=6144,
        img_gen_scales=[16],
        selected_hidden_states_layers=[5, 12, 20],
    ):
        super().__init__()
        self.img_gen_scales = img_gen_scales
        self.selected_hidden_states_layers = selected_hidden_states_layers
        self.query_tokens_dict = nn.ParameterDict()
        for scale in img_gen_scales:
            self.query_tokens_dict[f"{scale}x{scale}"] = nn.Parameter(
                torch.nn.functional.normalize(torch.randn(scale * scale, llm_hidden_size), dim=-1)
            )
        self.proj_in = nn.Linear(llm_hidden_size, connector_hidden_size)
        self.proj_out = nn.Linear(connector_hidden_size, diffusion_c_input_dim)
        self.proj_directvlm = nn.Sequential(
            MingImageRMSNorm(directvlm_dim, eps=1e-5),
            nn.Linear(directvlm_dim, diffusion_inner_dim, bias=True),
        )
