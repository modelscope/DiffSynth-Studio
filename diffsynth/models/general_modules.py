import torch, math


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    computation_device = None,
    align_dtype_to_timestep = False,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device if computation_device is None else computation_device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    if align_dtype_to_timestep:
        emb = emb.to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalTimesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, computation_device = None, scale=1, align_dtype_to_timestep=False):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.computation_device = computation_device
        self.scale = scale
        self.align_dtype_to_timestep = align_dtype_to_timestep

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            computation_device=self.computation_device,
            scale=self.scale,
            align_dtype_to_timestep=self.align_dtype_to_timestep,
        )
        return t_emb


class DiffusersCompatibleTimestepProj(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear_1 = torch.nn.Linear(dim_in, dim_out)
        self.act = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(dim_out, dim_out)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class TimestepEmbeddings(torch.nn.Module):
    def __init__(self, dim_in, dim_out, computation_device=None, diffusers_compatible_format=False, scale=1, align_dtype_to_timestep=False, use_additional_t_cond=False):
        super().__init__()
        self.time_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device, scale=scale, align_dtype_to_timestep=align_dtype_to_timestep)
        if diffusers_compatible_format:
            self.timestep_embedder = DiffusersCompatibleTimestepProj(dim_in, dim_out)
        else:
            self.timestep_embedder = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
            )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = torch.nn.Embedding(2, dim_out)

    def forward(self, timestep, dtype, addition_t_cond=None):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        if addition_t_cond is not None:
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=dtype)
            time_emb = time_emb + addition_t_emb
        return time_emb


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


class AdaLayerNorm(torch.nn.Module):
    def __init__(self, dim, single=False, dual=False):
        super().__init__()
        self.single = single
        self.dual = dual
        self.linear = torch.nn.Linear(dim, dim * [[6, 2][single], 9][dual])
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        elif self.dual:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.unsqueeze(1).chunk(9, dim=2)
            norm_x = self.norm(x)
            x = norm_x * (1 + scale_msa) + shift_msa
            norm_x2 = norm_x * (1 + scale_msa2) + shift_msa2
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_x2, gate_msa2
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
