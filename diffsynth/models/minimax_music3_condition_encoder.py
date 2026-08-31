import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMaxMusic3ConditionEncoder(nn.Module):

    def __init__(
        self,
        condition_hidden_dim: int = 4096,
        num_condition_layers: int = 8,
        out_dim: int = 2048,
        input_sampling_rate: int = 24000,
        input_hop_length: int = 960,
        output_sampling_rate: int = 44100,
        output_hop_length: int = 512,
    ):
        super().__init__()
        self.condition_hidden_dim = condition_hidden_dim
        self.num_condition_layers = num_condition_layers
        self.input_sampling_rate = input_sampling_rate
        self.input_hop_length = input_hop_length
        self.output_sampling_rate = output_sampling_rate
        self.output_hop_length = output_hop_length
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, self.num_condition_layers, self.condition_hidden_dim, num_frames)
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_states.dtype)
        hidden_states = torch.einsum("blht,l->bht", hidden_states, layer_weights)
        hidden_states = self.layer_scale.to(hidden_states.dtype) * hidden_states
        hidden_states = self.proj(hidden_states)
        latent_length = max(
            1,
            int(
                num_frames
                * self.output_sampling_rate
                / self.input_sampling_rate
                * self.input_hop_length
                / self.output_hop_length
            ),
        )
        hidden_states = F.interpolate(hidden_states, size=latent_length, mode="nearest")
        return hidden_states.transpose(1, 2)
