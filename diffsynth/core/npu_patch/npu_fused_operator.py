import torch
from ..device.npu_compatible_device import get_device_type
try:
    import torch_npu
except:
    pass


def rms_norm_forward_npu(self, hidden_states):
    "npu rms fused operator for RMSNorm.forward from diffsynth\models\general_modules.py"
    if hidden_states.dtype != self.weight.dtype:
        hidden_states = hidden_states.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.eps)[0]


def rms_norm_forward_transformers_npu(self, hidden_states):
    "npu rms fused operator for transformers"
    if hidden_states.dtype != self.weight.dtype:
        hidden_states = hidden_states.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


def rotary_emb_Zimage_npu(self, x_in: torch.Tensor, freqs_cis: torch.Tensor):
    "npu rope fused operator for Zimage"
    with torch.amp.autocast(get_device_type(), enabled=False):
        freqs_cis = freqs_cis.unsqueeze(2)
        cos, sin = torch.chunk(torch.view_as_real(freqs_cis), 2, dim=-1)
        cos = cos.expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = sin.expand(-1, -1, -1, -1, 2).flatten(-2)
        return torch_npu.npu_rotary_mul(x_in, cos, sin, rotary_mode="interleave").to(x_in)