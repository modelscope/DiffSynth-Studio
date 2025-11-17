from .svd_image_encoder import SVDImageEncoder
from .sd3_dit import RMSNorm
from transformers import CLIPImageProcessor
import torch


class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class IpAdapterModule(torch.nn.Module):
    def __init__(self, num_attention_heads, attention_head_dim, input_dim):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        output_dim = num_attention_heads * attention_head_dim
        self.to_k_ip = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.to_v_ip = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.norm_added_k = RMSNorm(attention_head_dim, eps=1e-5, elementwise_affine=False)
        

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        # ip_k
        ip_k = self.to_k_ip(hidden_states)
        ip_k = ip_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_k = self.norm_added_k(ip_k)
        # ip_v
        ip_v = self.to_v_ip(hidden_states)
        ip_v = ip_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return ip_k, ip_v


class FluxIpAdapter(torch.nn.Module):
    def __init__(self, num_attention_heads=24, attention_head_dim=128, cross_attention_dim=4096, num_tokens=128, num_blocks=57):
        super().__init__()
        self.ipadapter_modules = torch.nn.ModuleList([IpAdapterModule(num_attention_heads, attention_head_dim, cross_attention_dim) for _ in range(num_blocks)])
        self.image_proj = MLPProjModel(cross_attention_dim=cross_attention_dim, id_embeddings_dim=1152, num_tokens=num_tokens)
        self.set_adapter()

    def set_adapter(self):
        self.call_block_id = {i:i for i in range(len(self.ipadapter_modules))}

    def forward(self, hidden_states, scale=1.0):
        hidden_states = self.image_proj(hidden_states)
        hidden_states = hidden_states.view(1, -1, hidden_states.shape[-1])
        ip_kv_dict = {}
        for block_id in self.call_block_id:
            ipadapter_id = self.call_block_id[block_id]
            ip_k, ip_v = self.ipadapter_modules[ipadapter_id](hidden_states)
            ip_kv_dict[block_id] = {
                "ip_k": ip_k,
                "ip_v": ip_v,
                "scale": scale
            }
        return ip_kv_dict

    @staticmethod
    def state_dict_converter():
        return FluxIpAdapterStateDictConverter()


class FluxIpAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for name in state_dict["ip_adapter"]:
            name_ = 'ipadapter_modules.' + name
            state_dict_[name_] = state_dict["ip_adapter"][name]
        for name in state_dict["image_proj"]:
            name_ = "image_proj." + name
            state_dict_[name_] = state_dict["image_proj"][name]
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
