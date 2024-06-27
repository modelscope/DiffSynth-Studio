from .svd_image_encoder import SVDImageEncoder
from transformers import CLIPImageProcessor
import torch


class IpAdapterXLCLIPImageEmbedder(SVDImageEncoder):
    def __init__(self):
        super().__init__(embed_dim=1664, encoder_intermediate_size=8192, projection_dim=1280, num_encoder_layers=48, num_heads=16, head_dim=104)
        self.image_processor = CLIPImageProcessor()

    def forward(self, image):
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.embeddings.class_embedding.device, dtype=self.embeddings.class_embedding.dtype)
        return super().forward(pixel_values)


class IpAdapterImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=2048, clip_embeddings_dim=1280, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IpAdapterModule(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.to_k_ip = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.to_v_ip = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, hidden_states):
        ip_k = self.to_k_ip(hidden_states)
        ip_v = self.to_v_ip(hidden_states)
        return ip_k, ip_v


class SDXLIpAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shape_list = [(2048, 640)] * 4 + [(2048, 1280)] * 50 + [(2048, 640)] * 6 + [(2048, 1280)] * 10
        self.ipadapter_modules = torch.nn.ModuleList([IpAdapterModule(*shape) for shape in shape_list])
        self.image_proj = IpAdapterImageProjModel()
        self.set_full_adapter()

    def set_full_adapter(self):
        map_list = sum([
            [(7, i) for i in range(2)],
            [(10, i) for i in range(2)],
            [(15, i) for i in range(10)],
            [(18, i) for i in range(10)],
            [(25, i) for i in range(10)],
            [(28, i) for i in range(10)],
            [(31, i) for i in range(10)],
            [(35, i) for i in range(2)],
            [(38, i) for i in range(2)],
            [(41, i) for i in range(2)],
            [(21, i) for i in range(10)],
        ], [])
        self.call_block_id = {i: j for j, i in enumerate(map_list)}

    def set_less_adapter(self):
        map_list = sum([
            [(7, i) for i in range(2)],
            [(10, i) for i in range(2)],
            [(15, i) for i in range(10)],
            [(18, i) for i in range(10)],
            [(25, i) for i in range(10)],
            [(28, i) for i in range(10)],
            [(31, i) for i in range(10)],
            [(35, i) for i in range(2)],
            [(38, i) for i in range(2)],
            [(41, i) for i in range(2)],
            [(21, i) for i in range(10)],
        ], [])
        self.call_block_id = {i: j for j, i in enumerate(map_list) if j>=34 and j<44}

    def forward(self, hidden_states, scale=1.0):
        hidden_states = self.image_proj(hidden_states)
        hidden_states = hidden_states.view(1, -1, hidden_states.shape[-1])
        ip_kv_dict = {}
        for (block_id, transformer_id) in self.call_block_id:
            ipadapter_id = self.call_block_id[(block_id, transformer_id)]
            ip_k, ip_v = self.ipadapter_modules[ipadapter_id](hidden_states)
            if block_id not in ip_kv_dict:
                ip_kv_dict[block_id] = {}
            ip_kv_dict[block_id][transformer_id] = {
                "ip_k": ip_k,
                "ip_v": ip_v,
                "scale": scale
            }
        return ip_kv_dict

    def state_dict_converter(self):
        return SDXLIpAdapterStateDictConverter()


class SDXLIpAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for name in state_dict["ip_adapter"]:
            names = name.split(".")
            layer_id = str(int(names[0]) // 2)
            name_ = ".".join(["ipadapter_modules"] + [layer_id] + names[1:])
            state_dict_[name_] = state_dict["ip_adapter"][name]
        for name in state_dict["image_proj"]:
            name_ = "image_proj." + name
            state_dict_[name_] = state_dict["image_proj"][name]
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)

