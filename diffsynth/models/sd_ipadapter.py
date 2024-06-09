from .svd_image_encoder import SVDImageEncoder
from .sdxl_ipadapter import IpAdapterImageProjModel, IpAdapterModule, SDXLIpAdapterStateDictConverter
from transformers import CLIPImageProcessor
import torch


class IpAdapterCLIPImageEmbedder(SVDImageEncoder):
    def __init__(self):
        super().__init__()
        self.image_processor = CLIPImageProcessor()

    def forward(self, image):
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.embeddings.class_embedding.device, dtype=self.embeddings.class_embedding.dtype)
        return super().forward(pixel_values)


class SDIpAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shape_list = [(768, 320)] * 2 + [(768, 640)] * 2 + [(768, 1280)] * 5 + [(768, 640)] * 3  + [(768, 320)] * 3 + [(768, 1280)] * 1
        self.ipadapter_modules = torch.nn.ModuleList([IpAdapterModule(*shape) for shape in shape_list])
        self.image_proj = IpAdapterImageProjModel(cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4)
        self.set_full_adapter()

    def set_full_adapter(self):
        block_ids = [1, 4, 9, 12, 17, 20, 40, 43, 46, 50, 53, 56, 60, 63, 66, 29]
        self.call_block_id = {(i, 0): j for j, i in enumerate(block_ids)}

    def set_less_adapter(self):
        # IP-Adapter for SD v1.5 doesn't support this feature.
        self.set_full_adapter(self)

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
        return SDIpAdapterStateDictConverter()


class SDIpAdapterStateDictConverter(SDXLIpAdapterStateDictConverter):
    def __init__(self):
        pass
