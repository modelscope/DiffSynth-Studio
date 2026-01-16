from transformers import DINOv3ViTModel, DINOv3ViTImageProcessorFast
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTConfig
import torch

from ..core.device.npu_compatible_device import get_device_type


class DINOv3ImageEncoder(DINOv3ViTModel):
    def __init__(self):
        config = DINOv3ViTConfig(
            architectures = [
                "DINOv3ViTModel"
            ],
            attention_dropout = 0.0,
            drop_path_rate = 0.0,
            dtype = "float32",
            hidden_act = "silu",
            hidden_size = 4096,
            image_size = 224,
            initializer_range = 0.02,
            intermediate_size = 8192,
            key_bias = False,
            layer_norm_eps = 1e-05,
            layerscale_value = 1.0,
            mlp_bias = True,
            model_type = "dinov3_vit",
            num_attention_heads = 32,
            num_channels = 3,
            num_hidden_layers = 40,
            num_register_tokens = 4,
            patch_size = 16,
            pos_embed_jitter = None,
            pos_embed_rescale = 2.0,
            pos_embed_shift = None,
            proj_bias = True,
            query_bias = False,
            rope_theta = 100.0,
            transformers_version = "4.56.1",
            use_gated_mlp = True,
            value_bias = False
        )
        super().__init__(config)
        self.processor = DINOv3ViTImageProcessorFast(
            crop_size = None,
            data_format = "channels_first",
            default_to_square = True,
            device = None,
            disable_grouping = None,
            do_center_crop = None,
            do_convert_rgb = None,
            do_normalize = True,
            do_rescale = True,
            do_resize = True,
            image_mean = [
                0.485,
                0.456,
                0.406
            ],
            image_processor_type = "DINOv3ViTImageProcessorFast",
            image_std = [
                0.229,
                0.224,
                0.225
            ],
            input_data_format = None,
            resample = 2,
            rescale_factor = 0.00392156862745098,
            return_tensors = None,
            size = {
                "height": 224,
                "width": 224
            }
        )
        
    def forward(self, image, torch_dtype=torch.bfloat16, device=get_device_type()):
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(dtype=torch_dtype, device=device)
        bool_masked_pos = None
        head_mask = None
        
        pixel_values = pixel_values.to(torch_dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return pooled_output
