from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipVisionConfig
from transformers import SiglipImageProcessor
import torch


class Siglip2ImageEncoder(SiglipVisionTransformer):
    def __init__(self):
        config = SiglipVisionConfig(
            attention_dropout = 0.0,
            dtype = "float32",
            hidden_act = "gelu_pytorch_tanh",
            hidden_size = 1536,
            image_size = 384,
            intermediate_size = 6144,
            layer_norm_eps = 1e-06,
            model_type = "siglip_vision_model",
            num_attention_heads = 16,
            num_channels = 3,
            num_hidden_layers = 40,
            patch_size = 16,
            transformers_version = "4.56.1",
            _attn_implementation = "sdpa"
        )
        super().__init__(config)
        self.processor = SiglipImageProcessor(
            do_convert_rgb = None,
            do_normalize = True,
            do_rescale = True,
            do_resize = True,
            image_mean = [
                0.5,
                0.5,
                0.5
            ],
            image_processor_type = "SiglipImageProcessor",
            image_std = [
                0.5,
                0.5,
                0.5
            ],
            processor_class = "SiglipProcessor",
            resample = 2,
            rescale_factor = 0.00392156862745098,
            size = {
                "height": 384,
                "width": 384
            }
        )
        
    def forward(self, image, torch_dtype=torch.bfloat16, device="cuda"):
        pixel_values = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=device, dtype=torch_dtype)
        output_attentions = False
        output_hidden_states = False
        interpolate_pos_encoding = False

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return pooler_output
