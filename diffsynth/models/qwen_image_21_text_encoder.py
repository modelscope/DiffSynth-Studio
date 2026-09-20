from typing import Optional

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration


class QwenImage21TextEncoder(torch.nn.Module):
    QWEN_IMAGE_21_TEXT_ENCODER_CONFIG = {
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "use_cache": True,
            "vocab_size": 151936,
        },
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_vl",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
    }

    def __init__(self):
        super().__init__()
        config = Qwen3VLConfig(**self.QWEN_IMAGE_21_TEXT_ENCODER_CONFIG)
        self.model = Qwen3VLForConditionalGeneration(config)
        self.config = config
        # transformers keeps rotary inv_freq buffers in float32 even under a bf16 load,
        # while a whole-module `.to(dtype)` would downcast them; snapshot and restore.
        self._fp32_buffers = {
            name: buffer.detach().clone()
            for name, buffer in self.model.named_buffers()
            if buffer.dtype == torch.float32
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._restore_fp32_buffers()
        return result

    def _restore_fp32_buffers(self):
        for name, snapshot in self._fp32_buffers.items():
            module = self.model
            *path, attribute = name.split(".")
            for part in path:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            current = getattr(module, attribute, None)
            if current is not None and current.dtype != torch.float32:
                setattr(module, attribute, snapshot.to(device=current.device))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs["return_dict"] = True
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )
        return outputs.last_hidden_state

    def forward_joyaiimage(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        pre_norm_output = [None]
        def hook_fn(module, args, kwargs_output=None):
            pre_norm_output[0] = args[0]
        self.model.model.language_model.norm.register_forward_hook(hook_fn)
        _ = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        return pre_norm_output[0]
