import torch
from typing import Optional, Union


class QwenImageTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel
        config = Qwen2_5_VLConfig(**{
            "architectures": [
                "Qwen2_5_VLForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_token_id": 151655,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 128000,
            "max_window_layers": 28,
            "model_type": "qwen2_5_vl",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "rope_type": "default",
                "type": "default"
            },
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "text_config": {
                "architectures": [
                    "Qwen2_5_VLForConditionalGeneration"
                ],
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "hidden_act": "silu",
                "hidden_size": 3584,
                "image_token_id": None,
                "initializer_range": 0.02,
                "intermediate_size": 18944,
                "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
                ],
                "max_position_embeddings": 128000,
                "max_window_layers": 28,
                "model_type": "qwen2_5_vl_text",
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "rope_type": "default",
                "type": "default"
                },
                "rope_theta": 1000000.0,
                "sliding_window": None,
                "torch_dtype": "float32",
                "use_cache": True,
                "use_sliding_window": False,
                "video_token_id": None,
                "vision_end_token_id": 151653,
                "vision_start_token_id": 151652,
                "vision_token_id": 151654,
                "vocab_size": 152064
            },
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "transformers_version": "4.54.0",
            "use_cache": True,
            "use_sliding_window": False,
            "video_token_id": 151656,
            "vision_config": {
                "depth": 32,
                "fullatt_block_indexes": [
                    7,
                    15,
                    23,
                    31
                ],
                "hidden_act": "silu",
                "hidden_size": 1280,
                "in_channels": 3,
                "in_chans": 3,
                "initializer_range": 0.02,
                "intermediate_size": 3420,
                "model_type": "qwen2_5_vl",
                "num_heads": 16,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
                "tokens_per_second": 2,
                "torch_dtype": "float32",
                "window_size": 112
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "vocab_size": 152064
        })
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.config = config
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        output_attentions = False
        output_hidden_states = True

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs.hidden_states
