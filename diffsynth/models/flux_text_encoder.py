import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from .sd_text_encoder import SDTextEncoder


class FLUXTextEncoder1(SDTextEncoder):
    def __init__(self, vocab_size=49408):
        super().__init__(vocab_size=vocab_size)

    def forward(self, input_ids, clip_skip=2):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                hidden_states = embeds
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        return embeds, pooled_embeds

    @staticmethod
    def state_dict_converter():
        return FLUXTextEncoder1StateDictConverter()

class FLUXTextEncoder2(T5EncoderModel):
    def __init__(self):
        config = T5Config(
            _name_or_path = ".",
            architectures = ["T5EncoderModel"],
            classifier_dropout = 0.0,
            d_ff = 10240,
            d_kv = 64,
            d_model = 4096,
            decoder_start_token_id = 0,
            dense_act_fn = "gelu_new",
            dropout_rate = 0.1,
            eos_token_id = 1,
            feed_forward_proj = "gated-gelu",
            initializer_factor = 1.0,
            is_encoder_decoder = True,
            is_gated_act = True,
            layer_norm_epsilon = 1e-06,
            model_type = "t5",
            num_decoder_layers = 24,
            num_heads = 64,
            num_layers = 24,
            output_past = True,
            pad_token_id = 0,
            relative_attention_max_distance = 128,
            relative_attention_num_buckets = 32,
            tie_word_embeddings = False,
            torch_dtype = bfloat16,  # change
            transformers_version = "4.43.3",  # change
            use_cache = True,
            vocab_size = 32128
        )
        super().__init__(config)
        self.eval()

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids)
        prompt_emb = outputs.last_hidden_state
        return prompt_emb

    @staticmethod
    def state_dict_converter():
        return FLUXTextEncoder2StateDictConverter()


class FLUXTextEncoder1StateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "text_model.embeddings.position_embedding.weight": "position_embeds",
            "text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "text_model.final_layer_norm.bias": "final_layer_norm.bias"
        }
        attn_rename_dict = {
            "self_attn.q_proj": "attn.to_q",
            "self_attn.k_proj": "attn.to_k",
            "self_attn.v_proj": "attn.to_v",
            "self_attn.out_proj": "attn.to_out",
            "layer_norm1": "layer_norm1",
            "layer_norm2": "layer_norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
            elif name.startswith("text_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)

class FLUXTextEncoder2StateDictConverter():
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = state_dict
        return state_dict_

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)