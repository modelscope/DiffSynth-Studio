from transformers import BertModel, BertConfig, T5EncoderModel, T5Config
import torch



class HunyuanDiTCLIPTextEncoder(BertModel):
    def __init__(self):
        config = BertConfig(
            _name_or_path = "",
            architectures = ["BertModel"],
            attention_probs_dropout_prob = 0.1,
            bos_token_id = 0,
            classifier_dropout = None,
            directionality = "bidi",
            eos_token_id = 2,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.1,
            hidden_size = 1024,
            initializer_range = 0.02,
            intermediate_size = 4096,
            layer_norm_eps = 1e-12,
            max_position_embeddings = 512,
            model_type = "bert",
            num_attention_heads = 16,
            num_hidden_layers = 24,
            output_past = True,
            pad_token_id = 0,
            pooler_fc_size = 768,
            pooler_num_attention_heads = 12,
            pooler_num_fc_layers = 3,
            pooler_size_per_head = 128,
            pooler_type = "first_token_transform",
            position_embedding_type = "absolute",
            torch_dtype = "float32",
            transformers_version = "4.37.2",
            type_vocab_size = 2,
            use_cache = True,
            vocab_size = 47020
        )
        super().__init__(config, add_pooling_layer=False)
        self.eval()

    def forward(self, input_ids, attention_mask, clip_skip=1):
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device

        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        all_hidden_states = encoder_outputs.hidden_states
        prompt_emb = all_hidden_states[-clip_skip]
        if clip_skip > 1:
            mean, std = all_hidden_states[-1].mean(), all_hidden_states[-1].std()
            prompt_emb = (prompt_emb - prompt_emb.mean()) / prompt_emb.std() * std + mean
        return prompt_emb

    def state_dict_converter(self):
        return HunyuanDiTCLIPTextEncoderStateDictConverter()



class HunyuanDiTT5TextEncoder(T5EncoderModel):
    def __init__(self):
        config = T5Config(
            _name_or_path = "../HunyuanDiT/t2i/mt5",
            architectures = ["MT5ForConditionalGeneration"],
            classifier_dropout = 0.0,
            d_ff = 5120,
            d_kv = 64,
            d_model = 2048,
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
            num_heads = 32,
            num_layers = 24,
            output_past = True,
            pad_token_id = 0,
            relative_attention_max_distance = 128,
            relative_attention_num_buckets = 32,
            tie_word_embeddings = False,
            tokenizer_class = "T5Tokenizer",
            transformers_version = "4.37.2",
            use_cache = True,
            vocab_size = 250112
        )
        super().__init__(config)
        self.eval()

    def forward(self, input_ids, attention_mask, clip_skip=1):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_emb = outputs.hidden_states[-clip_skip]
        if clip_skip > 1:
            mean, std = outputs.hidden_states[-1].mean(), outputs.hidden_states[-1].std()
            prompt_emb = (prompt_emb - prompt_emb.mean()) / prompt_emb.std() * std + mean
        return prompt_emb
    
    def state_dict_converter(self):
        return HunyuanDiTT5TextEncoderStateDictConverter()



class HunyuanDiTCLIPTextEncoderStateDictConverter():
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {name[5:]: param for name, param in state_dict.items() if name.startswith("bert.")}
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)


class HunyuanDiTT5TextEncoderStateDictConverter():
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("encoder.")}
        state_dict_["shared.weight"] = state_dict["shared.weight"]
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
