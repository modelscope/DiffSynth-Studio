import torch


class CLIPAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, bias=True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.embed_dim = hidden_size
        self.scale = attention_head_dim ** -0.5

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, embed_dim = hidden_states.shape
        src_len = tgt_len

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, tgt_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, src_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, src_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = attn_weights.float().to(dtype=torch.float32)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(dtype=query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.activation_fn = quick_gelu if hidden_act == "quick_gelu" else torch.nn.functional.gelu
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, intermediate_size, hidden_act, layer_norm_eps):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_attention_heads, attention_head_dim, bias=True)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, attention_head_dim, intermediate_size, hidden_act, layer_norm_eps):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            CLIPEncoderLayer(hidden_size, num_attention_heads, attention_head_dim, intermediate_size, hidden_act, layer_norm_eps)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask=None, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


class CLIPTextEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids=None, position_ids=None):
        seq_length = input_ids.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        return token_embeds + position_embeds


class CLIPTextTransformer(torch.nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                 num_attention_heads=12, max_position_embeddings=77, vocab_size=49408,
                 layer_norm_eps=1e-05, hidden_act="quick_gelu"):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = CLIPEncoder(num_hidden_layers, hidden_size, num_attention_heads,
                                   hidden_size // num_attention_heads, intermediate_size, hidden_act, layer_norm_eps)
        self.final_layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, output_hidden_states=False):
        embeds = self.embeddings(input_ids, position_ids=position_ids)

        causal_mask = self._build_causal_mask(embeds.shape[1], device=embeds.device, dtype=embeds.dtype)
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = extended_mask.to(dtype=causal_mask.dtype)
            causal_mask = causal_mask + extended_mask

        encoder_outputs, all_hidden_states = self.encoder(
            embeds, attention_mask=causal_mask, output_hidden_states=output_hidden_states,
        )
        last_hidden_state = self.final_layer_norm(encoder_outputs)

        return last_hidden_state, all_hidden_states

    def _build_causal_mask(self, seq_len, device, dtype):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]


class CLIPTextModel(torch.nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                 num_attention_heads=12, max_position_embeddings=77, vocab_size=49408,
                 layer_norm_eps=1e-05, hidden_act="quick_gelu"):
        super().__init__()
        self.text_model = CLIPTextTransformer(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings, vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps, hidden_act=hidden_act,
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                output_hidden_states=False):
        last_hidden_state, all_hidden_states = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, output_hidden_states=output_hidden_states,
        )
        return last_hidden_state, all_hidden_states


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class SDTextEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        vocab_size=49408,
        layer_norm_eps=1e-05,
        hidden_act="quick_gelu",
    ):
        super().__init__()
        self.model = CLIPTextModel(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=True,
    ):
        last_hidden_state, all_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            return last_hidden_state, all_hidden_states
        return last_hidden_state
