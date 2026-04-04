import torch
from einops import rearrange


def low_version_attention(query, key, value, attn_bias=None):
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    return attn @ value


class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def interact_with_ipadapter(self, hidden_states, q, ip_k, ip_v, scale=1.0):
        batch_size = q.shape[0]
        ip_k = ip_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_v = ip_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v)
        hidden_states = hidden_states + scale * ip_hidden_states
        return hidden_states

    def torch_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if qkv_preprocessor is not None:
            q, k, v = qkv_preprocessor(q, k, v)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if ipadapter_kwargs is not None:
            hidden_states = self.interact_with_ipadapter(hidden_states, q, **ipadapter_kwargs)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states
    
    def xformers_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b f (n d) -> (b n) f d", n=self.num_heads)
        k = rearrange(k, "b f (n d) -> (b n) f d", n=self.num_heads)
        v = rearrange(v, "b f (n d) -> (b n) f d", n=self.num_heads)

        if attn_mask is not None:
            hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        else:
            import xformers.ops as xops
            hidden_states = xops.memory_efficient_attention(q, k, v)
        hidden_states = rearrange(hidden_states, "(b n) f d -> b f (n d)", n=self.num_heads)

        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)

        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None, ipadapter_kwargs=None, qkv_preprocessor=None):
        return self.torch_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attn_mask=attn_mask, ipadapter_kwargs=ipadapter_kwargs, qkv_preprocessor=qkv_preprocessor)





class CLIPEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_size, num_heads=12, head_dim=64, use_quick_gelu=True):
        super().__init__()
        self.attn = Attention(q_dim=embed_dim, num_heads=num_heads, head_dim=head_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, embed_dim)

        self.use_quick_gelu = use_quick_gelu

    def quickGELU(self, x):
        return x * torch.sigmoid(1.702 * x)
    
    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.use_quick_gelu:
            hidden_states = self.quickGELU(hidden_states)
        else:
            hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    

class SDTextEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=12, encoder_intermediate_size=3072):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=1):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
        embeds = self.final_layer_norm(embeds)
        return embeds
    
    @staticmethod
    def state_dict_converter():
        return SDTextEncoderStateDictConverter()


class SDTextEncoderStateDictConverter:
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
        rename_dict = {
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": "token_embedding.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": "encoders.0.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight": "encoders.0.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.bias": "encoders.0.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight": "encoders.0.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.bias": "encoders.0.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "encoders.0.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.bias": "encoders.0.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight": "encoders.0.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.bias": "encoders.0.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": "encoders.0.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias": "encoders.0.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight": "encoders.0.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.bias": "encoders.0.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": "encoders.0.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.bias": "encoders.0.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight": "encoders.0.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.bias": "encoders.1.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm1.weight": "encoders.1.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.bias": "encoders.1.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.layer_norm2.weight": "encoders.1.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.bias": "encoders.1.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc1.weight": "encoders.1.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.bias": "encoders.1.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.mlp.fc2.weight": "encoders.1.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.bias": "encoders.1.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.k_proj.weight": "encoders.1.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.bias": "encoders.1.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.out_proj.weight": "encoders.1.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.bias": "encoders.1.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.q_proj.weight": "encoders.1.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.bias": "encoders.1.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.1.self_attn.v_proj.weight": "encoders.1.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.bias": "encoders.10.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm1.weight": "encoders.10.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.bias": "encoders.10.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.layer_norm2.weight": "encoders.10.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.bias": "encoders.10.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc1.weight": "encoders.10.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.bias": "encoders.10.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.mlp.fc2.weight": "encoders.10.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.bias": "encoders.10.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.k_proj.weight": "encoders.10.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.bias": "encoders.10.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.out_proj.weight": "encoders.10.attn.to_out.weight",        
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.bias": "encoders.10.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.q_proj.weight": "encoders.10.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.bias": "encoders.10.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight": "encoders.10.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.bias": "encoders.11.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm1.weight": "encoders.11.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.bias": "encoders.11.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.layer_norm2.weight": "encoders.11.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.bias": "encoders.11.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc1.weight": "encoders.11.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.bias": "encoders.11.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.mlp.fc2.weight": "encoders.11.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.bias": "encoders.11.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight": "encoders.11.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.bias": "encoders.11.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.out_proj.weight": "encoders.11.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.bias": "encoders.11.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.q_proj.weight": "encoders.11.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.bias": "encoders.11.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.v_proj.weight": "encoders.11.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.bias": "encoders.2.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm1.weight": "encoders.2.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.bias": "encoders.2.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.layer_norm2.weight": "encoders.2.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.bias": "encoders.2.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc1.weight": "encoders.2.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.bias": "encoders.2.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.mlp.fc2.weight": "encoders.2.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.bias": "encoders.2.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.k_proj.weight": "encoders.2.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.bias": "encoders.2.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.out_proj.weight": "encoders.2.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.bias": "encoders.2.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.q_proj.weight": "encoders.2.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.bias": "encoders.2.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.2.self_attn.v_proj.weight": "encoders.2.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.bias": "encoders.3.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm1.weight": "encoders.3.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.bias": "encoders.3.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.layer_norm2.weight": "encoders.3.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.bias": "encoders.3.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight": "encoders.3.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.bias": "encoders.3.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc2.weight": "encoders.3.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.bias": "encoders.3.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.k_proj.weight": "encoders.3.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.bias": "encoders.3.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.out_proj.weight": "encoders.3.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.bias": "encoders.3.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.q_proj.weight": "encoders.3.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.bias": "encoders.3.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.3.self_attn.v_proj.weight": "encoders.3.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.bias": "encoders.4.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm1.weight": "encoders.4.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.bias": "encoders.4.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.layer_norm2.weight": "encoders.4.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.bias": "encoders.4.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc1.weight": "encoders.4.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.bias": "encoders.4.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.mlp.fc2.weight": "encoders.4.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.bias": "encoders.4.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.k_proj.weight": "encoders.4.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.bias": "encoders.4.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.out_proj.weight": "encoders.4.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.bias": "encoders.4.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.q_proj.weight": "encoders.4.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.bias": "encoders.4.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.4.self_attn.v_proj.weight": "encoders.4.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.bias": "encoders.5.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm1.weight": "encoders.5.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.bias": "encoders.5.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.layer_norm2.weight": "encoders.5.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.bias": "encoders.5.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc1.weight": "encoders.5.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.bias": "encoders.5.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.mlp.fc2.weight": "encoders.5.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.bias": "encoders.5.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.k_proj.weight": "encoders.5.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.bias": "encoders.5.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.out_proj.weight": "encoders.5.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.bias": "encoders.5.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight": "encoders.5.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.bias": "encoders.5.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.5.self_attn.v_proj.weight": "encoders.5.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.bias": "encoders.6.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm1.weight": "encoders.6.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.bias": "encoders.6.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.layer_norm2.weight": "encoders.6.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.bias": "encoders.6.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc1.weight": "encoders.6.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.bias": "encoders.6.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.mlp.fc2.weight": "encoders.6.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.bias": "encoders.6.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.k_proj.weight": "encoders.6.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.bias": "encoders.6.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.out_proj.weight": "encoders.6.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.bias": "encoders.6.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.q_proj.weight": "encoders.6.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.bias": "encoders.6.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.6.self_attn.v_proj.weight": "encoders.6.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.bias": "encoders.7.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm1.weight": "encoders.7.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.bias": "encoders.7.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.layer_norm2.weight": "encoders.7.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.bias": "encoders.7.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc1.weight": "encoders.7.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.bias": "encoders.7.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.mlp.fc2.weight": "encoders.7.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.bias": "encoders.7.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.k_proj.weight": "encoders.7.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.bias": "encoders.7.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.out_proj.weight": "encoders.7.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.bias": "encoders.7.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.q_proj.weight": "encoders.7.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.bias": "encoders.7.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.7.self_attn.v_proj.weight": "encoders.7.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.bias": "encoders.8.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm1.weight": "encoders.8.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.bias": "encoders.8.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.layer_norm2.weight": "encoders.8.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.bias": "encoders.8.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc1.weight": "encoders.8.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.bias": "encoders.8.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.mlp.fc2.weight": "encoders.8.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.bias": "encoders.8.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.k_proj.weight": "encoders.8.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.bias": "encoders.8.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.out_proj.weight": "encoders.8.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.bias": "encoders.8.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.q_proj.weight": "encoders.8.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.bias": "encoders.8.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.8.self_attn.v_proj.weight": "encoders.8.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.bias": "encoders.9.layer_norm1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm1.weight": "encoders.9.layer_norm1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.bias": "encoders.9.layer_norm2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.layer_norm2.weight": "encoders.9.layer_norm2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.bias": "encoders.9.fc1.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc1.weight": "encoders.9.fc1.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.bias": "encoders.9.fc2.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.mlp.fc2.weight": "encoders.9.fc2.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.bias": "encoders.9.attn.to_k.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.k_proj.weight": "encoders.9.attn.to_k.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.bias": "encoders.9.attn.to_out.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.out_proj.weight": "encoders.9.attn.to_out.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.bias": "encoders.9.attn.to_q.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.q_proj.weight": "encoders.9.attn.to_q.weight",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.bias": "encoders.9.attn.to_v.bias",
            "cond_stage_model.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight": "encoders.9.attn.to_v.weight",
            "cond_stage_model.transformer.text_model.final_layer_norm.bias": "final_layer_norm.bias",
            "cond_stage_model.transformer.text_model.final_layer_norm.weight": "final_layer_norm.weight",
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": "position_embeds"
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight":
                    param = param.reshape((1, param.shape[0], param.shape[1]))
                state_dict_[rename_dict[name]] = param
        return state_dict_



class LoRALayerBlock(torch.nn.Module):
    def __init__(self, L, dim_in, dim_out):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(1, L, dim_in))
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, lora_A, lora_B):
        x = self.x @ lora_A.T @ lora_B.T
        x = self.layer_norm(x)
        return x
    

class LoRAEmbedder(torch.nn.Module):
    def __init__(self, lora_patterns=None, L=1, out_dim=2048):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
            
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"]
            model_dict[name.replace(".", "___")] = LoRALayerBlock(L, dim[0], dim[1])
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
        proj_dict = {}
        for lora_pattern in lora_patterns:
            layer_type, dim = lora_pattern["type"], lora_pattern["dim"]
            if layer_type not in proj_dict:
                proj_dict[layer_type.replace(".", "___")] = torch.nn.Linear(dim[1], out_dim)
        self.proj_dict = torch.nn.ModuleDict(proj_dict)
        
        self.lora_patterns = lora_patterns
        
        
    def default_lora_patterns(self):
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": (3072, 9216), "attn.a_to_out": (3072, 3072), "ff_a.0": (3072, 12288), "ff_a.2": (12288, 3072), "norm1_a.linear": (3072, 18432),
            "attn.b_to_qkv": (3072, 9216), "attn.b_to_out": (3072, 3072), "ff_b.0": (3072, 12288), "ff_b.2": (12288, 3072), "norm1_b.linear": (3072, 18432),
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        lora_dict = {"to_qkv_mlp": (3072, 21504), "proj_out": (15360, 3072), "norm.linear": (3072, 9216)}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        return lora_patterns
        
    def forward(self, lora):
        lora_emb = []
        for lora_pattern in self.lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            lora_A = lora[name + ".lora_A.weight"]
            lora_B = lora[name + ".lora_B.weight"]
            lora_out = self.model_dict[name.replace(".", "___")](lora_A, lora_B)
            lora_out = self.proj_dict[layer_type.replace(".", "___")](lora_out)
            lora_emb.append(lora_out)
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb
    
    
class FluxLoRAEncoder(torch.nn.Module):
    def __init__(self, embed_dim=4096, encoder_intermediate_size=8192, num_encoder_layers=1, num_embeds_per_lora=16, num_special_embeds=1):
        super().__init__()
        self.num_embeds_per_lora = num_embeds_per_lora
        # embedder
        self.embedder = LoRAEmbedder(L=num_embeds_per_lora, out_dim=embed_dim)
        
        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size, num_heads=32, head_dim=128) for _ in range(num_encoder_layers)])

        # special embedding
        self.special_embeds = torch.nn.Parameter(torch.randn(1, num_special_embeds, embed_dim))
        self.num_special_embeds = num_special_embeds
        
        # final layer
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.final_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, lora):
        lora_embeds = self.embedder(lora)
        special_embeds = self.special_embeds.to(dtype=lora_embeds.dtype, device=lora_embeds.device)
        embeds = torch.concat([special_embeds, lora_embeds], dim=1)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds)
        embeds = embeds[:, :self.num_special_embeds]
        embeds = self.final_layer_norm(embeds)
        embeds = self.final_linear(embeds)
        return embeds
    
    @staticmethod
    def state_dict_converter():
        return FluxLoRAEncoderStateDictConverter()


class FluxLoRAEncoderStateDictConverter:
    def from_civitai(self, state_dict):
        return state_dict
