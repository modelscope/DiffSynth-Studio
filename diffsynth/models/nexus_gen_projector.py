import math
import torch
import torch.nn as nn
from typing import Optional, Tuple



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        from transformers.modeling_rope_utils import _compute_default_rope_parameters
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len


    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NexusGenImageEmbeddingMerger(nn.Module):
    def __init__(self, num_layers=1, out_channel=4096, expand_ratio=4, device='cpu'):
        super().__init__()
        from transformers import Qwen2_5_VLConfig
        from transformers.activations import ACT2FN
        config = Qwen2_5_VLConfig(**{
            "_name_or_path": "DiffSynth-Studio/Nexus-GenV2",
            "architectures": [
                "Qwen2_5_VLForConditionalGeneration"
            ],
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoConfig": "configuration_qwen2_5_vl.Qwen2_5_VLConfig",
                "AutoModel": "modeling_qwen2_5_vl.Qwen2_5_VLModel",
                "AutoModelForCausalLM": "modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration"
            },
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
            "pad_token_id": 151643,
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
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.49.0",
            "use_cache": False,
            "use_sliding_window": False,
            "video_token_id": 151656,
            "vision_config": {
                "hidden_size": 1280,
                "in_chans": 3,
                "model_type": "qwen2_5_vl",
                "spatial_patch_size": 14,
                "tokens_per_second": 2,
                "torch_dtype": "bfloat16"
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "vocab_size": 152064
        })
        self.config = config
        self.num_layers = num_layers
        self.layers = nn.ModuleList([Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(num_layers)])
        self.projector = nn.Sequential(Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                                       nn.Linear(config.hidden_size, out_channel * expand_ratio),
                                       Qwen2RMSNorm(out_channel * expand_ratio, eps=config.rms_norm_eps),
                                       ACT2FN[config.hidden_act], nn.Linear(out_channel * expand_ratio, out_channel),
                                       Qwen2RMSNorm(out_channel, eps=config.rms_norm_eps))
        self.base_grid = torch.tensor([[1, 72, 72]], device=device)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config, device=device)

    def get_position_ids(self, image_grid_thw):
        """
        Generates position ids for the input embeddings grid.
        modified from the qwen2_vl mrope.
        """
        batch_size = image_grid_thw.shape[0]
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        t, h, w = (
            image_grid_thw[0][0],
            image_grid_thw[0][1],
            image_grid_thw[0][2],
        )
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t.item(),
            h.item() // spatial_merge_size,
            w.item() // spatial_merge_size,
        )
        scale_h = self.base_grid[0][1].item() / h.item()
        scale_w = self.base_grid[0][2].item() / w.item()

        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
        time_tensor = expanded_range * self.config.vision_config.tokens_per_second
        t_index = time_tensor.long().flatten().to(image_grid_thw.device)
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten().to(image_grid_thw.device) * scale_h
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten().to(image_grid_thw.device) * scale_w
        # 3, B, L
        position_ids = torch.stack([t_index, h_index, w_index]).unsqueeze(0).repeat(batch_size, 1, 1).permute(1, 0, 2)
        return position_ids

    def forward(self, embeds, embeds_grid, ref_embeds=None, ref_embeds_grid=None):
        position_ids = self.get_position_ids(embeds_grid)
        hidden_states = embeds
        if ref_embeds is not None:
            position_ids_ref_embeds = self.get_position_ids(ref_embeds_grid)
            position_ids = torch.cat((position_ids, position_ids_ref_embeds), dim=-1)
            hidden_states = torch.cat((embeds, ref_embeds), dim=1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = self.projector(hidden_states)
        return hidden_states

    @staticmethod
    def state_dict_converter():
        return NexusGenMergerStateDictConverter()


class NexusGenMergerStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        merger_state_dict = {key.replace("embedding_merger.", ""): value for key, value in state_dict.items() if key.startswith('embedding_merger.')}
        return merger_state_dict


class NexusGenAdapter(nn.Module):
    """
    Adapter for Nexus-Gen generation decoder.
    """
    def __init__(self, input_dim=3584, output_dim=4096):
        super(NexusGenAdapter, self).__init__()
        self.adapter = nn.Sequential(nn.Linear(input_dim, output_dim),
                                     nn.LayerNorm(output_dim), nn.ReLU(),
                                     nn.Linear(output_dim, output_dim),
                                     nn.LayerNorm(output_dim))

    def forward(self, x):
        return self.adapter(x)

    @staticmethod
    def state_dict_converter():
        return NexusGenAdapterStateDictConverter()


class NexusGenAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        adapter_state_dict = {key: value for key, value in state_dict.items() if key.startswith('adapter.')}
        return adapter_state_dict
