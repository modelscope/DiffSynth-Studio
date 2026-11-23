# HiUni: a Qwen3-VL backbone shared by understanding, generation and editing, with a Pixel Diffusion Head and HiGate.
import math
import torch
import torch.nn.functional as F


HIUNI_QWEN3_VL_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 151643, "dtype": "bfloat16", "eos_token_id": 151645,
        "head_dim": 128, "hidden_act": "silu", "hidden_size": 2560, "initializer_range": 0.02, "intermediate_size": 9728,
        "max_position_embeddings": 262144, "model_type": "qwen3_vl_text", "num_attention_heads": 32, "num_hidden_layers": 36,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_theta": 5000000, "tie_word_embeddings": True, "use_cache": True,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024, "in_channels": 3,
        "initializer_range": 0.02, "intermediate_size": 4096, "model_type": "qwen3_vl", "num_heads": 16, "num_position_embeddings": 2304,
        "out_hidden_size": 2560, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}
HIUNI_TIME_TOKEN_ID = 151661


def merge_patches(raw, height, width):
    # Raster patches [4*h*w, d] -> merged 2x2 blocks [h*w, 4*d] laid out as [patch_channel, block_row, block_column].
    dim = raw.shape[-1]
    return raw.reshape(height, 2, width, 2, dim).permute(0, 2, 4, 1, 3).reshape(-1, dim * 4)


def unmerge_patches(merged, height, width):
    dim = merged.shape[-1] // 4
    return merged.reshape(height, width, dim, 2, 2).permute(0, 3, 1, 4, 2).reshape(-1, dim)


def vision_patches(raw, height, width, patch_size=16):
    # Raster RGB patches -> Qwen3-VL's block-major layout with the frame duplicated for temporal_patch_size=2.
    patches = raw.reshape(height, 2, width, 2, 3, patch_size, patch_size).permute(0, 2, 1, 3, 4, 5, 6)
    return patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1).reshape(4 * height * width, 6 * patch_size * patch_size)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def modulate(x, shift, scale):
    return (x.float() * (1 + scale.float()) + shift.float()).to(x.dtype)


class HiUniRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HiUniRotaryEmbedding(torch.nn.Module):
    # Interleaved 3D M-RoPE, identical to the Qwen3-VL text rotary embedding but with the head's own head_dim.
    def __init__(self, head_dim, theta=5000000.0, mrope_section=(24, 20, 20)):
        super().__init__()
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ position_ids[:, :, None, :].float()).transpose(2, 3)
            freqs_t = freqs[0].clone()
            for dim, offset in ((1, 1), (2, 2)):
                idx = slice(offset, self.mrope_section[dim] * 3, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]
            emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class HiUniAttention(torch.nn.Module):
    def __init__(self, hidden_size=2560, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, hidden_size // num_heads
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = HiUniRMSNorm(self.head_dim)
        self.k_norm = HiUniRMSNorm(self.head_dim)
        self.rotary_emb = HiUniRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask, position_ids):
        b, s, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2))
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        n_rep = self.num_heads // self.num_kv_heads
        k = k[:, :, None].expand(b, self.num_kv_heads, n_rep, s, self.head_dim).reshape(b, self.num_heads, s, self.head_dim)
        v = v[:, :, None].expand(b, self.num_kv_heads, n_rep, s, self.head_dim).reshape(b, self.num_heads, s, self.head_dim)
        out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attention_mask)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, -1))


class HiUniMLP(torch.nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9728):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HiUniDiffusionBlock(torch.nn.Module):
    # Pre-norm transformer block of the Pixel Diffusion Head; time conditioning enters only through the final layer.
    def __init__(self, hidden_size=2560, num_heads=32, num_kv_heads=8, intermediate_size=9728):
        super().__init__()
        self.self_attn = HiUniAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = HiUniMLP(hidden_size, intermediate_size)
        self.input_layernorm = HiUniRMSNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = HiUniRMSNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask, position_ids):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_ids)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HiUniFinalLayer(torch.nn.Module):
    def __init__(self, hidden_size=2560, out_channels=3072):
        super().__init__()
        self.norm_final = HiUniRMSNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, time_emb, gen_mask):
        shift, scale = self.adaLN_modulation(time_emb).chunk(2, dim=-1)
        shift_full, scale_full = torch.zeros_like(x), torch.zeros_like(x)
        shift_full[gen_mask] = shift.to(x.dtype)
        scale_full[gen_mask] = scale.to(x.dtype)
        return self.linear(modulate(self.norm_final(x), shift_full, scale_full))


class HiUniTimestepEmbedder(torch.nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(256, hidden_size), torch.nn.GELU(), torch.nn.Linear(hidden_size, hidden_size))

    def forward(self, timesteps):
        # timesteps are on the training scale [0, 1000].
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timesteps.device).float() / 128)
        phase = timesteps.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((phase.cos(), phase.sin()), dim=-1).to(self.mlp[0].weight.dtype))


class HiUniGate(torch.nn.Module):
    # HiGate: a residual pathway that adds normalized pixel features back to the backbone states.
    def __init__(self, dim=2560):
        super().__init__()
        self.gate_fc1 = torch.nn.Linear(dim, dim)
        self.gate_fc2 = torch.nn.Linear(dim, 1)
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.hi_info_norm = torch.nn.LayerNorm(dim)

    def forward(self, low, high):
        gate = self.gate_fc2(F.silu(self.gate_fc1(low)))
        return self.layer_norm(low + gate * self.hi_info_norm(high))


class HiUniModel(torch.nn.Module):
    # Keys follow the HiUni checkpoint: model.visual.*, model.language_model.*, time_token.*, time_embeddings.*,
    # diffusion_head_a.N.*, diffusion_head_b.*, pixel_proj.*, pixel_dit_gate.*. lm_head is tied to embed_tokens.
    def __init__(self, num_diffusion_layers=10, gate_inject_layer=3):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLModel
        config = Qwen3VLConfig(**HIUNI_QWEN3_VL_CONFIG)
        for sub_config in (config, config.text_config, config.vision_config):
            sub_config._attn_implementation = "sdpa"
        self.config = config
        self.model = Qwen3VLModel(config)
        dim = config.text_config.hidden_size
        self.time_token = HiUniTimestepEmbedder(dim)
        self.time_embeddings = HiUniTimestepEmbedder(dim)
        self.diffusion_head_a = torch.nn.ModuleList([HiUniDiffusionBlock(dim) for _ in range(num_diffusion_layers)])
        self.diffusion_head_b = HiUniFinalLayer(dim, 3072)
        self.pixel_proj = torch.nn.Linear(3072, dim)
        self.pixel_dit_gate = HiUniGate(dim)
        self.gate_inject_layer = gate_inject_layer
        # Rotary inv_freq buffers stay in float32 under a bf16 load; a whole-module `.to(dtype)` would downcast them.
        self._fp32_buffers = {name: buffer.detach().clone() for name, buffer in self.named_buffers() if buffer.dtype == torch.float32}

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._restore_fp32_buffers()
        return result

    def _restore_fp32_buffers(self):
        for name, snapshot in self._fp32_buffers.items():
            *path, attribute = name.split(".")
            module = self
            for part in path:
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            current = getattr(module, attribute, None)
            if current is not None and current.dtype != torch.float32:
                setattr(module, attribute, snapshot.to(device=current.device))

    @property
    def dtype(self):
        return self.pixel_proj.weight.dtype

    def visual(self, pixel_values, grid_thw):
        # Merged ViT features without DeepStack; transformers 4.x returns a tuple, 5.x an output object.
        output = self.model.visual(pixel_values.to(self.dtype), grid_thw=grid_thw)
        return output.pooler_output if hasattr(output, "pooler_output") else output[0]

    def backbone(self, embeddings, attention_mask, position_ids):
        return self.model.language_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                         position_ids=position_ids, use_cache=False).last_hidden_state

    def text_logits(self, input_ids, position_ids, attention_mask, image_features=None, num_images=0):
        # Understanding / chat: clean images, time tokens embed t=0, logits of the last position.
        embeddings = self.model.language_model.embed_tokens(input_ids)
        if image_features is not None:
            embeddings = embeddings.clone()
            embeddings[input_ids.eq(151655)] = image_features.to(embeddings.dtype)
            times = self.time_token(torch.zeros(num_images, device=input_ids.device))
            embeddings[input_ids.eq(HIUNI_TIME_TOKEN_ID)] = times.to(embeddings.dtype)
        hidden = self.backbone(embeddings, attention_mask, position_ids)
        return F.linear(hidden[:, -1], self.model.language_model.embed_tokens.weight)

    def forward(self, input_ids, position_ids, attention_mask, latents, timestep, height, width,
                ref_features=None, ref_raster=None, ref_height=None, ref_width=None):
        # One CFG branch. latents: raster patches [4*height*width, 768] where height/width count merged 32px blocks.
        # timestep is on the [0, 1000] scale. Returns predicted clean patches in the same layout (float32).
        dtype = self.dtype
        pad = input_ids.eq(151655)
        pad_index = pad[0].nonzero(as_tuple=True)[0]
        ref_mask = torch.zeros_like(pad)
        has_ref = ref_features is not None and pad_index.numel() == height * width + ref_height * ref_width
        if has_ref:
            ref_mask[0, pad_index[:ref_height * ref_width]] = True
        gen_mask = pad & ~ref_mask
        grid = torch.tensor([[1, height * 2, width * 2]], device=latents.device)
        gen_features = self.visual(vision_patches(latents.to(dtype), height, width), grid)
        embeddings = self.model.language_model.embed_tokens(input_ids).clone()
        if has_ref:
            embeddings[ref_mask] = ref_features.to(embeddings.dtype)
        embeddings[gen_mask] = gen_features.to(embeddings.dtype)
        embeddings[input_ids.eq(HIUNI_TIME_TOKEN_ID)] = self.time_token(timestep).to(embeddings.dtype)
        # Generation runs without DeepStack injection, as in training.
        hidden = self.backbone(embeddings, attention_mask, position_ids)
        time_emb = self.time_embeddings(timestep).expand(height * width, -1)
        gen_high = self.pixel_proj(merge_patches(latents.to(dtype), height, width))
        ref_high = self.pixel_proj(merge_patches(ref_raster.to(dtype), ref_height, ref_width)) if has_ref else None
        for index, layer in enumerate(self.diffusion_head_a):
            if index == self.gate_inject_layer:
                if ref_high is not None:
                    hidden[ref_mask] = self.pixel_dit_gate(hidden[ref_mask], ref_high).to(hidden.dtype)
                hidden[gen_mask] = self.pixel_dit_gate(hidden[gen_mask], gen_high).to(hidden.dtype)
            hidden = layer(hidden, attention_mask, position_ids)
        output = self.diffusion_head_b(hidden, time_emb, gen_mask)[gen_mask]
        return unmerge_patches(output, height, width).float()
