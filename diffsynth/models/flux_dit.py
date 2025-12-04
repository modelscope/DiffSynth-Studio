import torch
from .general_modules import TimestepEmbeddings, AdaLayerNorm, RMSNorm
from einops import rearrange


def interact_with_ipadapter(hidden_states, q, ip_k, ip_v, scale=1.0):
    batch_size, num_tokens = hidden_states.shape[0:2]
    ip_hidden_states = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v)
    ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, num_tokens, -1)
    hidden_states = hidden_states + scale * ip_hidden_states
    return hidden_states


class RoPEEmbedding(torch.nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim


    def rope(self, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."

        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
        omega = 1.0 / (theta**scale)

        batch_size, seq_length = pos.shape
        out = torch.einsum("...n,d->...nd", pos, omega)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)

        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
        return out.float()


    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([self.rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)



class FluxJointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_b = RMSNorm(head_dim, eps=1e-6)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)


    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        batch_size = hidden_states_a.shape[0]

        # Part A
        qkv_a = self.a_to_qkv(hidden_states_a)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        # Part B
        qkv_b = self.b_to_qkv(hidden_states_b)
        qkv_b = qkv_b.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_b, k_b, v_b = qkv_b.chunk(3, dim=1)
        q_b, k_b = self.norm_q_b(q_b), self.norm_k_b(k_b)

        q = torch.concat([q_b, q_a], dim=2)
        k = torch.concat([k_b, k_a], dim=2)
        v = torch.concat([v_b, v_a], dim=2)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = hidden_states[:, :hidden_states_b.shape[1]], hidden_states[:, hidden_states_b.shape[1]:]
        if ipadapter_kwargs_list is not None:
            hidden_states_a = interact_with_ipadapter(hidden_states_a, q_a, **ipadapter_kwargs_list)
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            return hidden_states_a, hidden_states_b



class FluxJointTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = FluxJointAttention(dim, dim, num_attention_heads, dim // num_attention_heads)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )

        self.norm2_b = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_b = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )


    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b, image_rotary_emb, attn_mask, ipadapter_kwargs_list)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b



class FluxSingleAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)

        self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(head_dim, eps=1e-6)


    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


    def forward(self, hidden_states, image_rotary_emb):
        batch_size = hidden_states.shape[0]

        qkv_a = self.a_to_qkv(hidden_states)
        qkv_a = qkv_a.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q_a, k_a, v = qkv_a.chunk(3, dim=1)
        q_a, k_a = self.norm_q_a(q_a), self.norm_k_a(k_a)

        q, k = self.apply_rope(q_a, k_a, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states



class AdaLayerNormSingle(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, 3 * dim, bias=True)
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)


    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa



class FluxSingleTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.dim = dim

        self.norm = AdaLayerNormSingle(dim)
        self.to_qkv_mlp = torch.nn.Linear(dim, dim * (3 + 4))
        self.norm_q_a = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(self.head_dim, eps=1e-6)

        self.proj_out = torch.nn.Linear(dim * 5, dim)


    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


    def process_attention(self, hidden_states, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        batch_size = hidden_states.shape[0]

        qkv = hidden_states.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        q, k = self.norm_q_a(q), self.norm_k_a(k)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        if ipadapter_kwargs_list is not None:
            hidden_states = interact_with_ipadapter(hidden_states, q, **ipadapter_kwargs_list)
        return hidden_states


    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb, attn_mask=None, ipadapter_kwargs_list=None):
        residual = hidden_states_a
        norm_hidden_states, gate = self.norm(hidden_states_a, emb=temb)
        hidden_states_a = self.to_qkv_mlp(norm_hidden_states)
        attn_output, mlp_hidden_states = hidden_states_a[:, :, :self.dim * 3], hidden_states_a[:, :, self.dim * 3:]

        attn_output = self.process_attention(attn_output, image_rotary_emb, attn_mask, ipadapter_kwargs_list)
        mlp_hidden_states = torch.nn.functional.gelu(mlp_hidden_states, approximate="tanh")

        hidden_states_a = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states_a = gate.unsqueeze(1) * self.proj_out(hidden_states_a)
        hidden_states_a = residual + hidden_states_a

        return hidden_states_a, hidden_states_b



class AdaLayerNormContinuous(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(dim, dim * 2, bias=True)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x, conditioning):
        emb = self.linear(self.silu(conditioning))
        shift, scale = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x



class FluxDiT(torch.nn.Module):
    def __init__(self, disable_guidance_embedder=False, input_dim=64, num_blocks=19):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072)
        self.guidance_embedder = None if disable_guidance_embedder else TimestepEmbeddings(256, 3072)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(768, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072))
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.x_embedder = torch.nn.Linear(input_dim, 3072)

        self.blocks = torch.nn.ModuleList([FluxJointTransformerBlock(3072, 24) for _ in range(num_blocks)])
        self.single_blocks = torch.nn.ModuleList([FluxSingleTransformerBlock(3072, 24) for _ in range(38)])

        self.final_norm_out = AdaLayerNormContinuous(3072)
        self.final_proj_out = torch.nn.Linear(3072, 64)
        
        self.input_dim = input_dim


    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states


    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(hidden_states, "B (H W) (C P Q) -> B C (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
        return hidden_states


    def prepare_image_ids(self, latents):
        batch_size, _, height, width = latents.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(device=latents.device, dtype=latents.dtype)

        return latent_image_ids


    def construct_mask(self, entity_masks, prompt_seq_len, image_seq_len):
        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        total_seq_len = N * prompt_seq_len + image_seq_len
        patched_masks = [self.patchify(entity_masks[i]) for i in range(N)]
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        image_start = N * prompt_seq_len
        image_end = N * prompt_seq_len + image_seq_len
        # prompt-image mask
        for i in range(N):
            prompt_start = i * prompt_seq_len
            prompt_end = (i + 1) * prompt_seq_len
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, prompt_seq_len, 1)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt mask
        for i in range(N):
            for j in range(N):
                if i != j:
                    prompt_start_i = i * prompt_seq_len
                    prompt_end_i = (i + 1) * prompt_seq_len
                    prompt_start_j = j * prompt_seq_len
                    prompt_end_j = (j + 1) * prompt_seq_len
                    attention_mask[:, prompt_start_i:prompt_end_i, prompt_start_j:prompt_end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        return attention_mask


    def process_entity_masks(self, hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids, repeat_dim):
        max_masks = 0
        attention_mask = None
        prompt_embs = [prompt_emb]
        if entity_masks is not None:
            # entity_masks
            batch_size, max_masks = entity_masks.shape[0], entity_masks.shape[1]
            entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
            entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
            # global mask
            global_mask = torch.ones_like(entity_masks[0]).to(device=hidden_states.device, dtype=hidden_states.dtype)
            entity_masks = entity_masks + [global_mask] # append global to last
            # attention mask
            attention_mask = self.construct_mask(entity_masks, prompt_emb.shape[1], hidden_states.shape[1])
            attention_mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            attention_mask = attention_mask.unsqueeze(1)
            # embds: n_masks * b * seq * d
            local_embs = [entity_prompt_emb[:, i, None].squeeze(1) for i in range(max_masks)]
            prompt_embs = local_embs + prompt_embs # append global to last
        prompt_embs = [self.context_embedder(prompt_emb) for prompt_emb in prompt_embs]
        prompt_emb = torch.cat(prompt_embs, dim=1)

        # positional embedding
        text_ids = torch.cat([text_ids] * (max_masks + 1), dim=1)
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        return prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        hidden_states,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None,
        tiled=False, tile_size=128, tile_stride=64, entity_prompt_emb=None, entity_masks=None,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        # (Deprecated) The real forward is in `pipelines.flux_image`.
        return None
