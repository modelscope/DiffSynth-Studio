import torch
from .sd3_dit import TimestepEmbeddings, AdaLayerNorm
from einops import rearrange
from .tiler import TileWorker



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
    


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((dim,)))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype) * self.weight
        return hidden_states
    


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


    def forward(self, hidden_states_a, hidden_states_b, image_rotary_emb):
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

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_b, hidden_states_a = hidden_states[:, :hidden_states_b.shape[1]], hidden_states[:, hidden_states_b.shape[1]:]
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


    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b, image_rotary_emb)

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
        # self.proj_in = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), torch.nn.GELU(approximate="tanh"))
        # self.attn = FluxSingleAttention(dim, dim, num_attention_heads, dim // num_attention_heads)
        self.linear = torch.nn.Linear(dim, dim * (3 + 4))
        self.norm_q_a = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k_a = RMSNorm(self.head_dim, eps=1e-6)

        self.proj_out = torch.nn.Linear(dim * 5, dim)


    def apply_rope(self, xq, xk, freqs_cis):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

    
    def process_attention(self, hidden_states, image_rotary_emb):
        batch_size = hidden_states.shape[0]

        qkv = hidden_states.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        q, k = self.norm_q_a(q), self.norm_k_a(k)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states


    def forward(self, hidden_states_a, hidden_states_b, temb, image_rotary_emb):
        residual = hidden_states_a
        norm_hidden_states, gate = self.norm(hidden_states_a, emb=temb)
        hidden_states_a = self.linear(norm_hidden_states)
        attn_output, mlp_hidden_states = hidden_states_a[:, :, :self.dim * 3], hidden_states_a[:, :, self.dim * 3:]

        attn_output = self.process_attention(attn_output, image_rotary_emb)
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
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None] + shift[:, None]
        return x



class FluxDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072)
        self.guidance_embedder = TimestepEmbeddings(256, 3072)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(768, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072))
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.x_embedder = torch.nn.Linear(64, 3072)

        self.blocks = torch.nn.ModuleList([FluxJointTransformerBlock(3072, 24) for _ in range(19)])
        self.single_blocks = torch.nn.ModuleList([FluxSingleTransformerBlock(3072, 24) for _ in range(38)])

        self.norm_out = AdaLayerNormContinuous(3072)
        self.proj_out = torch.nn.Linear(3072, 64)


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
    

    def tiled_forward(
        self,
        hidden_states,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids,
        tile_size=128, tile_stride=64,
        **kwargs
    ):
        # Due to the global positional embedding, we cannot implement layer-wise tiled forward.
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x, timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None),
            hidden_states,
            tile_size,
            tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )
        return hidden_states


    def forward(
        self,
        hidden_states,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None,
        tiled=False, tile_size=128, tile_stride=64,
        **kwargs
    ):
        if tiled:
            return self.tiled_forward(
                hidden_states,
                timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids,
                tile_size=tile_size, tile_stride=tile_stride,
                **kwargs
            )
        
        if image_ids is None:
            image_ids = self.prepare_image_ids(hidden_states)
        
        conditioning = self.time_embedder(timestep, hidden_states.dtype)\
                     + self.guidance_embedder(guidance, hidden_states.dtype)\
                     + self.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.context_embedder(prompt_emb)
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        height, width = hidden_states.shape[-2:]
        hidden_states = self.patchify(hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        
        for block in self.blocks:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)

        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for block in self.single_blocks:
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        hidden_states = self.norm_out(hidden_states, conditioning)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.unpatchify(hidden_states, height, width)

        return hidden_states


    @staticmethod
    def state_dict_converter():
        return FluxDiTStateDictConverter()
    


class FluxDiTStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "context_embedder": "context_embedder",
            "x_embedder": "x_embedder",
            "time_text_embed.timestep_embedder.linear_1": "time_embedder.timestep_embedder.0",
            "time_text_embed.timestep_embedder.linear_2": "time_embedder.timestep_embedder.2",
            "time_text_embed.guidance_embedder.linear_1": "guidance_embedder.timestep_embedder.0",
            "time_text_embed.guidance_embedder.linear_2": "guidance_embedder.timestep_embedder.2",
            "time_text_embed.text_embedder.linear_1": "pooled_text_embedder.0",
            "time_text_embed.text_embedder.linear_2": "pooled_text_embedder.2",
            "norm_out.linear": "norm_out.linear",
            "proj_out": "proj_out",

            "norm1.linear": "norm1_a.linear",
            "norm1_context.linear": "norm1_b.linear",
            "attn.to_q": "attn.a_to_q",
            "attn.to_k": "attn.a_to_k",
            "attn.to_v": "attn.a_to_v",
            "attn.to_out.0": "attn.a_to_out",
            "attn.add_q_proj": "attn.b_to_q",
            "attn.add_k_proj": "attn.b_to_k",
            "attn.add_v_proj": "attn.b_to_v",
            "attn.to_add_out": "attn.b_to_out",
            "ff.net.0.proj": "ff_a.0",
            "ff.net.2": "ff_a.2",
            "ff_context.net.0.proj": "ff_b.0",
            "ff_context.net.2": "ff_b.2",
            "attn.norm_q": "attn.norm_q_a",
            "attn.norm_k": "attn.norm_k_a",
            "attn.norm_added_q": "attn.norm_q_b",
            "attn.norm_added_k": "attn.norm_k_b",
        }
        rename_dict_single = {
            "attn.to_q": "a_to_q",
            "attn.to_k": "a_to_k",
            "attn.to_v": "a_to_v",
            "attn.norm_q": "norm_q_a",
            "attn.norm_k": "norm_k_a",
            "norm.linear": "norm.linear",
            "proj_mlp": "proj_in_besides_attn",
            "proj_out": "proj_out",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            elif name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[:-len(suffix)]
                if prefix in rename_dict:
                    state_dict_[rename_dict[prefix] + suffix] = param
                elif prefix.startswith("transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict:
                        name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                elif prefix.startswith("single_transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "single_blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict_single:
                        name_ = ".".join(names[:2] + [rename_dict_single[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                    else:
                        print(name)
                else:
                    print(name)
        for name in list(state_dict_.keys()):
            if ".proj_in_besides_attn." in name:
                name_ = name.replace(".proj_in_besides_attn.", ".linear.")
                param = torch.concat([
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_q.")],
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_k.")],
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_v.")],
                    state_dict_[name],
                ], dim=0)
                state_dict_[name_] = param
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_q."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_k."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_v."))
                state_dict_.pop(name)
        for name in list(state_dict_.keys()):
            for component in ["a", "b"]:
                if f".{component}_to_q." in name:
                    name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                    param = torch.concat([
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                    ], dim=0)
                    state_dict_[name_] = param
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "time_in.in_layer.bias": "time_embedder.timestep_embedder.0.bias",
            "time_in.in_layer.weight": "time_embedder.timestep_embedder.0.weight",
            "time_in.out_layer.bias": "time_embedder.timestep_embedder.2.bias",
            "time_in.out_layer.weight": "time_embedder.timestep_embedder.2.weight",
            "txt_in.bias": "context_embedder.bias",
            "txt_in.weight": "context_embedder.weight",
            "vector_in.in_layer.bias": "pooled_text_embedder.0.bias",
            "vector_in.in_layer.weight": "pooled_text_embedder.0.weight",
            "vector_in.out_layer.bias": "pooled_text_embedder.2.bias",
            "vector_in.out_layer.weight": "pooled_text_embedder.2.weight",
            "final_layer.linear.bias": "proj_out.bias",
            "final_layer.linear.weight": "proj_out.weight",
            "guidance_in.in_layer.bias": "guidance_embedder.timestep_embedder.0.bias",
            "guidance_in.in_layer.weight": "guidance_embedder.timestep_embedder.0.weight",
            "guidance_in.out_layer.bias": "guidance_embedder.timestep_embedder.2.bias",
            "guidance_in.out_layer.weight": "guidance_embedder.timestep_embedder.2.weight",
            "img_in.bias": "x_embedder.bias",
            "img_in.weight": "x_embedder.weight",
            "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
            "final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        }
        suffix_rename_dict = {
            "img_attn.norm.key_norm.scale": "attn.norm_k_a.weight",
            "img_attn.norm.query_norm.scale": "attn.norm_q_a.weight",
            "img_attn.proj.bias": "attn.a_to_out.bias",
            "img_attn.proj.weight": "attn.a_to_out.weight",
            "img_attn.qkv.bias": "attn.a_to_qkv.bias",
            "img_attn.qkv.weight": "attn.a_to_qkv.weight",
            "img_mlp.0.bias": "ff_a.0.bias",
            "img_mlp.0.weight": "ff_a.0.weight",
            "img_mlp.2.bias": "ff_a.2.bias",
            "img_mlp.2.weight": "ff_a.2.weight",
            "img_mod.lin.bias": "norm1_a.linear.bias",
            "img_mod.lin.weight": "norm1_a.linear.weight",
            "txt_attn.norm.key_norm.scale": "attn.norm_k_b.weight",
            "txt_attn.norm.query_norm.scale": "attn.norm_q_b.weight",
            "txt_attn.proj.bias": "attn.b_to_out.bias",
            "txt_attn.proj.weight": "attn.b_to_out.weight",
            "txt_attn.qkv.bias": "attn.b_to_qkv.bias",
            "txt_attn.qkv.weight": "attn.b_to_qkv.weight",
            "txt_mlp.0.bias": "ff_b.0.bias",
            "txt_mlp.0.weight": "ff_b.0.weight",
            "txt_mlp.2.bias": "ff_b.2.bias",
            "txt_mlp.2.weight": "ff_b.2.weight",
            "txt_mod.lin.bias": "norm1_b.linear.bias",
            "txt_mod.lin.weight": "norm1_b.linear.weight",

            "linear1.bias": "linear.bias",
            "linear1.weight": "linear.weight",
            "linear2.bias": "proj_out.bias",
            "linear2.weight": "proj_out.weight",
            "modulation.lin.bias": "norm.linear.bias",
            "modulation.lin.weight": "norm.linear.weight",
            "norm.key_norm.scale": "norm_k_a.weight",
            "norm.query_norm.scale": "norm_q_a.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            names = name.split(".")
            if name in rename_dict:
                rename = rename_dict[name]
                if name.startswith("final_layer.adaLN_modulation.1."):
                    param = torch.concat([param[3072:], param[:3072]], dim=0)
                state_dict_[rename] = param
            elif names[0] == "double_blocks":
                rename = f"blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                state_dict_[rename] = param
            elif names[0] == "single_blocks":
                if ".".join(names[2:]) in suffix_rename_dict:
                    rename = f"single_blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                    state_dict_[rename] = param
            else:
                print(name)
        return state_dict_
                