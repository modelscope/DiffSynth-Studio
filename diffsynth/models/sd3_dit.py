import torch
from einops import rearrange
from .svd_unet import TemporalTimesteps
from .tiler import TileWorker



class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states



class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=2, in_channels=16, embed_dim=1536, pos_embed_max_size=192):
        super().__init__()
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.pos_embed_max_size, self.pos_embed_max_size, embed_dim))

    def cropped_pos_embed(self, height, width):
        height = height // self.patch_size
        width = width // self.patch_size
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed[:, top : top + height, left : left + width, :].flatten(1, 2)
        return spatial_pos_embed

    def forward(self, latent):
        height, width = latent.shape[-2:]
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        pos_embed = self.cropped_pos_embed(height, width)
        return latent + pos_embed



class TimestepEmbeddings(torch.nn.Module):
    def __init__(self, dim_in, dim_out, computation_device=None):
        super().__init__()
        self.time_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device)
        self.timestep_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )

    def forward(self, timestep, dtype):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb



class AdaLayerNorm(torch.nn.Module):
    def __init__(self, dim, single=False, dual=False):
        super().__init__()
        self.single = single
        self.dual = dual
        self.linear = torch.nn.Linear(dim, dim * [[6, 2][single], 9][dual])
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        elif self.dual:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.unsqueeze(1).chunk(9, dim=2)
            norm_x = self.norm(x)
            x = norm_x * (1 + scale_msa) + shift_msa
            norm_x2 = norm_x * (1 + scale_msa2) + shift_msa2
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_x2, gate_msa2
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp



class JointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False, use_rms_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)

        if use_rms_norm:
            self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
            self.norm_q_b = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_b = RMSNorm(head_dim, eps=1e-6)
        else:
            self.norm_q_a = None
            self.norm_k_a = None
            self.norm_q_b = None
            self.norm_k_b = None


    def process_qkv(self, hidden_states, to_qkv, norm_q, norm_k):
        batch_size = hidden_states.shape[0]
        qkv = to_qkv(hidden_states)
        qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        if norm_q is not None:
            q = norm_q(q)
        if norm_k is not None:
            k = norm_k(k)
        return q, k, v


    def forward(self, hidden_states_a, hidden_states_b):
        batch_size = hidden_states_a.shape[0]

        qa, ka, va = self.process_qkv(hidden_states_a, self.a_to_qkv, self.norm_q_a, self.norm_k_a)
        qb, kb, vb = self.process_qkv(hidden_states_b, self.b_to_qkv, self.norm_q_b, self.norm_k_b)
        q = torch.concat([qa, qb], dim=2)
        k = torch.concat([ka, kb], dim=2)
        v = torch.concat([va, vb], dim=2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states_a, hidden_states_b = hidden_states[:, :hidden_states_a.shape[1]], hidden_states[:, hidden_states_a.shape[1]:]
        hidden_states_a = self.a_to_out(hidden_states_a)
        if self.only_out_a:
            return hidden_states_a
        else:
            hidden_states_b = self.b_to_out(hidden_states_b)
            return hidden_states_a, hidden_states_b
        


class SingleAttention(torch.nn.Module):
    def __init__(self, dim_a, num_heads, head_dim, use_rms_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.a_to_out = torch.nn.Linear(dim_a, dim_a)

        if use_rms_norm:
            self.norm_q_a = RMSNorm(head_dim, eps=1e-6)
            self.norm_k_a = RMSNorm(head_dim, eps=1e-6)
        else:
            self.norm_q_a = None
            self.norm_k_a = None


    def process_qkv(self, hidden_states, to_qkv, norm_q, norm_k):
        batch_size = hidden_states.shape[0]
        qkv = to_qkv(hidden_states)
        qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)
        if norm_q is not None:
            q = norm_q(q)
        if norm_k is not None:
            k = norm_k(k)
        return q, k, v


    def forward(self, hidden_states_a):
        batch_size = hidden_states_a.shape[0]
        q, k, v = self.process_qkv(hidden_states_a, self.a_to_qkv, self.norm_q_a, self.norm_k_a)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.a_to_out(hidden_states)
        return hidden_states
        


class DualTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, use_rms_norm=False):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, dual=True)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)
        self.attn2 = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)

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


    def forward(self, hidden_states_a, hidden_states_b, temb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a, norm_hidden_states_a_2, gate_msa_a_2 = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        hidden_states_a = hidden_states_a + gate_msa_a_2 * self.attn2(norm_hidden_states_a_2)
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b



class JointTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, use_rms_norm=False, dual=False):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim, dual=dual)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)
        if dual:
            self.attn2 = SingleAttention(dim, num_attention_heads, dim // num_attention_heads, use_rms_norm=use_rms_norm)

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


    def forward(self, hidden_states_a, hidden_states_b, temb):
        if self.norm1_a.dual:
            norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a, norm_hidden_states_a_2, gate_msa_a_2 = self.norm1_a(hidden_states_a, emb=temb)
        else:
            norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        if self.norm1_a.dual:
            hidden_states_a = hidden_states_a + gate_msa_a_2 * self.attn2(norm_hidden_states_a_2)
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b



class JointTransformerFinalBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads, use_rms_norm=False):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim, single=True)

        self.attn = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads, only_out_a=True, use_rms_norm=use_rms_norm)

        self.norm2_a = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_a = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )


    def forward(self, hidden_states_a, hidden_states_b, temb):
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        return hidden_states_a, hidden_states_b



class SD3DiT(torch.nn.Module):
    def __init__(self, embed_dim=1536, num_layers=24, use_rms_norm=False, num_dual_blocks=0, pos_embed_max_size=192):
        super().__init__()
        self.pos_embedder = PatchEmbed(patch_size=2, in_channels=16, embed_dim=embed_dim, pos_embed_max_size=pos_embed_max_size)
        self.time_embedder = TimestepEmbeddings(256, embed_dim)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(2048, embed_dim), torch.nn.SiLU(), torch.nn.Linear(embed_dim, embed_dim))
        self.context_embedder = torch.nn.Linear(4096, embed_dim)
        self.blocks = torch.nn.ModuleList([JointTransformerBlock(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm, dual=True) for _ in range(num_dual_blocks)]
                                          + [JointTransformerBlock(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm) for _ in range(num_layers-1-num_dual_blocks)]
                                          + [JointTransformerFinalBlock(embed_dim, embed_dim//64, use_rms_norm=use_rms_norm)])
        self.norm_out = AdaLayerNorm(embed_dim, single=True)
        self.proj_out = torch.nn.Linear(embed_dim, 64)

    def tiled_forward(self, hidden_states, timestep, prompt_emb, pooled_prompt_emb, tile_size=128, tile_stride=64):
        # Due to the global positional embedding, we cannot implement layer-wise tiled forward.
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x, timestep, prompt_emb, pooled_prompt_emb),
            hidden_states,
            tile_size,
            tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )
        return hidden_states

    def forward(self, hidden_states, timestep, prompt_emb, pooled_prompt_emb, tiled=False, tile_size=128, tile_stride=64, use_gradient_checkpointing=False):
        if tiled:
            return self.tiled_forward(hidden_states, timestep, prompt_emb, pooled_prompt_emb, tile_size, tile_stride)
        conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
        prompt_emb = self.context_embedder(prompt_emb)

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embedder(hidden_states)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, prompt_emb, conditioning,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning)
        
        hidden_states = self.norm_out(hidden_states, conditioning)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "B (H W) (P Q C) -> B C (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
        return hidden_states
        
    @staticmethod
    def state_dict_converter():
        return SD3DiTStateDictConverter()



class SD3DiTStateDictConverter:
    def __init__(self):
        pass

    def infer_architecture(self, state_dict):
        embed_dim = state_dict["blocks.0.ff_a.0.weight"].shape[1]
        num_layers = 100
        while num_layers > 0 and f"blocks.{num_layers-1}.ff_a.0.bias" not in state_dict:
            num_layers -= 1
        use_rms_norm = "blocks.0.attn.norm_q_a.weight" in state_dict
        num_dual_blocks = 0
        while f"blocks.{num_dual_blocks}.attn2.a_to_out.bias" in state_dict:
            num_dual_blocks += 1
        pos_embed_max_size = state_dict["pos_embedder.pos_embed"].shape[1]
        return {
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "use_rms_norm": use_rms_norm,
            "num_dual_blocks": num_dual_blocks,
            "pos_embed_max_size": pos_embed_max_size
        }

    def from_diffusers(self, state_dict):
        rename_dict = {
            "context_embedder": "context_embedder",
            "pos_embed.pos_embed": "pos_embedder.pos_embed",
            "pos_embed.proj": "pos_embedder.proj",
            "time_text_embed.timestep_embedder.linear_1": "time_embedder.timestep_embedder.0",
            "time_text_embed.timestep_embedder.linear_2": "time_embedder.timestep_embedder.2",
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
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                if name == "pos_embed.pos_embed":
                    param = param.reshape((1, 192, 192, param.shape[-1]))
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
        merged_keys = [name for name in state_dict_ if ".a_to_q." in name or ".b_to_q." in name]
        for key in merged_keys:
            param = torch.concat([
                state_dict_[key.replace("to_q", "to_q")],
                state_dict_[key.replace("to_q", "to_k")],
                state_dict_[key.replace("to_q", "to_v")],
            ], dim=0)
            name = key.replace("to_q", "to_qkv")
            state_dict_.pop(key.replace("to_q", "to_q"))
            state_dict_.pop(key.replace("to_q", "to_k"))
            state_dict_.pop(key.replace("to_q", "to_v"))
            state_dict_[name] = param
        return state_dict_, self.infer_architecture(state_dict_)
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "model.diffusion_model.context_embedder.bias": "context_embedder.bias",
            "model.diffusion_model.context_embedder.weight": "context_embedder.weight",
            "model.diffusion_model.final_layer.linear.bias": "proj_out.bias",
            "model.diffusion_model.final_layer.linear.weight": "proj_out.weight",

            "model.diffusion_model.pos_embed": "pos_embedder.pos_embed",
            "model.diffusion_model.t_embedder.mlp.0.bias": "time_embedder.timestep_embedder.0.bias",
            "model.diffusion_model.t_embedder.mlp.0.weight": "time_embedder.timestep_embedder.0.weight",
            "model.diffusion_model.t_embedder.mlp.2.bias": "time_embedder.timestep_embedder.2.bias",
            "model.diffusion_model.t_embedder.mlp.2.weight": "time_embedder.timestep_embedder.2.weight",
            "model.diffusion_model.x_embedder.proj.bias": "pos_embedder.proj.bias",
            "model.diffusion_model.x_embedder.proj.weight": "pos_embedder.proj.weight",
            "model.diffusion_model.y_embedder.mlp.0.bias": "pooled_text_embedder.0.bias",
            "model.diffusion_model.y_embedder.mlp.0.weight": "pooled_text_embedder.0.weight",
            "model.diffusion_model.y_embedder.mlp.2.bias": "pooled_text_embedder.2.bias",
            "model.diffusion_model.y_embedder.mlp.2.weight": "pooled_text_embedder.2.weight",
            
            "model.diffusion_model.joint_blocks.23.context_block.adaLN_modulation.1.weight": "blocks.23.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.23.context_block.adaLN_modulation.1.bias": "blocks.23.norm1_b.linear.bias",
            "model.diffusion_model.final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
            "model.diffusion_model.final_layer.adaLN_modulation.1.bias": "norm_out.linear.bias",
        }
        for i in range(40):
            rename_dict.update({
                f"model.diffusion_model.joint_blocks.{i}.context_block.adaLN_modulation.1.bias": f"blocks.{i}.norm1_b.linear.bias",
                f"model.diffusion_model.joint_blocks.{i}.context_block.adaLN_modulation.1.weight": f"blocks.{i}.norm1_b.linear.weight",
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.proj.bias": f"blocks.{i}.attn.b_to_out.bias",
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.proj.weight": f"blocks.{i}.attn.b_to_out.weight",
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.qkv.bias": [f'blocks.{i}.attn.b_to_q.bias', f'blocks.{i}.attn.b_to_k.bias', f'blocks.{i}.attn.b_to_v.bias'],
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.qkv.weight": [f'blocks.{i}.attn.b_to_q.weight', f'blocks.{i}.attn.b_to_k.weight', f'blocks.{i}.attn.b_to_v.weight'],
                f"model.diffusion_model.joint_blocks.{i}.context_block.mlp.fc1.bias": f"blocks.{i}.ff_b.0.bias",
                f"model.diffusion_model.joint_blocks.{i}.context_block.mlp.fc1.weight": f"blocks.{i}.ff_b.0.weight",
                f"model.diffusion_model.joint_blocks.{i}.context_block.mlp.fc2.bias": f"blocks.{i}.ff_b.2.bias",
                f"model.diffusion_model.joint_blocks.{i}.context_block.mlp.fc2.weight": f"blocks.{i}.ff_b.2.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.adaLN_modulation.1.bias": f"blocks.{i}.norm1_a.linear.bias",
                f"model.diffusion_model.joint_blocks.{i}.x_block.adaLN_modulation.1.weight": f"blocks.{i}.norm1_a.linear.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.proj.bias": f"blocks.{i}.attn.a_to_out.bias",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.proj.weight": f"blocks.{i}.attn.a_to_out.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.qkv.bias": [f'blocks.{i}.attn.a_to_q.bias', f'blocks.{i}.attn.a_to_k.bias', f'blocks.{i}.attn.a_to_v.bias'],
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.qkv.weight": [f'blocks.{i}.attn.a_to_q.weight', f'blocks.{i}.attn.a_to_k.weight', f'blocks.{i}.attn.a_to_v.weight'],
                f"model.diffusion_model.joint_blocks.{i}.x_block.mlp.fc1.bias": f"blocks.{i}.ff_a.0.bias",
                f"model.diffusion_model.joint_blocks.{i}.x_block.mlp.fc1.weight": f"blocks.{i}.ff_a.0.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.mlp.fc2.bias": f"blocks.{i}.ff_a.2.bias",
                f"model.diffusion_model.joint_blocks.{i}.x_block.mlp.fc2.weight": f"blocks.{i}.ff_a.2.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.ln_q.weight": f"blocks.{i}.attn.norm_q_a.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn.ln_k.weight": f"blocks.{i}.attn.norm_k_a.weight",
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.ln_q.weight": f"blocks.{i}.attn.norm_q_b.weight",
                f"model.diffusion_model.joint_blocks.{i}.context_block.attn.ln_k.weight": f"blocks.{i}.attn.norm_k_b.weight",

                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.ln_q.weight": f"blocks.{i}.attn2.norm_q_a.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.ln_k.weight": f"blocks.{i}.attn2.norm_k_a.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.qkv.weight": f"blocks.{i}.attn2.a_to_qkv.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.qkv.bias": f"blocks.{i}.attn2.a_to_qkv.bias",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.proj.weight": f"blocks.{i}.attn2.a_to_out.weight",
                f"model.diffusion_model.joint_blocks.{i}.x_block.attn2.proj.bias": f"blocks.{i}.attn2.a_to_out.bias",
            })
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "model.diffusion_model.pos_embed":
                    pos_embed_max_size = int(param.shape[1] ** 0.5 + 0.4)
                    param = param.reshape((1, pos_embed_max_size, pos_embed_max_size, param.shape[-1]))
                if isinstance(rename_dict[name], str):
                    state_dict_[rename_dict[name]] = param
                else:
                    name_ = rename_dict[name][0].replace(".a_to_q.", ".a_to_qkv.").replace(".b_to_q.", ".b_to_qkv.")
                    state_dict_[name_] = param
        extra_kwargs = self.infer_architecture(state_dict_)
        num_layers = extra_kwargs["num_layers"]
        for name in [
            f"blocks.{num_layers-1}.norm1_b.linear.weight", f"blocks.{num_layers-1}.norm1_b.linear.bias", "norm_out.linear.weight", "norm_out.linear.bias",
        ]:
            param = state_dict_[name]
            dim = param.shape[0] // 2
            param = torch.concat([param[dim:], param[:dim]], axis=0)
            state_dict_[name] = param
        return state_dict_, self.infer_architecture(state_dict_)
