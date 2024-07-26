import torch
from einops import rearrange
from .svd_unet import TemporalTimesteps
from .tiler import TileWorker



class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=2, in_channels=16, embed_dim=1536, pos_embed_max_size=192):
        super().__init__()
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.pos_embed_max_size, self.pos_embed_max_size, 1536))

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
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.time_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )

    def forward(self, timestep, dtype):
        time_emb = self.time_proj(timestep).to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb



class AdaLayerNorm(torch.nn.Module):
    def __init__(self, dim, single=False):
        super().__init__()
        self.single = single
        self.linear = torch.nn.Linear(dim, dim * (2 if single else 6))
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            scale, shift = emb.unsqueeze(1).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale) + shift
            return x
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.unsqueeze(1).chunk(6, dim=2)
            x = self.norm(x) * (1 + scale_msa) + shift_msa
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp



class JointAttention(torch.nn.Module):
    def __init__(self, dim_a, dim_b, num_heads, head_dim, only_out_a=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.only_out_a = only_out_a

        self.a_to_qkv = torch.nn.Linear(dim_a, dim_a * 3)
        self.b_to_qkv = torch.nn.Linear(dim_b, dim_b * 3)

        self.a_to_out = torch.nn.Linear(dim_a, dim_a)
        if not only_out_a:
            self.b_to_out = torch.nn.Linear(dim_b, dim_b)

    def forward(self, hidden_states_a, hidden_states_b):
        batch_size = hidden_states_a.shape[0]

        qkv = torch.concat([self.a_to_qkv(hidden_states_a), self.b_to_qkv(hidden_states_b)], dim=1)
        qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=1)

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



class JointTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim)

        self.attn = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads)

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
        norm_hidden_states_a, gate_msa_a, shift_mlp_a, scale_mlp_a, gate_mlp_a = self.norm1_a(hidden_states_a, emb=temb)
        norm_hidden_states_b, gate_msa_b, shift_mlp_b, scale_mlp_b, gate_mlp_b = self.norm1_b(hidden_states_b, emb=temb)

        # Attention
        attn_output_a, attn_output_b = self.attn(norm_hidden_states_a, norm_hidden_states_b)

        # Part A
        hidden_states_a = hidden_states_a + gate_msa_a * attn_output_a
        norm_hidden_states_a = self.norm2_a(hidden_states_a) * (1 + scale_mlp_a) + shift_mlp_a
        hidden_states_a = hidden_states_a + gate_mlp_a * self.ff_a(norm_hidden_states_a)

        # Part B
        hidden_states_b = hidden_states_b + gate_msa_b * attn_output_b
        norm_hidden_states_b = self.norm2_b(hidden_states_b) * (1 + scale_mlp_b) + shift_mlp_b
        hidden_states_b = hidden_states_b + gate_mlp_b * self.ff_b(norm_hidden_states_b)

        return hidden_states_a, hidden_states_b



class JointTransformerFinalBlock(torch.nn.Module):
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        self.norm1_a = AdaLayerNorm(dim)
        self.norm1_b = AdaLayerNorm(dim, single=True)

        self.attn = JointAttention(dim, dim, num_attention_heads, dim // num_attention_heads, only_out_a=True)

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
    def __init__(self):
        super().__init__()
        self.pos_embedder = PatchEmbed(patch_size=2, in_channels=16, embed_dim=1536, pos_embed_max_size=192)
        self.time_embedder = TimestepEmbeddings(256, 1536)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(2048, 1536), torch.nn.SiLU(), torch.nn.Linear(1536, 1536))
        self.context_embedder = torch.nn.Linear(4096, 1536)
        self.blocks = torch.nn.ModuleList([JointTransformerBlock(1536, 24) for _ in range(23)] + [JointTransformerFinalBlock(1536, 24)])
        self.norm_out = AdaLayerNorm(1536, single=True)
        self.proj_out = torch.nn.Linear(1536, 64)

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
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                if name == "pos_embed.pos_embed":
                    param = param.reshape((1, 192, 192, 1536))
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
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "model.diffusion_model.context_embedder.bias": "context_embedder.bias",
            "model.diffusion_model.context_embedder.weight": "context_embedder.weight",
            "model.diffusion_model.final_layer.linear.bias": "proj_out.bias",
            "model.diffusion_model.final_layer.linear.weight": "proj_out.weight",
            "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias": "blocks.0.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.weight": "blocks.0.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.bias": "blocks.0.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.0.context_block.attn.proj.weight": "blocks.0.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.bias": ['blocks.0.attn.b_to_q.bias', 'blocks.0.attn.b_to_k.bias', 'blocks.0.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight": ['blocks.0.attn.b_to_q.weight', 'blocks.0.attn.b_to_k.weight', 'blocks.0.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.0.context_block.mlp.fc1.bias": "blocks.0.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.0.context_block.mlp.fc1.weight": "blocks.0.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.0.context_block.mlp.fc2.bias": "blocks.0.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.0.context_block.mlp.fc2.weight": "blocks.0.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.0.x_block.adaLN_modulation.1.bias": "blocks.0.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.0.x_block.adaLN_modulation.1.weight": "blocks.0.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.0.x_block.attn.proj.bias": "blocks.0.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight": "blocks.0.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.bias": ['blocks.0.attn.a_to_q.bias', 'blocks.0.attn.a_to_k.bias', 'blocks.0.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": ['blocks.0.attn.a_to_q.weight', 'blocks.0.attn.a_to_k.weight', 'blocks.0.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.0.x_block.mlp.fc1.bias": "blocks.0.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.0.x_block.mlp.fc1.weight": "blocks.0.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.0.x_block.mlp.fc2.bias": "blocks.0.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.0.x_block.mlp.fc2.weight": "blocks.0.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.1.context_block.adaLN_modulation.1.bias": "blocks.1.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.1.context_block.adaLN_modulation.1.weight": "blocks.1.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.1.context_block.attn.proj.bias": "blocks.1.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.1.context_block.attn.proj.weight": "blocks.1.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.1.context_block.attn.qkv.bias": ['blocks.1.attn.b_to_q.bias', 'blocks.1.attn.b_to_k.bias', 'blocks.1.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.1.context_block.attn.qkv.weight": ['blocks.1.attn.b_to_q.weight', 'blocks.1.attn.b_to_k.weight', 'blocks.1.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.1.context_block.mlp.fc1.bias": "blocks.1.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.1.context_block.mlp.fc1.weight": "blocks.1.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.1.context_block.mlp.fc2.bias": "blocks.1.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.1.context_block.mlp.fc2.weight": "blocks.1.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.1.x_block.adaLN_modulation.1.bias": "blocks.1.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.1.x_block.adaLN_modulation.1.weight": "blocks.1.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.1.x_block.attn.proj.bias": "blocks.1.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.1.x_block.attn.proj.weight": "blocks.1.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.1.x_block.attn.qkv.bias": ['blocks.1.attn.a_to_q.bias', 'blocks.1.attn.a_to_k.bias', 'blocks.1.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.1.x_block.attn.qkv.weight": ['blocks.1.attn.a_to_q.weight', 'blocks.1.attn.a_to_k.weight', 'blocks.1.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.1.x_block.mlp.fc1.bias": "blocks.1.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.1.x_block.mlp.fc1.weight": "blocks.1.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.1.x_block.mlp.fc2.bias": "blocks.1.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.1.x_block.mlp.fc2.weight": "blocks.1.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.10.context_block.adaLN_modulation.1.bias": "blocks.10.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.10.context_block.adaLN_modulation.1.weight": "blocks.10.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.10.context_block.attn.proj.bias": "blocks.10.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.10.context_block.attn.proj.weight": "blocks.10.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.10.context_block.attn.qkv.bias": ['blocks.10.attn.b_to_q.bias', 'blocks.10.attn.b_to_k.bias', 'blocks.10.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.10.context_block.attn.qkv.weight": ['blocks.10.attn.b_to_q.weight', 'blocks.10.attn.b_to_k.weight', 'blocks.10.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.10.context_block.mlp.fc1.bias": "blocks.10.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.10.context_block.mlp.fc1.weight": "blocks.10.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.10.context_block.mlp.fc2.bias": "blocks.10.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.10.context_block.mlp.fc2.weight": "blocks.10.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.10.x_block.adaLN_modulation.1.bias": "blocks.10.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.10.x_block.adaLN_modulation.1.weight": "blocks.10.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.10.x_block.attn.proj.bias": "blocks.10.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.10.x_block.attn.proj.weight": "blocks.10.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.10.x_block.attn.qkv.bias": ['blocks.10.attn.a_to_q.bias', 'blocks.10.attn.a_to_k.bias', 'blocks.10.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.10.x_block.attn.qkv.weight": ['blocks.10.attn.a_to_q.weight', 'blocks.10.attn.a_to_k.weight', 'blocks.10.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.10.x_block.mlp.fc1.bias": "blocks.10.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.10.x_block.mlp.fc1.weight": "blocks.10.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.10.x_block.mlp.fc2.bias": "blocks.10.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.10.x_block.mlp.fc2.weight": "blocks.10.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.11.context_block.adaLN_modulation.1.bias": "blocks.11.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.11.context_block.adaLN_modulation.1.weight": "blocks.11.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.11.context_block.attn.proj.bias": "blocks.11.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.11.context_block.attn.proj.weight": "blocks.11.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.11.context_block.attn.qkv.bias": ['blocks.11.attn.b_to_q.bias', 'blocks.11.attn.b_to_k.bias', 'blocks.11.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.11.context_block.attn.qkv.weight": ['blocks.11.attn.b_to_q.weight', 'blocks.11.attn.b_to_k.weight', 'blocks.11.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.11.context_block.mlp.fc1.bias": "blocks.11.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.11.context_block.mlp.fc1.weight": "blocks.11.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.11.context_block.mlp.fc2.bias": "blocks.11.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.11.context_block.mlp.fc2.weight": "blocks.11.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.11.x_block.adaLN_modulation.1.bias": "blocks.11.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.11.x_block.adaLN_modulation.1.weight": "blocks.11.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.11.x_block.attn.proj.bias": "blocks.11.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.11.x_block.attn.proj.weight": "blocks.11.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.11.x_block.attn.qkv.bias": ['blocks.11.attn.a_to_q.bias', 'blocks.11.attn.a_to_k.bias', 'blocks.11.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.11.x_block.attn.qkv.weight": ['blocks.11.attn.a_to_q.weight', 'blocks.11.attn.a_to_k.weight', 'blocks.11.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.11.x_block.mlp.fc1.bias": "blocks.11.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.11.x_block.mlp.fc1.weight": "blocks.11.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.11.x_block.mlp.fc2.bias": "blocks.11.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.11.x_block.mlp.fc2.weight": "blocks.11.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.12.context_block.adaLN_modulation.1.bias": "blocks.12.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.12.context_block.adaLN_modulation.1.weight": "blocks.12.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.12.context_block.attn.proj.bias": "blocks.12.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.12.context_block.attn.proj.weight": "blocks.12.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.12.context_block.attn.qkv.bias": ['blocks.12.attn.b_to_q.bias', 'blocks.12.attn.b_to_k.bias', 'blocks.12.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.12.context_block.attn.qkv.weight": ['blocks.12.attn.b_to_q.weight', 'blocks.12.attn.b_to_k.weight', 'blocks.12.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.12.context_block.mlp.fc1.bias": "blocks.12.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.12.context_block.mlp.fc1.weight": "blocks.12.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.12.context_block.mlp.fc2.bias": "blocks.12.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.12.context_block.mlp.fc2.weight": "blocks.12.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.12.x_block.adaLN_modulation.1.bias": "blocks.12.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.12.x_block.adaLN_modulation.1.weight": "blocks.12.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.12.x_block.attn.proj.bias": "blocks.12.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.12.x_block.attn.proj.weight": "blocks.12.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.12.x_block.attn.qkv.bias": ['blocks.12.attn.a_to_q.bias', 'blocks.12.attn.a_to_k.bias', 'blocks.12.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.12.x_block.attn.qkv.weight": ['blocks.12.attn.a_to_q.weight', 'blocks.12.attn.a_to_k.weight', 'blocks.12.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.12.x_block.mlp.fc1.bias": "blocks.12.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.12.x_block.mlp.fc1.weight": "blocks.12.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.12.x_block.mlp.fc2.bias": "blocks.12.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.12.x_block.mlp.fc2.weight": "blocks.12.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.13.context_block.adaLN_modulation.1.bias": "blocks.13.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.13.context_block.adaLN_modulation.1.weight": "blocks.13.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.13.context_block.attn.proj.bias": "blocks.13.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.13.context_block.attn.proj.weight": "blocks.13.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.13.context_block.attn.qkv.bias": ['blocks.13.attn.b_to_q.bias', 'blocks.13.attn.b_to_k.bias', 'blocks.13.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.13.context_block.attn.qkv.weight": ['blocks.13.attn.b_to_q.weight', 'blocks.13.attn.b_to_k.weight', 'blocks.13.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.13.context_block.mlp.fc1.bias": "blocks.13.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.13.context_block.mlp.fc1.weight": "blocks.13.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.13.context_block.mlp.fc2.bias": "blocks.13.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.13.context_block.mlp.fc2.weight": "blocks.13.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.13.x_block.adaLN_modulation.1.bias": "blocks.13.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.13.x_block.adaLN_modulation.1.weight": "blocks.13.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.13.x_block.attn.proj.bias": "blocks.13.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.13.x_block.attn.proj.weight": "blocks.13.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.13.x_block.attn.qkv.bias": ['blocks.13.attn.a_to_q.bias', 'blocks.13.attn.a_to_k.bias', 'blocks.13.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.13.x_block.attn.qkv.weight": ['blocks.13.attn.a_to_q.weight', 'blocks.13.attn.a_to_k.weight', 'blocks.13.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.13.x_block.mlp.fc1.bias": "blocks.13.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.13.x_block.mlp.fc1.weight": "blocks.13.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.13.x_block.mlp.fc2.bias": "blocks.13.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.13.x_block.mlp.fc2.weight": "blocks.13.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.14.context_block.adaLN_modulation.1.bias": "blocks.14.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.14.context_block.adaLN_modulation.1.weight": "blocks.14.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.14.context_block.attn.proj.bias": "blocks.14.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.14.context_block.attn.proj.weight": "blocks.14.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.14.context_block.attn.qkv.bias": ['blocks.14.attn.b_to_q.bias', 'blocks.14.attn.b_to_k.bias', 'blocks.14.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.14.context_block.attn.qkv.weight": ['blocks.14.attn.b_to_q.weight', 'blocks.14.attn.b_to_k.weight', 'blocks.14.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.14.context_block.mlp.fc1.bias": "blocks.14.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.14.context_block.mlp.fc1.weight": "blocks.14.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.14.context_block.mlp.fc2.bias": "blocks.14.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.14.context_block.mlp.fc2.weight": "blocks.14.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.14.x_block.adaLN_modulation.1.bias": "blocks.14.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.14.x_block.adaLN_modulation.1.weight": "blocks.14.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.14.x_block.attn.proj.bias": "blocks.14.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.14.x_block.attn.proj.weight": "blocks.14.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.14.x_block.attn.qkv.bias": ['blocks.14.attn.a_to_q.bias', 'blocks.14.attn.a_to_k.bias', 'blocks.14.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.14.x_block.attn.qkv.weight": ['blocks.14.attn.a_to_q.weight', 'blocks.14.attn.a_to_k.weight', 'blocks.14.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.14.x_block.mlp.fc1.bias": "blocks.14.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.14.x_block.mlp.fc1.weight": "blocks.14.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.14.x_block.mlp.fc2.bias": "blocks.14.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.14.x_block.mlp.fc2.weight": "blocks.14.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.15.context_block.adaLN_modulation.1.bias": "blocks.15.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.15.context_block.adaLN_modulation.1.weight": "blocks.15.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.15.context_block.attn.proj.bias": "blocks.15.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.15.context_block.attn.proj.weight": "blocks.15.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.15.context_block.attn.qkv.bias": ['blocks.15.attn.b_to_q.bias', 'blocks.15.attn.b_to_k.bias', 'blocks.15.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.15.context_block.attn.qkv.weight": ['blocks.15.attn.b_to_q.weight', 'blocks.15.attn.b_to_k.weight', 'blocks.15.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.15.context_block.mlp.fc1.bias": "blocks.15.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.15.context_block.mlp.fc1.weight": "blocks.15.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.15.context_block.mlp.fc2.bias": "blocks.15.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.15.context_block.mlp.fc2.weight": "blocks.15.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.15.x_block.adaLN_modulation.1.bias": "blocks.15.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.15.x_block.adaLN_modulation.1.weight": "blocks.15.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.15.x_block.attn.proj.bias": "blocks.15.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.15.x_block.attn.proj.weight": "blocks.15.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.15.x_block.attn.qkv.bias": ['blocks.15.attn.a_to_q.bias', 'blocks.15.attn.a_to_k.bias', 'blocks.15.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.15.x_block.attn.qkv.weight": ['blocks.15.attn.a_to_q.weight', 'blocks.15.attn.a_to_k.weight', 'blocks.15.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.15.x_block.mlp.fc1.bias": "blocks.15.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.15.x_block.mlp.fc1.weight": "blocks.15.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.15.x_block.mlp.fc2.bias": "blocks.15.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.15.x_block.mlp.fc2.weight": "blocks.15.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.16.context_block.adaLN_modulation.1.bias": "blocks.16.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.16.context_block.adaLN_modulation.1.weight": "blocks.16.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.16.context_block.attn.proj.bias": "blocks.16.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.16.context_block.attn.proj.weight": "blocks.16.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.16.context_block.attn.qkv.bias": ['blocks.16.attn.b_to_q.bias', 'blocks.16.attn.b_to_k.bias', 'blocks.16.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.16.context_block.attn.qkv.weight": ['blocks.16.attn.b_to_q.weight', 'blocks.16.attn.b_to_k.weight', 'blocks.16.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.16.context_block.mlp.fc1.bias": "blocks.16.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.16.context_block.mlp.fc1.weight": "blocks.16.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.16.context_block.mlp.fc2.bias": "blocks.16.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.16.context_block.mlp.fc2.weight": "blocks.16.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.16.x_block.adaLN_modulation.1.bias": "blocks.16.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.16.x_block.adaLN_modulation.1.weight": "blocks.16.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.16.x_block.attn.proj.bias": "blocks.16.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.16.x_block.attn.proj.weight": "blocks.16.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.16.x_block.attn.qkv.bias": ['blocks.16.attn.a_to_q.bias', 'blocks.16.attn.a_to_k.bias', 'blocks.16.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.16.x_block.attn.qkv.weight": ['blocks.16.attn.a_to_q.weight', 'blocks.16.attn.a_to_k.weight', 'blocks.16.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.16.x_block.mlp.fc1.bias": "blocks.16.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.16.x_block.mlp.fc1.weight": "blocks.16.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.16.x_block.mlp.fc2.bias": "blocks.16.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.16.x_block.mlp.fc2.weight": "blocks.16.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.17.context_block.adaLN_modulation.1.bias": "blocks.17.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.17.context_block.adaLN_modulation.1.weight": "blocks.17.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.17.context_block.attn.proj.bias": "blocks.17.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.17.context_block.attn.proj.weight": "blocks.17.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.17.context_block.attn.qkv.bias": ['blocks.17.attn.b_to_q.bias', 'blocks.17.attn.b_to_k.bias', 'blocks.17.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.17.context_block.attn.qkv.weight": ['blocks.17.attn.b_to_q.weight', 'blocks.17.attn.b_to_k.weight', 'blocks.17.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.17.context_block.mlp.fc1.bias": "blocks.17.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.17.context_block.mlp.fc1.weight": "blocks.17.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.17.context_block.mlp.fc2.bias": "blocks.17.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.17.context_block.mlp.fc2.weight": "blocks.17.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.17.x_block.adaLN_modulation.1.bias": "blocks.17.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.17.x_block.adaLN_modulation.1.weight": "blocks.17.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.17.x_block.attn.proj.bias": "blocks.17.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.17.x_block.attn.proj.weight": "blocks.17.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.17.x_block.attn.qkv.bias": ['blocks.17.attn.a_to_q.bias', 'blocks.17.attn.a_to_k.bias', 'blocks.17.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.17.x_block.attn.qkv.weight": ['blocks.17.attn.a_to_q.weight', 'blocks.17.attn.a_to_k.weight', 'blocks.17.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.17.x_block.mlp.fc1.bias": "blocks.17.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.17.x_block.mlp.fc1.weight": "blocks.17.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.17.x_block.mlp.fc2.bias": "blocks.17.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.17.x_block.mlp.fc2.weight": "blocks.17.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.18.context_block.adaLN_modulation.1.bias": "blocks.18.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.18.context_block.adaLN_modulation.1.weight": "blocks.18.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.18.context_block.attn.proj.bias": "blocks.18.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.18.context_block.attn.proj.weight": "blocks.18.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.18.context_block.attn.qkv.bias": ['blocks.18.attn.b_to_q.bias', 'blocks.18.attn.b_to_k.bias', 'blocks.18.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.18.context_block.attn.qkv.weight": ['blocks.18.attn.b_to_q.weight', 'blocks.18.attn.b_to_k.weight', 'blocks.18.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.18.context_block.mlp.fc1.bias": "blocks.18.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.18.context_block.mlp.fc1.weight": "blocks.18.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.18.context_block.mlp.fc2.bias": "blocks.18.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.18.context_block.mlp.fc2.weight": "blocks.18.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.18.x_block.adaLN_modulation.1.bias": "blocks.18.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.18.x_block.adaLN_modulation.1.weight": "blocks.18.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.18.x_block.attn.proj.bias": "blocks.18.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.18.x_block.attn.proj.weight": "blocks.18.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.18.x_block.attn.qkv.bias": ['blocks.18.attn.a_to_q.bias', 'blocks.18.attn.a_to_k.bias', 'blocks.18.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.18.x_block.attn.qkv.weight": ['blocks.18.attn.a_to_q.weight', 'blocks.18.attn.a_to_k.weight', 'blocks.18.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.18.x_block.mlp.fc1.bias": "blocks.18.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.18.x_block.mlp.fc1.weight": "blocks.18.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.18.x_block.mlp.fc2.bias": "blocks.18.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.18.x_block.mlp.fc2.weight": "blocks.18.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.19.context_block.adaLN_modulation.1.bias": "blocks.19.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.19.context_block.adaLN_modulation.1.weight": "blocks.19.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.19.context_block.attn.proj.bias": "blocks.19.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.19.context_block.attn.proj.weight": "blocks.19.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.19.context_block.attn.qkv.bias": ['blocks.19.attn.b_to_q.bias', 'blocks.19.attn.b_to_k.bias', 'blocks.19.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.19.context_block.attn.qkv.weight": ['blocks.19.attn.b_to_q.weight', 'blocks.19.attn.b_to_k.weight', 'blocks.19.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.19.context_block.mlp.fc1.bias": "blocks.19.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.19.context_block.mlp.fc1.weight": "blocks.19.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.19.context_block.mlp.fc2.bias": "blocks.19.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.19.context_block.mlp.fc2.weight": "blocks.19.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.19.x_block.adaLN_modulation.1.bias": "blocks.19.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.19.x_block.adaLN_modulation.1.weight": "blocks.19.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.19.x_block.attn.proj.bias": "blocks.19.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.19.x_block.attn.proj.weight": "blocks.19.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.19.x_block.attn.qkv.bias": ['blocks.19.attn.a_to_q.bias', 'blocks.19.attn.a_to_k.bias', 'blocks.19.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.19.x_block.attn.qkv.weight": ['blocks.19.attn.a_to_q.weight', 'blocks.19.attn.a_to_k.weight', 'blocks.19.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.19.x_block.mlp.fc1.bias": "blocks.19.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.19.x_block.mlp.fc1.weight": "blocks.19.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.19.x_block.mlp.fc2.bias": "blocks.19.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.19.x_block.mlp.fc2.weight": "blocks.19.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.2.context_block.adaLN_modulation.1.bias": "blocks.2.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.2.context_block.adaLN_modulation.1.weight": "blocks.2.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.2.context_block.attn.proj.bias": "blocks.2.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.2.context_block.attn.proj.weight": "blocks.2.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.2.context_block.attn.qkv.bias": ['blocks.2.attn.b_to_q.bias', 'blocks.2.attn.b_to_k.bias', 'blocks.2.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.2.context_block.attn.qkv.weight": ['blocks.2.attn.b_to_q.weight', 'blocks.2.attn.b_to_k.weight', 'blocks.2.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.2.context_block.mlp.fc1.bias": "blocks.2.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.2.context_block.mlp.fc1.weight": "blocks.2.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.2.context_block.mlp.fc2.bias": "blocks.2.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.2.context_block.mlp.fc2.weight": "blocks.2.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.2.x_block.adaLN_modulation.1.bias": "blocks.2.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.2.x_block.adaLN_modulation.1.weight": "blocks.2.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.2.x_block.attn.proj.bias": "blocks.2.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.2.x_block.attn.proj.weight": "blocks.2.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.2.x_block.attn.qkv.bias": ['blocks.2.attn.a_to_q.bias', 'blocks.2.attn.a_to_k.bias', 'blocks.2.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.2.x_block.attn.qkv.weight": ['blocks.2.attn.a_to_q.weight', 'blocks.2.attn.a_to_k.weight', 'blocks.2.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.2.x_block.mlp.fc1.bias": "blocks.2.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.2.x_block.mlp.fc1.weight": "blocks.2.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.2.x_block.mlp.fc2.bias": "blocks.2.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.2.x_block.mlp.fc2.weight": "blocks.2.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.20.context_block.adaLN_modulation.1.bias": "blocks.20.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.20.context_block.adaLN_modulation.1.weight": "blocks.20.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.20.context_block.attn.proj.bias": "blocks.20.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.20.context_block.attn.proj.weight": "blocks.20.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.20.context_block.attn.qkv.bias": ['blocks.20.attn.b_to_q.bias', 'blocks.20.attn.b_to_k.bias', 'blocks.20.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.20.context_block.attn.qkv.weight": ['blocks.20.attn.b_to_q.weight', 'blocks.20.attn.b_to_k.weight', 'blocks.20.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.20.context_block.mlp.fc1.bias": "blocks.20.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.20.context_block.mlp.fc1.weight": "blocks.20.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.20.context_block.mlp.fc2.bias": "blocks.20.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.20.context_block.mlp.fc2.weight": "blocks.20.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.20.x_block.adaLN_modulation.1.bias": "blocks.20.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.20.x_block.adaLN_modulation.1.weight": "blocks.20.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.20.x_block.attn.proj.bias": "blocks.20.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.20.x_block.attn.proj.weight": "blocks.20.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.20.x_block.attn.qkv.bias": ['blocks.20.attn.a_to_q.bias', 'blocks.20.attn.a_to_k.bias', 'blocks.20.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.20.x_block.attn.qkv.weight": ['blocks.20.attn.a_to_q.weight', 'blocks.20.attn.a_to_k.weight', 'blocks.20.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.20.x_block.mlp.fc1.bias": "blocks.20.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.20.x_block.mlp.fc1.weight": "blocks.20.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.20.x_block.mlp.fc2.bias": "blocks.20.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.20.x_block.mlp.fc2.weight": "blocks.20.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.21.context_block.adaLN_modulation.1.bias": "blocks.21.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.21.context_block.adaLN_modulation.1.weight": "blocks.21.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.21.context_block.attn.proj.bias": "blocks.21.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.21.context_block.attn.proj.weight": "blocks.21.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.21.context_block.attn.qkv.bias": ['blocks.21.attn.b_to_q.bias', 'blocks.21.attn.b_to_k.bias', 'blocks.21.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.21.context_block.attn.qkv.weight": ['blocks.21.attn.b_to_q.weight', 'blocks.21.attn.b_to_k.weight', 'blocks.21.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.21.context_block.mlp.fc1.bias": "blocks.21.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.21.context_block.mlp.fc1.weight": "blocks.21.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.21.context_block.mlp.fc2.bias": "blocks.21.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.21.context_block.mlp.fc2.weight": "blocks.21.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.21.x_block.adaLN_modulation.1.bias": "blocks.21.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.21.x_block.adaLN_modulation.1.weight": "blocks.21.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.21.x_block.attn.proj.bias": "blocks.21.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.21.x_block.attn.proj.weight": "blocks.21.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.21.x_block.attn.qkv.bias": ['blocks.21.attn.a_to_q.bias', 'blocks.21.attn.a_to_k.bias', 'blocks.21.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.21.x_block.attn.qkv.weight": ['blocks.21.attn.a_to_q.weight', 'blocks.21.attn.a_to_k.weight', 'blocks.21.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.21.x_block.mlp.fc1.bias": "blocks.21.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.21.x_block.mlp.fc1.weight": "blocks.21.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.21.x_block.mlp.fc2.bias": "blocks.21.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.21.x_block.mlp.fc2.weight": "blocks.21.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.22.context_block.adaLN_modulation.1.bias": "blocks.22.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.22.context_block.adaLN_modulation.1.weight": "blocks.22.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.22.context_block.attn.proj.bias": "blocks.22.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.22.context_block.attn.proj.weight": "blocks.22.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.22.context_block.attn.qkv.bias": ['blocks.22.attn.b_to_q.bias', 'blocks.22.attn.b_to_k.bias', 'blocks.22.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.22.context_block.attn.qkv.weight": ['blocks.22.attn.b_to_q.weight', 'blocks.22.attn.b_to_k.weight', 'blocks.22.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.22.context_block.mlp.fc1.bias": "blocks.22.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.22.context_block.mlp.fc1.weight": "blocks.22.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.22.context_block.mlp.fc2.bias": "blocks.22.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.22.context_block.mlp.fc2.weight": "blocks.22.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.22.x_block.adaLN_modulation.1.bias": "blocks.22.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.22.x_block.adaLN_modulation.1.weight": "blocks.22.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.22.x_block.attn.proj.bias": "blocks.22.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.22.x_block.attn.proj.weight": "blocks.22.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.22.x_block.attn.qkv.bias": ['blocks.22.attn.a_to_q.bias', 'blocks.22.attn.a_to_k.bias', 'blocks.22.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.22.x_block.attn.qkv.weight": ['blocks.22.attn.a_to_q.weight', 'blocks.22.attn.a_to_k.weight', 'blocks.22.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.22.x_block.mlp.fc1.bias": "blocks.22.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.22.x_block.mlp.fc1.weight": "blocks.22.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.22.x_block.mlp.fc2.bias": "blocks.22.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.22.x_block.mlp.fc2.weight": "blocks.22.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.23.context_block.attn.qkv.bias": ['blocks.23.attn.b_to_q.bias', 'blocks.23.attn.b_to_k.bias', 'blocks.23.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.23.context_block.attn.qkv.weight": ['blocks.23.attn.b_to_q.weight', 'blocks.23.attn.b_to_k.weight', 'blocks.23.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.23.x_block.adaLN_modulation.1.bias": "blocks.23.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.23.x_block.adaLN_modulation.1.weight": "blocks.23.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.23.x_block.attn.proj.bias": "blocks.23.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.23.x_block.attn.proj.weight": "blocks.23.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.23.x_block.attn.qkv.bias": ['blocks.23.attn.a_to_q.bias', 'blocks.23.attn.a_to_k.bias', 'blocks.23.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.23.x_block.attn.qkv.weight": ['blocks.23.attn.a_to_q.weight', 'blocks.23.attn.a_to_k.weight', 'blocks.23.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.23.x_block.mlp.fc1.bias": "blocks.23.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.23.x_block.mlp.fc1.weight": "blocks.23.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.23.x_block.mlp.fc2.bias": "blocks.23.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.23.x_block.mlp.fc2.weight": "blocks.23.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.3.context_block.adaLN_modulation.1.bias": "blocks.3.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.3.context_block.adaLN_modulation.1.weight": "blocks.3.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.3.context_block.attn.proj.bias": "blocks.3.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.3.context_block.attn.proj.weight": "blocks.3.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.3.context_block.attn.qkv.bias": ['blocks.3.attn.b_to_q.bias', 'blocks.3.attn.b_to_k.bias', 'blocks.3.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.3.context_block.attn.qkv.weight": ['blocks.3.attn.b_to_q.weight', 'blocks.3.attn.b_to_k.weight', 'blocks.3.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.3.context_block.mlp.fc1.bias": "blocks.3.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.3.context_block.mlp.fc1.weight": "blocks.3.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.3.context_block.mlp.fc2.bias": "blocks.3.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.3.context_block.mlp.fc2.weight": "blocks.3.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.3.x_block.adaLN_modulation.1.bias": "blocks.3.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.3.x_block.adaLN_modulation.1.weight": "blocks.3.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.3.x_block.attn.proj.bias": "blocks.3.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.3.x_block.attn.proj.weight": "blocks.3.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.3.x_block.attn.qkv.bias": ['blocks.3.attn.a_to_q.bias', 'blocks.3.attn.a_to_k.bias', 'blocks.3.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.3.x_block.attn.qkv.weight": ['blocks.3.attn.a_to_q.weight', 'blocks.3.attn.a_to_k.weight', 'blocks.3.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.3.x_block.mlp.fc1.bias": "blocks.3.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.3.x_block.mlp.fc1.weight": "blocks.3.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.3.x_block.mlp.fc2.bias": "blocks.3.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.3.x_block.mlp.fc2.weight": "blocks.3.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.4.context_block.adaLN_modulation.1.bias": "blocks.4.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.4.context_block.adaLN_modulation.1.weight": "blocks.4.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.4.context_block.attn.proj.bias": "blocks.4.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.4.context_block.attn.proj.weight": "blocks.4.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.4.context_block.attn.qkv.bias": ['blocks.4.attn.b_to_q.bias', 'blocks.4.attn.b_to_k.bias', 'blocks.4.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.4.context_block.attn.qkv.weight": ['blocks.4.attn.b_to_q.weight', 'blocks.4.attn.b_to_k.weight', 'blocks.4.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.4.context_block.mlp.fc1.bias": "blocks.4.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.4.context_block.mlp.fc1.weight": "blocks.4.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.4.context_block.mlp.fc2.bias": "blocks.4.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.4.context_block.mlp.fc2.weight": "blocks.4.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.4.x_block.adaLN_modulation.1.bias": "blocks.4.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.4.x_block.adaLN_modulation.1.weight": "blocks.4.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.4.x_block.attn.proj.bias": "blocks.4.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.4.x_block.attn.proj.weight": "blocks.4.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.4.x_block.attn.qkv.bias": ['blocks.4.attn.a_to_q.bias', 'blocks.4.attn.a_to_k.bias', 'blocks.4.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.4.x_block.attn.qkv.weight": ['blocks.4.attn.a_to_q.weight', 'blocks.4.attn.a_to_k.weight', 'blocks.4.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.4.x_block.mlp.fc1.bias": "blocks.4.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.4.x_block.mlp.fc1.weight": "blocks.4.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.4.x_block.mlp.fc2.bias": "blocks.4.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.4.x_block.mlp.fc2.weight": "blocks.4.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.5.context_block.adaLN_modulation.1.bias": "blocks.5.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.5.context_block.adaLN_modulation.1.weight": "blocks.5.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.5.context_block.attn.proj.bias": "blocks.5.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.5.context_block.attn.proj.weight": "blocks.5.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.5.context_block.attn.qkv.bias": ['blocks.5.attn.b_to_q.bias', 'blocks.5.attn.b_to_k.bias', 'blocks.5.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.5.context_block.attn.qkv.weight": ['blocks.5.attn.b_to_q.weight', 'blocks.5.attn.b_to_k.weight', 'blocks.5.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.5.context_block.mlp.fc1.bias": "blocks.5.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.5.context_block.mlp.fc1.weight": "blocks.5.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.5.context_block.mlp.fc2.bias": "blocks.5.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.5.context_block.mlp.fc2.weight": "blocks.5.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.5.x_block.adaLN_modulation.1.bias": "blocks.5.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.5.x_block.adaLN_modulation.1.weight": "blocks.5.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.5.x_block.attn.proj.bias": "blocks.5.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.5.x_block.attn.proj.weight": "blocks.5.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.5.x_block.attn.qkv.bias": ['blocks.5.attn.a_to_q.bias', 'blocks.5.attn.a_to_k.bias', 'blocks.5.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.5.x_block.attn.qkv.weight": ['blocks.5.attn.a_to_q.weight', 'blocks.5.attn.a_to_k.weight', 'blocks.5.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.5.x_block.mlp.fc1.bias": "blocks.5.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.5.x_block.mlp.fc1.weight": "blocks.5.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.5.x_block.mlp.fc2.bias": "blocks.5.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.5.x_block.mlp.fc2.weight": "blocks.5.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.6.context_block.adaLN_modulation.1.bias": "blocks.6.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.6.context_block.adaLN_modulation.1.weight": "blocks.6.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.6.context_block.attn.proj.bias": "blocks.6.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.6.context_block.attn.proj.weight": "blocks.6.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.6.context_block.attn.qkv.bias": ['blocks.6.attn.b_to_q.bias', 'blocks.6.attn.b_to_k.bias', 'blocks.6.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.6.context_block.attn.qkv.weight": ['blocks.6.attn.b_to_q.weight', 'blocks.6.attn.b_to_k.weight', 'blocks.6.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.6.context_block.mlp.fc1.bias": "blocks.6.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.6.context_block.mlp.fc1.weight": "blocks.6.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.6.context_block.mlp.fc2.bias": "blocks.6.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.6.context_block.mlp.fc2.weight": "blocks.6.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.6.x_block.adaLN_modulation.1.bias": "blocks.6.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.6.x_block.adaLN_modulation.1.weight": "blocks.6.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.6.x_block.attn.proj.bias": "blocks.6.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.6.x_block.attn.proj.weight": "blocks.6.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.6.x_block.attn.qkv.bias": ['blocks.6.attn.a_to_q.bias', 'blocks.6.attn.a_to_k.bias', 'blocks.6.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.6.x_block.attn.qkv.weight": ['blocks.6.attn.a_to_q.weight', 'blocks.6.attn.a_to_k.weight', 'blocks.6.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.6.x_block.mlp.fc1.bias": "blocks.6.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.6.x_block.mlp.fc1.weight": "blocks.6.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.6.x_block.mlp.fc2.bias": "blocks.6.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.6.x_block.mlp.fc2.weight": "blocks.6.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.7.context_block.adaLN_modulation.1.bias": "blocks.7.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.7.context_block.adaLN_modulation.1.weight": "blocks.7.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.7.context_block.attn.proj.bias": "blocks.7.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.7.context_block.attn.proj.weight": "blocks.7.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.7.context_block.attn.qkv.bias": ['blocks.7.attn.b_to_q.bias', 'blocks.7.attn.b_to_k.bias', 'blocks.7.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.7.context_block.attn.qkv.weight": ['blocks.7.attn.b_to_q.weight', 'blocks.7.attn.b_to_k.weight', 'blocks.7.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.7.context_block.mlp.fc1.bias": "blocks.7.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.7.context_block.mlp.fc1.weight": "blocks.7.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.7.context_block.mlp.fc2.bias": "blocks.7.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.7.context_block.mlp.fc2.weight": "blocks.7.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.7.x_block.adaLN_modulation.1.bias": "blocks.7.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.7.x_block.adaLN_modulation.1.weight": "blocks.7.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.7.x_block.attn.proj.bias": "blocks.7.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.7.x_block.attn.proj.weight": "blocks.7.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.7.x_block.attn.qkv.bias": ['blocks.7.attn.a_to_q.bias', 'blocks.7.attn.a_to_k.bias', 'blocks.7.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.7.x_block.attn.qkv.weight": ['blocks.7.attn.a_to_q.weight', 'blocks.7.attn.a_to_k.weight', 'blocks.7.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.7.x_block.mlp.fc1.bias": "blocks.7.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.7.x_block.mlp.fc1.weight": "blocks.7.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.7.x_block.mlp.fc2.bias": "blocks.7.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.7.x_block.mlp.fc2.weight": "blocks.7.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.8.context_block.adaLN_modulation.1.bias": "blocks.8.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.8.context_block.adaLN_modulation.1.weight": "blocks.8.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.8.context_block.attn.proj.bias": "blocks.8.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.8.context_block.attn.proj.weight": "blocks.8.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.8.context_block.attn.qkv.bias": ['blocks.8.attn.b_to_q.bias', 'blocks.8.attn.b_to_k.bias', 'blocks.8.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.8.context_block.attn.qkv.weight": ['blocks.8.attn.b_to_q.weight', 'blocks.8.attn.b_to_k.weight', 'blocks.8.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.8.context_block.mlp.fc1.bias": "blocks.8.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.8.context_block.mlp.fc1.weight": "blocks.8.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.8.context_block.mlp.fc2.bias": "blocks.8.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.8.context_block.mlp.fc2.weight": "blocks.8.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.8.x_block.adaLN_modulation.1.bias": "blocks.8.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.8.x_block.adaLN_modulation.1.weight": "blocks.8.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.8.x_block.attn.proj.bias": "blocks.8.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.8.x_block.attn.proj.weight": "blocks.8.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.8.x_block.attn.qkv.bias": ['blocks.8.attn.a_to_q.bias', 'blocks.8.attn.a_to_k.bias', 'blocks.8.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.8.x_block.attn.qkv.weight": ['blocks.8.attn.a_to_q.weight', 'blocks.8.attn.a_to_k.weight', 'blocks.8.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.8.x_block.mlp.fc1.bias": "blocks.8.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.8.x_block.mlp.fc1.weight": "blocks.8.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.8.x_block.mlp.fc2.bias": "blocks.8.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.8.x_block.mlp.fc2.weight": "blocks.8.ff_a.2.weight",
            "model.diffusion_model.joint_blocks.9.context_block.adaLN_modulation.1.bias": "blocks.9.norm1_b.linear.bias",
            "model.diffusion_model.joint_blocks.9.context_block.adaLN_modulation.1.weight": "blocks.9.norm1_b.linear.weight",
            "model.diffusion_model.joint_blocks.9.context_block.attn.proj.bias": "blocks.9.attn.b_to_out.bias",
            "model.diffusion_model.joint_blocks.9.context_block.attn.proj.weight": "blocks.9.attn.b_to_out.weight",
            "model.diffusion_model.joint_blocks.9.context_block.attn.qkv.bias": ['blocks.9.attn.b_to_q.bias', 'blocks.9.attn.b_to_k.bias', 'blocks.9.attn.b_to_v.bias'],
            "model.diffusion_model.joint_blocks.9.context_block.attn.qkv.weight": ['blocks.9.attn.b_to_q.weight', 'blocks.9.attn.b_to_k.weight', 'blocks.9.attn.b_to_v.weight'],
            "model.diffusion_model.joint_blocks.9.context_block.mlp.fc1.bias": "blocks.9.ff_b.0.bias",
            "model.diffusion_model.joint_blocks.9.context_block.mlp.fc1.weight": "blocks.9.ff_b.0.weight",
            "model.diffusion_model.joint_blocks.9.context_block.mlp.fc2.bias": "blocks.9.ff_b.2.bias",
            "model.diffusion_model.joint_blocks.9.context_block.mlp.fc2.weight": "blocks.9.ff_b.2.weight",
            "model.diffusion_model.joint_blocks.9.x_block.adaLN_modulation.1.bias": "blocks.9.norm1_a.linear.bias",
            "model.diffusion_model.joint_blocks.9.x_block.adaLN_modulation.1.weight": "blocks.9.norm1_a.linear.weight",
            "model.diffusion_model.joint_blocks.9.x_block.attn.proj.bias": "blocks.9.attn.a_to_out.bias",
            "model.diffusion_model.joint_blocks.9.x_block.attn.proj.weight": "blocks.9.attn.a_to_out.weight",
            "model.diffusion_model.joint_blocks.9.x_block.attn.qkv.bias": ['blocks.9.attn.a_to_q.bias', 'blocks.9.attn.a_to_k.bias', 'blocks.9.attn.a_to_v.bias'],
            "model.diffusion_model.joint_blocks.9.x_block.attn.qkv.weight": ['blocks.9.attn.a_to_q.weight', 'blocks.9.attn.a_to_k.weight', 'blocks.9.attn.a_to_v.weight'],
            "model.diffusion_model.joint_blocks.9.x_block.mlp.fc1.bias": "blocks.9.ff_a.0.bias",
            "model.diffusion_model.joint_blocks.9.x_block.mlp.fc1.weight": "blocks.9.ff_a.0.weight",
            "model.diffusion_model.joint_blocks.9.x_block.mlp.fc2.bias": "blocks.9.ff_a.2.bias",
            "model.diffusion_model.joint_blocks.9.x_block.mlp.fc2.weight": "blocks.9.ff_a.2.weight",
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
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name.startswith("model.diffusion_model.joint_blocks.23.context_block.adaLN_modulation.1."):
                    param = torch.concat([param[1536:], param[:1536]], axis=0)
                elif name.startswith("model.diffusion_model.final_layer.adaLN_modulation.1."):
                    param = torch.concat([param[1536:], param[:1536]], axis=0)
                elif name == "model.diffusion_model.pos_embed":
                    param = param.reshape((1, 192, 192, 1536))
                if isinstance(rename_dict[name], str):
                    state_dict_[rename_dict[name]] = param
                else:
                    name_ = rename_dict[name][0].replace(".a_to_q.", ".a_to_qkv.").replace(".b_to_q.", ".b_to_qkv.")
                    state_dict_[name_] = param
        return state_dict_
