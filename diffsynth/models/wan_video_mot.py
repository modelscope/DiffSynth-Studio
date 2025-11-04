import torch
from .wan_video_dit import DiTBlock, SelfAttention, rope_apply, flash_attention, modulate, MLP
from .utils import hash_state_dict_keys
import einops
import torch.nn as nn


class MotSelfAttention(SelfAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, num_heads, eps)
    def forward(self, x, freqs, is_before_attn=False):
        if is_before_attn:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            return q, k, v
        else:
            return self.o(x)


class MotWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id

        self.self_attn = MotSelfAttention(dim, num_heads, eps)


    def forward(self, wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot):

        # 1. prepare scale parameter
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            wan_block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        
        scale_params_mot_ref = self.modulation + t_mod_mot.float()
        scale_params_mot_ref = einops.rearrange(scale_params_mot_ref, '(b n) t c -> b n t c', n=1)
        shift_msa_mot_ref, scale_msa_mot_ref, gate_msa_mot_ref, c_shift_msa_mot_ref, c_scale_msa_mot_ref, c_gate_msa_mot_ref = scale_params_mot_ref.chunk(6, dim=2)

        # 2. Self-attention
        input_x = modulate(wan_block.norm1(x), shift_msa, scale_msa)
        # original block self-attn
        attn1 = wan_block.self_attn
        q = attn1.norm_q(attn1.q(input_x))
        k = attn1.norm_k(attn1.k(input_x))
        v = attn1.v(input_x)
        q = rope_apply(q, freqs, attn1.num_heads)
        k = rope_apply(k, freqs, attn1.num_heads)

        # mot block self-attn
        norm_x_mot = einops.rearrange(self.norm1(x_mot.float()), 'b (n t) c -> b n t c', n=1)
        norm_x_mot = modulate(norm_x_mot, shift_msa_mot_ref, scale_msa_mot_ref).type_as(x_mot)
        norm_x_mot = einops.rearrange(norm_x_mot, 'b n t c -> b (n t) c', n=1)
        q_mot,k_mot,v_mot = self.self_attn(norm_x_mot, freqs_mot, is_before_attn=True)

        tmp_hidden_states = flash_attention(
            torch.cat([q, q_mot], dim=-2),
            torch.cat([k, k_mot], dim=-2),
            torch.cat([v, v_mot], dim=-2),
            num_heads=attn1.num_heads)

        attn_output, attn_output_mot = torch.split(tmp_hidden_states, [q.shape[-2], q_mot.shape[-2]], dim=-2)
        
        attn_output = attn1.o(attn_output)
        x = wan_block.gate(x, gate_msa, attn_output)

        attn_output_mot = self.self_attn(x=attn_output_mot,freqs=freqs_mot, is_before_attn=False)
        # gate
        attn_output_mot = einops.rearrange(attn_output_mot, 'b (n t) c -> b n t c', n=1)
        attn_output_mot = attn_output_mot * gate_msa_mot_ref
        attn_output_mot = einops.rearrange(attn_output_mot, 'b n t c -> b (n t) c', n=1)
        x_mot = (x_mot.float() + attn_output_mot).type_as(x_mot)

        # 3. cross-attention and feed-forward
        x = x + wan_block.cross_attn(wan_block.norm3(x), context)
        input_x = modulate(wan_block.norm2(x), shift_mlp, scale_mlp)
        x = wan_block.gate(x, gate_mlp, wan_block.ffn(input_x))

        x_mot = x_mot + self.cross_attn(self.norm3(x_mot),context_mot)
        # modulate
        norm_x_mot_ref = einops.rearrange(self.norm2(x_mot.float()), 'b (n t) c -> b n t c', n=1)
        norm_x_mot_ref = (norm_x_mot_ref * (1 + c_scale_msa_mot_ref) + c_shift_msa_mot_ref).type_as(x_mot)
        norm_x_mot_ref = einops.rearrange(norm_x_mot_ref, 'b n t c -> b (n t) c', n=1)
        input_x_mot = self.ffn(norm_x_mot_ref)
        # gate
        input_x_mot = einops.rearrange(input_x_mot, 'b (n t) c -> b n t c', n=1)
        input_x_mot = input_x_mot.float() * c_gate_msa_mot_ref
        input_x_mot = einops.rearrange(input_x_mot, 'b n t c -> b (n t) c', n=1)
        x_mot = (x_mot.float() + input_x_mot).type_as(x_mot)

        return x, x_mot


class MotWanModel(torch.nn.Module):
    def __init__(
        self,
        mot_layers=(0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
        patch_size=(1, 2, 2),
        has_image_input=True,
        has_image_pos_emb=False,
        dim=5120,
        num_heads=40,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        in_dim=36,
        eps=1e-6,
    ):
        super().__init__()
        self.mot_layers = mot_layers
        self.freq_dim = freq_dim
        self.dim = dim

        self.mot_layers_mapping = {i: n for n, i in enumerate(self.mot_layers)}
        self.head_dim = dim // num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)

        # mot blocks
        self.blocks = torch.nn.ModuleList([
            MotWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.mot_layers
        ])
    

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        return x

    def compute_freqs_mot(self, f, h, w, end: int = 1024, theta: float = 10000.0):
        def precompute_freqs_cis(dim: int, start: int = 0, end: int = 1024, theta: float = 10000.0):
            # 1d rope precompute
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                        [: (dim // 2)].double() / dim))
            freqs = torch.outer(torch.arange(start, end, device=freqs.device), freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        f_freqs_cis = precompute_freqs_cis(self.head_dim - 2 * (self.head_dim // 3), -f, end, theta)
        h_freqs_cis = precompute_freqs_cis(self.head_dim // 3, 0, end, theta)
        w_freqs_cis = precompute_freqs_cis(self.head_dim // 3, 0, end, theta)

        freqs = torch.cat([
            f_freqs_cis[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs_cis[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_freqs_cis[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1)
        return freqs

    def forward(self, wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot, block_id):
        block = self.blocks[self.mot_layers_mapping[block_id]]
        x, x_mot = block(wan_block, x, context, t_mod, freqs, x_mot, context_mot, t_mod_mot, freqs_mot)
        return x, x_mot
    
    @staticmethod
    def state_dict_converter():
        return MotWanModelDictConverter()
    
    
class MotWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_diffusers(self, state_dict):
        
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.attn2.add_k_proj.bias":"blocks.0.cross_attn.k_img.bias",
            "blocks.0.attn2.add_k_proj.weight":"blocks.0.cross_attn.k_img.weight",
            "blocks.0.attn2.add_v_proj.bias":"blocks.0.cross_attn.v_img.bias",
            "blocks.0.attn2.add_v_proj.weight":"blocks.0.cross_attn.v_img.weight",
            "blocks.0.attn2.norm_added_k.weight":"blocks.0.cross_attn.norm_k_img.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "condition_embedder.image_embedder.ff.net.0.proj.bias":"img_emb.proj.1.bias",
            "condition_embedder.image_embedder.ff.net.0.proj.weight":"img_emb.proj.1.weight",
            "condition_embedder.image_embedder.ff.net.2.bias":"img_emb.proj.3.bias",
            "condition_embedder.image_embedder.ff.net.2.weight":"img_emb.proj.3.weight",
            "condition_embedder.image_embedder.norm1.bias":"img_emb.proj.0.bias",
            "condition_embedder.image_embedder.norm1.weight":"img_emb.proj.0.weight",
            "condition_embedder.image_embedder.norm2.bias":"img_emb.proj.4.bias",
            "condition_embedder.image_embedder.norm2.weight":"img_emb.proj.4.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict = {name: param for name, param in state_dict.items() if '_mot_ref' in name}
        if hash_state_dict_keys(state_dict) == '19debbdb7f4d5ba93b4ddb1cbe5788c7':
            mot_layers = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36)
        else:
            mot_layers = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36)
        mot_layers_mapping = {i:n for n, i in enumerate(mot_layers)}

        state_dict_ = {}

        for name, param in state_dict.items():
            name = name.replace("_mot_ref", "")
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                if name.split(".")[1].isdigit():
                    block_id = int(name.split(".")[1])
                    name = name.replace(str(block_id), str(mot_layers_mapping[block_id]))
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param

        if hash_state_dict_keys(state_dict_) == '6507c8213a3c476df5958b01dcf302d0': # vap 14B
            config = {
                "mot_layers":(0, 4, 8, 12, 16, 20, 24, 28, 32, 36),
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "num_heads": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict_, config


    