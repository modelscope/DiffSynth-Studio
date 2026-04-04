import torch
from .wan_video_dit import DiTBlock, SelfAttention, rope_apply, flash_attention, modulate, MLP
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
