from .attention import Attention
from .tiler import TileWorker
from einops import repeat, rearrange
import math
import torch


class HunyuanDiTRotaryEmbedding(torch.nn.Module):

    def __init__(self, q_norm_shape=88, k_norm_shape=88, rotary_emb_on_k=True):
        super().__init__()
        self.q_norm = torch.nn.LayerNorm((q_norm_shape,), elementwise_affine=True, eps=1e-06)
        self.k_norm = torch.nn.LayerNorm((k_norm_shape,), elementwise_affine=True, eps=1e-06)
        self.rotary_emb_on_k = rotary_emb_on_k
        self.k_cache, self.v_cache = [], []

    def reshape_for_broadcast(self, freqs_cis, x):
        ndim = x.ndim
        shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)

    def rotate_half(self, x):
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    def apply_rotary_emb(self, xq, xk, freqs_cis):
        xk_out = None
        cos, sin = self.reshape_for_broadcast(freqs_cis, xq)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos + self.rotate_half(xq.float()) * sin).type_as(xq)
        if xk is not None:
            xk_out = (xk.float() * cos + self.rotate_half(xk.float()) * sin).type_as(xk)
        return xq_out, xk_out

    def forward(self, q, k, v, freqs_cis_img, to_cache=False):
        # norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        if self.rotary_emb_on_k:
            q, k = self.apply_rotary_emb(q, k, freqs_cis_img)
        else:
            q, _ = self.apply_rotary_emb(q, None, freqs_cis_img)
        
        if to_cache:
            self.k_cache.append(k)
            self.v_cache.append(v)
        elif len(self.k_cache) > 0 and len(self.v_cache) > 0:
            k = torch.concat([k] + self.k_cache, dim=2)
            v = torch.concat([v] + self.v_cache, dim=2)
            self.k_cache, self.v_cache = [], []
        return q, k, v


class FP32_Layernorm(torch.nn.LayerNorm):
    def forward(self, inputs):
        origin_dtype = inputs.dtype
        return torch.nn.functional.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).to(origin_dtype)


class FP32_SiLU(torch.nn.SiLU):
    def forward(self, inputs):
        origin_dtype = inputs.dtype
        return torch.nn.functional.silu(inputs.float(), inplace=False).to(origin_dtype)
    

class HunyuanDiTFinalLayer(torch.nn.Module):
    def __init__(self, final_hidden_size=1408, condition_dim=1408, patch_size=2, out_channels=8):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = torch.nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = torch.nn.Sequential(
            FP32_SiLU(),
            torch.nn.Linear(condition_dim, 2 * final_hidden_size, bias=True)
        )

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, hidden_states, condition_emb):
        shift, scale = self.adaLN_modulation(condition_emb).chunk(2, dim=1)
        hidden_states = self.modulate(self.norm_final(hidden_states), shift, scale)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class HunyuanDiTBlock(torch.nn.Module):

    def __init__(
        self,
        hidden_dim=1408,
        condition_dim=1408,
        num_heads=16,
        mlp_ratio=4.3637,
        text_dim=1024,
        skip_connection=False
    ):
        super().__init__()
        self.norm1 = FP32_Layernorm((hidden_dim,), eps=1e-6, elementwise_affine=True)
        self.rota1 = HunyuanDiTRotaryEmbedding(hidden_dim//num_heads, hidden_dim//num_heads)
        self.attn1 = Attention(hidden_dim, num_heads, hidden_dim//num_heads, bias_q=True, bias_kv=True, bias_out=True)
        self.norm2 = FP32_Layernorm((hidden_dim,), eps=1e-6, elementwise_affine=True)
        self.rota2 = HunyuanDiTRotaryEmbedding(hidden_dim//num_heads, hidden_dim//num_heads, rotary_emb_on_k=False)
        self.attn2 = Attention(hidden_dim, num_heads, hidden_dim//num_heads, kv_dim=text_dim, bias_q=True, bias_kv=True, bias_out=True)
        self.norm3 = FP32_Layernorm((hidden_dim,), eps=1e-6, elementwise_affine=True)
        self.modulation = torch.nn.Sequential(FP32_SiLU(), torch.nn.Linear(condition_dim, hidden_dim, bias=True))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, int(hidden_dim*mlp_ratio), bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(int(hidden_dim*mlp_ratio), hidden_dim, bias=True)
        )
        if skip_connection:
            self.skip_norm = FP32_Layernorm((hidden_dim * 2,), eps=1e-6, elementwise_affine=True)
            self.skip_linear = torch.nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        else:
            self.skip_norm, self.skip_linear = None, None

    def forward(self, hidden_states, condition_emb, text_emb, freq_cis_img, residual=None, to_cache=False):
        # Long Skip Connection
        if self.skip_norm is not None and self.skip_linear is not None:
            hidden_states = torch.cat([hidden_states, residual], dim=-1)
            hidden_states = self.skip_norm(hidden_states)
            hidden_states = self.skip_linear(hidden_states)

        # Self-Attention
        shift_msa = self.modulation(condition_emb).unsqueeze(dim=1)
        attn_input = self.norm1(hidden_states) + shift_msa
        hidden_states = hidden_states + self.attn1(attn_input, qkv_preprocessor=lambda q, k, v: self.rota1(q, k, v, freq_cis_img, to_cache=to_cache))

        # Cross-Attention
        attn_input = self.norm3(hidden_states)
        hidden_states = hidden_states + self.attn2(attn_input, text_emb, qkv_preprocessor=lambda q, k, v: self.rota2(q, k, v, freq_cis_img))

        # FFN Layer
        mlp_input = self.norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)
        return hidden_states
    

class AttentionPool(torch.nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim = None):
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = torch.nn.functional.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        patch_size=(2, 2),
        in_chans=4,
        embed_dim=1408,
        bias=True,
    ):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x
    

def timestep_embedding(t, dim, max_period=10000, repeat_only=False):
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)   # size: [dim/2], 一个指数衰减的曲线
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(t, "b -> b d", d=dim)
    return embedding
    

class TimestepEmbedder(torch.nn.Module):
    def __init__(self, hidden_size=1408, frequency_embedding_size=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class HunyuanDiT(torch.nn.Module):
    def __init__(self, num_layers_down=21, num_layers_up=19, in_channels=4, out_channels=8, hidden_dim=1408, text_dim=1024, t5_dim=2048, text_length=77, t5_length=256):
        super().__init__()

        # Embedders
        self.text_emb_padding = torch.nn.Parameter(torch.randn(text_length + t5_length, text_dim, dtype=torch.float32))
        self.t5_embedder = torch.nn.Sequential(
            torch.nn.Linear(t5_dim, t5_dim * 4, bias=True),
            FP32_SiLU(),
            torch.nn.Linear(t5_dim * 4, text_dim, bias=True),
        )
        self.t5_pooler = AttentionPool(t5_length, t5_dim, num_heads=8, output_dim=1024)
        self.style_embedder = torch.nn.Parameter(torch.randn(hidden_dim))
        self.patch_embedder = PatchEmbed(in_chans=in_channels)
        self.timestep_embedder = TimestepEmbedder()
        self.extra_embedder = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 + 1024 + hidden_dim, hidden_dim * 4),
            FP32_SiLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Transformer blocks
        self.num_layers_down = num_layers_down
        self.num_layers_up = num_layers_up
        self.blocks = torch.nn.ModuleList(
            [HunyuanDiTBlock(skip_connection=False) for _ in range(num_layers_down)] + \
            [HunyuanDiTBlock(skip_connection=True) for _ in range(num_layers_up)]
        )

        # Output layers
        self.final_layer = HunyuanDiTFinalLayer()
        self.out_channels = out_channels

    def prepare_text_emb(self, text_emb, text_emb_t5, text_emb_mask, text_emb_mask_t5):
        text_emb_mask = text_emb_mask.bool()
        text_emb_mask_t5 = text_emb_mask_t5.bool()
        text_emb_t5 = self.t5_embedder(text_emb_t5)
        text_emb = torch.cat([text_emb, text_emb_t5], dim=1)
        text_emb_mask = torch.cat([text_emb_mask, text_emb_mask_t5], dim=-1)
        text_emb = torch.where(text_emb_mask.unsqueeze(2), text_emb, self.text_emb_padding.to(text_emb))
        return text_emb
    
    def prepare_extra_emb(self, text_emb_t5, timestep, size_emb, dtype, batch_size):
        # Text embedding
        pooled_text_emb_t5 = self.t5_pooler(text_emb_t5)

        # Timestep embedding
        timestep_emb = self.timestep_embedder(timestep)

        # Size embedding
        size_emb = timestep_embedding(size_emb.view(-1), 256).to(dtype)
        size_emb = size_emb.view(-1, 6 * 256)

        # Style embedding
        style_emb = repeat(self.style_embedder, "D -> B D", B=batch_size)

        # Concatenate all extra vectors
        extra_emb = torch.cat([pooled_text_emb_t5, size_emb, style_emb], dim=1)
        condition_emb = timestep_emb + self.extra_embedder(extra_emb)

        return condition_emb

    def unpatchify(self, x, h, w):
        return rearrange(x, "B (H W) (P Q C) -> B C (H P) (W Q)", H=h, W=w, P=2, Q=2)
    
    def build_mask(self, data, is_bound):
        _, _, H, W = data.shape
        h = repeat(torch.arange(H), "H -> H W", H=H, W=W)
        w = repeat(torch.arange(W), "W -> H W", H=H, W=W)
        border_width = (H + W) // 4
        pad = torch.ones_like(h) * border_width
        mask = torch.stack([
            pad if is_bound[0] else h + 1,
            pad if is_bound[1] else H - h,
            pad if is_bound[2] else w + 1,
            pad if is_bound[3] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=data.dtype, device=data.device)
        mask = rearrange(mask, "H W -> 1 H W")
        return mask
    
    def tiled_block_forward(self, block, hidden_states, condition_emb, text_emb, freq_cis_img, residual, torch_dtype, data_device, computation_device, tile_size, tile_stride):
        B, C, H, W = hidden_states.shape

        weight = torch.zeros((1, 1, H, W), dtype=torch_dtype, device=data_device)
        values = torch.zeros((B, C, H, W), dtype=torch_dtype, device=data_device)

        # Split tasks
        tasks = []
        for h in range(0, H, tile_stride):
            for w in range(0, W, tile_stride):
                if (h-tile_stride >= 0 and h-tile_stride+tile_size >= H) or (w-tile_stride >= 0 and w-tile_stride+tile_size >= W):
                    continue
                h_, w_ = h + tile_size, w + tile_size
                if h_ > H: h, h_ = H - tile_size, H
                if w_ > W: w, w_ = W - tile_size, W
                tasks.append((h, h_, w, w_))

        # Run
        for hl, hr, wl, wr in tasks:
            hidden_states_batch = hidden_states[:, :, hl:hr, wl:wr].to(computation_device)
            hidden_states_batch = rearrange(hidden_states_batch, "B C H W -> B (H W) C")
            if residual is not None:
                residual_batch = residual[:, :, hl:hr, wl:wr].to(computation_device)
                residual_batch = rearrange(residual_batch, "B C H W -> B (H W) C")
            else:
                residual_batch = None

            # Forward
            hidden_states_batch = block(hidden_states_batch, condition_emb, text_emb, freq_cis_img, residual_batch).to(data_device)
            hidden_states_batch = rearrange(hidden_states_batch, "B (H W) C -> B C H W", H=hr-hl)

            mask = self.build_mask(hidden_states_batch, is_bound=(hl==0, hr>=H, wl==0, wr>=W))
            values[:, :, hl:hr, wl:wr] += hidden_states_batch * mask
            weight[:, :, hl:hr, wl:wr] += mask
        values /= weight
        return values

    def forward(
        self, hidden_states, text_emb, text_emb_t5, text_emb_mask, text_emb_mask_t5, timestep, size_emb, freq_cis_img,
        tiled=False, tile_size=64, tile_stride=32,
        to_cache=False,
        use_gradient_checkpointing=False,
    ):
        # Embeddings
        text_emb = self.prepare_text_emb(text_emb, text_emb_t5, text_emb_mask, text_emb_mask_t5)
        condition_emb = self.prepare_extra_emb(text_emb_t5, timestep, size_emb, hidden_states.dtype, hidden_states.shape[0])
        
        # Input
        height, width = hidden_states.shape[-2], hidden_states.shape[-1]
        hidden_states = self.patch_embedder(hidden_states)

        # Blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        if tiled:
            hidden_states = rearrange(hidden_states, "B (H W) C -> B C H W", H=height//2)
            residuals = []
            for block_id, block in enumerate(self.blocks):
                residual = residuals.pop() if block_id >= self.num_layers_down else None
                hidden_states = self.tiled_block_forward(
                    block, hidden_states, condition_emb, text_emb, freq_cis_img, residual,
                    torch_dtype=hidden_states.dtype, data_device=hidden_states.device, computation_device=hidden_states.device,
                    tile_size=tile_size, tile_stride=tile_stride
                )
                if block_id < self.num_layers_down - 2:
                    residuals.append(hidden_states)
            hidden_states = rearrange(hidden_states, "B C H W -> B (H W) C")
        else:
            residuals = []
            for block_id, block in enumerate(self.blocks):
                residual = residuals.pop() if block_id >= self.num_layers_down else None
                if self.training and use_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states, condition_emb, text_emb, freq_cis_img, residual,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(hidden_states, condition_emb, text_emb, freq_cis_img, residual, to_cache=to_cache)
                if block_id < self.num_layers_down - 2:
                    residuals.append(hidden_states)

        # Output
        hidden_states = self.final_layer(hidden_states, condition_emb)
        hidden_states = self.unpatchify(hidden_states, height//2, width//2)
        hidden_states, _ = hidden_states.chunk(2, dim=1)
        return hidden_states
    
    def state_dict_converter(self):
        return HunyuanDiTStateDictConverter()



class HunyuanDiTStateDictConverter():
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            name_ = name
            name_ = name_.replace(".default_modulation.", ".modulation.")
            name_ = name_.replace(".mlp.fc1.", ".mlp.0.")
            name_ = name_.replace(".mlp.fc2.", ".mlp.2.")
            name_ = name_.replace(".attn1.q_norm.", ".rota1.q_norm.")
            name_ = name_.replace(".attn2.q_norm.", ".rota2.q_norm.")
            name_ = name_.replace(".attn1.k_norm.", ".rota1.k_norm.")
            name_ = name_.replace(".attn2.k_norm.", ".rota2.k_norm.")
            name_ = name_.replace(".q_proj.", ".to_q.")
            name_ = name_.replace(".out_proj.", ".to_out.")
            name_ = name_.replace("text_embedding_padding", "text_emb_padding")
            name_ = name_.replace("mlp_t5.0.", "t5_embedder.0.")
            name_ = name_.replace("mlp_t5.2.", "t5_embedder.2.")
            name_ = name_.replace("pooler.", "t5_pooler.")
            name_ = name_.replace("x_embedder.", "patch_embedder.")
            name_ = name_.replace("t_embedder.", "timestep_embedder.")
            name_ = name_.replace("t5_pooler.to_q.", "t5_pooler.q_proj.")
            name_ = name_.replace("style_embedder.weight", "style_embedder")
            if ".kv_proj." in name_:
                param_k = param[:param.shape[0]//2]
                param_v = param[param.shape[0]//2:]
                state_dict_[name_.replace(".kv_proj.", ".to_k.")] = param_k
                state_dict_[name_.replace(".kv_proj.", ".to_v.")] = param_v
            elif ".Wqkv." in name_:
                param_q = param[:param.shape[0]//3]
                param_k = param[param.shape[0]//3:param.shape[0]//3*2]
                param_v = param[param.shape[0]//3*2:]
                state_dict_[name_.replace(".Wqkv.", ".to_q.")] = param_q
                state_dict_[name_.replace(".Wqkv.", ".to_k.")] = param_k
                state_dict_[name_.replace(".Wqkv.", ".to_v.")] = param_v
            elif "style_embedder" in name_:
                state_dict_[name_] = param.squeeze()
            else:
                state_dict_[name_] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
