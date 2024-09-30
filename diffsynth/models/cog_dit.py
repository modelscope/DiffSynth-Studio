import torch
from einops import rearrange, repeat
from .sd3_dit import TimestepEmbeddings
from .attention import Attention
from .utils import load_state_dict_from_folder
from .tiler import TileWorker2Dto3D
import numpy as np



class CogPatchify(torch.nn.Module):
    def __init__(self, dim_in, dim_out, patch_size) -> None:
        super().__init__()
        self.proj = torch.nn.Conv3d(dim_in, dim_out, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = rearrange(hidden_states, "B C T H W -> B (T H W) C")
        return hidden_states
    


class CogAdaLayerNorm(torch.nn.Module):
    def __init__(self, dim, dim_cond, single=False):
        super().__init__()
        self.single = single
        self.linear = torch.nn.Linear(dim_cond, dim * (2 if single else 6))
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)


    def forward(self, hidden_states, prompt_emb, emb):
        emb = self.linear(torch.nn.functional.silu(emb))
        if self.single:
            shift, scale = emb.unsqueeze(1).chunk(2, dim=2)
            hidden_states = self.norm(hidden_states) * (1 + scale) + shift
            return hidden_states
        else:
            shift_a, scale_a, gate_a, shift_b, scale_b, gate_b = emb.unsqueeze(1).chunk(6, dim=2)
            hidden_states = self.norm(hidden_states) * (1 + scale_a) + shift_a
            prompt_emb = self.norm(prompt_emb) * (1 + scale_b) + shift_b
            return hidden_states, prompt_emb, gate_a, gate_b



class CogDiTBlock(torch.nn.Module):
    def __init__(self, dim, dim_cond, num_heads):
        super().__init__()
        self.norm1 = CogAdaLayerNorm(dim, dim_cond)
        self.attn1 = Attention(q_dim=dim, num_heads=48, head_dim=dim//num_heads, bias_q=True, bias_kv=True, bias_out=True)
        self.norm_q = torch.nn.LayerNorm((dim//num_heads,), eps=1e-06, elementwise_affine=True)
        self.norm_k = torch.nn.LayerNorm((dim//num_heads,), eps=1e-06, elementwise_affine=True)

        self.norm2 = CogAdaLayerNorm(dim, dim_cond)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*4),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(dim*4, dim)
        )
    

    def apply_rotary_emb(self, x, freqs_cis):
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    

    def process_qkv(self, q, k, v, image_rotary_emb, text_seq_length):
        q = self.norm_q(q)
        k = self.norm_k(k)
        q[:, :, text_seq_length:] = self.apply_rotary_emb(q[:, :, text_seq_length:], image_rotary_emb)
        k[:, :, text_seq_length:] = self.apply_rotary_emb(k[:, :, text_seq_length:], image_rotary_emb)
        return q, k, v
        

    def forward(self, hidden_states, prompt_emb, time_emb, image_rotary_emb):
        # Attention
        norm_hidden_states, norm_encoder_hidden_states, gate_a, gate_b = self.norm1(
            hidden_states, prompt_emb, time_emb
        )
        attention_io = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        attention_io = self.attn1(
            attention_io,
            qkv_preprocessor=lambda q, k, v: self.process_qkv(q, k, v, image_rotary_emb, prompt_emb.shape[1])
        )

        hidden_states = hidden_states + gate_a * attention_io[:, prompt_emb.shape[1]:]
        prompt_emb = prompt_emb + gate_b * attention_io[:, :prompt_emb.shape[1]]

        # Feed forward
        norm_hidden_states, norm_encoder_hidden_states, gate_a, gate_b = self.norm2(
            hidden_states, prompt_emb, time_emb
        )
        ff_io = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_io = self.ff(ff_io)

        hidden_states = hidden_states + gate_a * ff_io[:, prompt_emb.shape[1]:]
        prompt_emb = prompt_emb + gate_b * ff_io[:, :prompt_emb.shape[1]]

        return hidden_states, prompt_emb



class CogDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify = CogPatchify(16, 3072, 2)
        self.time_embedder = TimestepEmbeddings(3072, 512)
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.blocks = torch.nn.ModuleList([CogDiTBlock(3072, 512, 48) for _ in range(42)])
        self.norm_final = torch.nn.LayerNorm((3072,), eps=1e-05, elementwise_affine=True)
        self.norm_out = CogAdaLayerNorm(3072, 512, single=True)
        self.proj_out = torch.nn.Linear(3072, 64, bias=True)


    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)
    

    def get_3d_rotary_pos_embed(
        self, embed_dim, crops_coords, grid_size, temporal_size, theta: int = 10000, use_real: bool = True
    ):
        start, stop = crops_coords
        grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
        grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)

        # Compute dimensions for each axis
        dim_t = embed_dim // 4
        dim_h = embed_dim // 8 * 3
        dim_w = embed_dim // 8 * 3

        # Temporal frequencies
        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
        grid_t = torch.from_numpy(grid_t).float()
        freqs_t = torch.einsum("n , f -> n f", grid_t, freqs_t)
        freqs_t = freqs_t.repeat_interleave(2, dim=-1)

        # Spatial frequencies for height and width
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        freqs_h = torch.einsum("n , f -> n f", grid_h, freqs_h)
        freqs_w = torch.einsum("n , f -> n f", grid_w, freqs_w)
        freqs_h = freqs_h.repeat_interleave(2, dim=-1)
        freqs_w = freqs_w.repeat_interleave(2, dim=-1)

        # Broadcast and concatenate tensors along specified dimension
        def broadcast(tensors, dim=-1):
            num_tensors = len(tensors)
            shape_lens = {len(t.shape) for t in tensors}
            assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
            shape_len = list(shape_lens)[0]
            dim = (dim + shape_len) if dim < 0 else dim
            dims = list(zip(*(list(t.shape) for t in tensors)))
            expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
            assert all(
                [*(len(set(t[1])) <= 2 for t in expandable_dims)]
            ), "invalid dimensions for broadcastable concatenation"
            max_dims = [(t[0], max(t[1])) for t in expandable_dims]
            expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
            expanded_dims.insert(dim, (dim, dims[dim]))
            expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
            tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
            return torch.cat(tensors, dim=dim)

        freqs = broadcast((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)

        t, h, w, d = freqs.shape
        freqs = freqs.view(t * h * w, d)

        # Generate sine and cosine components
        sin = freqs.sin()
        cos = freqs.cos()

        if use_real:
            return cos, sin
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            return freqs_cis
    

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ):
        grid_height = height // 2
        grid_width = width // 2
        base_size_width = 720 // (8 * 2)
        base_size_height = 480 // (8 * 2)

        grid_crops_coords = self.get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = self.get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin


    def unpatchify(self, hidden_states, height, width):
        hidden_states = rearrange(hidden_states, "B (T H W) (C P Q) -> B C T (H P) (W Q)", P=2, Q=2, H=height//2, W=width//2)
        return hidden_states
    

    def build_mask(self, T, H, W, dtype, device, is_bound):
        t = repeat(torch.arange(T), "T -> T H W", T=T, H=H, W=W)
        h = repeat(torch.arange(H), "H -> T H W", T=T, H=H, W=W)
        w = repeat(torch.arange(W), "W -> T H W", T=T, H=H, W=W)
        border_width = (H + W) // 4
        pad = torch.ones_like(h) * border_width
        mask = torch.stack([
            pad if is_bound[0] else t + 1,
            pad if is_bound[1] else T - t,
            pad if is_bound[2] else h + 1,
            pad if is_bound[3] else H - h,
            pad if is_bound[4] else w + 1,
            pad if is_bound[5] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=dtype, device=device)
        mask = rearrange(mask, "T H W -> 1 1 T H W")
        return mask
    

    def tiled_forward(self, hidden_states, timestep, prompt_emb, tile_size=(60, 90), tile_stride=(30, 45)):
        B, C, T, H, W = hidden_states.shape
        value = torch.zeros((B, C, T, H, W), dtype=hidden_states.dtype, device=hidden_states.device)
        weight = torch.zeros((B, C, T, H, W), dtype=hidden_states.dtype, device=hidden_states.device)

        # Split tasks
        tasks = []
        for h in range(0, H, tile_stride):
            for w in range(0, W, tile_stride):
                if (h-tile_stride >= 0 and h-tile_stride+tile_size >= H) or (w-tile_stride >= 0 and w-tile_stride+tile_size >= W):
                    continue
                h_, w_ = h + tile_size, w + tile_size
                if h_ > H: h, h_ = max(H - tile_size, 0), H
                if w_ > W: w, w_ = max(W - tile_size, 0), W
                tasks.append((h, h_, w, w_))

        # Run
        for hl, hr, wl, wr in tasks:
            mask = self.build_mask(
                value.shape[2], (hr-hl), (wr-wl),
                hidden_states.dtype, hidden_states.device,
                is_bound=(True, True, hl==0, hr>=H, wl==0, wr>=W)
            )
            model_output = self.forward(hidden_states[:, :, :, hl:hr, wl:wr], timestep, prompt_emb)
            value[:, :, :, hl:hr, wl:wr] += model_output * mask
            weight[:, :, :, hl:hr, wl:wr] += mask
        value = value / weight

        return value


    def forward(self, hidden_states, timestep, prompt_emb, image_rotary_emb=None, tiled=False, tile_size=90, tile_stride=30, use_gradient_checkpointing=False):
        if tiled:
            return TileWorker2Dto3D().tiled_forward(
                forward_fn=lambda x: self.forward(x, timestep, prompt_emb),
                model_input=hidden_states,
                tile_size=tile_size, tile_stride=tile_stride,
                tile_device=hidden_states.device, tile_dtype=hidden_states.dtype,
                computation_device=self.context_embedder.weight.device, computation_dtype=self.context_embedder.weight.dtype
            )
        num_frames, height, width = hidden_states.shape[-3:]
        if image_rotary_emb is None:
            image_rotary_emb = self.prepare_rotary_positional_embeddings(height, width, num_frames, device=self.context_embedder.weight.device)
        hidden_states = self.patchify(hidden_states)
        time_emb = self.time_embedder(timestep, dtype=hidden_states.dtype)
        prompt_emb = self.context_embedder(prompt_emb)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                hidden_states, prompt_emb = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states, prompt_emb, time_emb, image_rotary_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states, prompt_emb = block(hidden_states, prompt_emb, time_emb, image_rotary_emb)

        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]
        hidden_states = self.norm_out(hidden_states, prompt_emb, time_emb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.unpatchify(hidden_states, height, width)

        return hidden_states
    

    @staticmethod
    def state_dict_converter():
        return CogDiTStateDictConverter()
    

    @staticmethod
    def from_pretrained(file_path, torch_dtype=torch.bfloat16):
        model = CogDiT().to(torch_dtype)
        state_dict = load_state_dict_from_folder(file_path, torch_dtype=torch_dtype)
        state_dict = CogDiT.state_dict_converter().from_diffusers(state_dict)
        model.load_state_dict(state_dict)
        return model



class CogDiTStateDictConverter:
    def __init__(self):
        pass


    def from_diffusers(self, state_dict):
        rename_dict = {
            "patch_embed.proj.weight": "patchify.proj.weight",
            "patch_embed.proj.bias": "patchify.proj.bias",
            "patch_embed.text_proj.weight": "context_embedder.weight",
            "patch_embed.text_proj.bias": "context_embedder.bias",
            "time_embedding.linear_1.weight": "time_embedder.timestep_embedder.0.weight",
            "time_embedding.linear_1.bias": "time_embedder.timestep_embedder.0.bias",
            "time_embedding.linear_2.weight": "time_embedder.timestep_embedder.2.weight",
            "time_embedding.linear_2.bias": "time_embedder.timestep_embedder.2.bias",

            "norm_final.weight": "norm_final.weight",
            "norm_final.bias": "norm_final.bias",
            "norm_out.linear.weight": "norm_out.linear.weight",
            "norm_out.linear.bias": "norm_out.linear.bias",
            "norm_out.norm.weight": "norm_out.norm.weight",
            "norm_out.norm.bias": "norm_out.norm.bias",
            "proj_out.weight": "proj_out.weight",
            "proj_out.bias": "proj_out.bias",
        }
        suffix_dict = {
            "norm1.linear.weight": "norm1.linear.weight",
            "norm1.linear.bias": "norm1.linear.bias",
            "norm1.norm.weight": "norm1.norm.weight",
            "norm1.norm.bias": "norm1.norm.bias",
            "attn1.norm_q.weight": "norm_q.weight",
            "attn1.norm_q.bias": "norm_q.bias",
            "attn1.norm_k.weight": "norm_k.weight",
            "attn1.norm_k.bias": "norm_k.bias",
            "attn1.to_q.weight": "attn1.to_q.weight",
            "attn1.to_q.bias": "attn1.to_q.bias",
            "attn1.to_k.weight": "attn1.to_k.weight",
            "attn1.to_k.bias": "attn1.to_k.bias",
            "attn1.to_v.weight": "attn1.to_v.weight",
            "attn1.to_v.bias": "attn1.to_v.bias",
            "attn1.to_out.0.weight": "attn1.to_out.weight",
            "attn1.to_out.0.bias": "attn1.to_out.bias",
            "norm2.linear.weight": "norm2.linear.weight",
            "norm2.linear.bias": "norm2.linear.bias",
            "norm2.norm.weight": "norm2.norm.weight",
            "norm2.norm.bias": "norm2.norm.bias",
            "ff.net.0.proj.weight": "ff.0.weight",
            "ff.net.0.proj.bias": "ff.0.bias",
            "ff.net.2.weight": "ff.2.weight",
            "ff.net.2.bias": "ff.2.bias",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                if name == "patch_embed.proj.weight":
                    param = param.unsqueeze(2)
                state_dict_[rename_dict[name]] = param
            else:
                names = name.split(".")
                if names[0] == "transformer_blocks":
                    suffix = ".".join(names[2:])
                    state_dict_[f"blocks.{names[1]}." + suffix_dict[suffix]] = param
        return state_dict_
    

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
