import torch
from typing import Optional
from einops import rearrange
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor
    
def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    s_per_rank = x.shape[1]

    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    freqs = pad_freqs(freqs, s_per_rank * sp_size)
    freqs_rank = freqs[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]

    x_out = torch.view_as_real(x_out * freqs_rank).flatten(2)
    return x_out.to(x.dtype)

def usp_dit_forward(self,
            x: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            clip_feature: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            use_gradient_checkpointing: bool = False,
            use_gradient_checkpointing_offload: bool = False,
            **kwargs,
            ):
    t = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    context = self.text_embedding(context)
    
    if self.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = self.patchify(x)
    
    freqs = torch.cat([
        self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    # Context Parallel
    x = torch.chunk(
        x, get_sequence_parallel_world_size(),
        dim=1)[get_sequence_parallel_rank()]

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        else:
            x = block(x, context, t_mod, freqs)

    x = self.head(x, t)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, (f, h, w))
    return x


def usp_attn_forward(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    x = xFuserLongContextAttention()(
        None,
        query=q,
        key=k,
        value=v,
    )
    x = x.flatten(2)

    del q, k, v
    torch.cuda.empty_cache()
    return self.o(x)