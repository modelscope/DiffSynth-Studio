import torch
from typing import Optional
from einops import rearrange
from yunchang.kernels import AttnType
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ... import IS_NPU_AVAILABLE
from ...core.device import parse_nccl_backend, parse_device_type
from ...core.gradient import gradient_checkpoint_forward


def initialize_usp(device_type):
    import torch.distributed as dist
    from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
    dist.init_process_group(backend=parse_nccl_backend(device_type), init_method="env://")
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )
    getattr(torch, device_type).set_device(dist.get_rank())


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    original_tensor_device = original_tensor.device
    if original_tensor.device == "npu":
        original_tensor = original_tensor.cpu()
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0).to(device=original_tensor_device)
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
    freqs_rank = freqs_rank.to(torch.complex64) if freqs_rank.device.type == "npu" else freqs_rank
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

    # Context Parallel
    chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
    pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
    chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
    x = chunks[get_sequence_parallel_rank()]

    for block in self.blocks:
        if self.training:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = self.head(x, t)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)
    x = x[:, :-pad_shape] if pad_shape > 0 else x

    # unpatchify
    x = self.unpatchify(x, (f, h, w))
    return x


def usp_vace_forward(
    self, x, vace_context, context, t_mod, freqs,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    # Compute full sequence length from the sharded x
    full_seq_len = x.shape[1] * get_sequence_parallel_world_size()

    # Embed vace_context via patch embedding
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([
        torch.cat([u, u.new_zeros(1, full_seq_len - u.size(1), u.size(2))],
                  dim=1) for u in c
    ])

    # Chunk VACE context along sequence dim BEFORE processing through blocks
    c = torch.chunk(c, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # Process through vace_blocks (self_attn already monkey-patched to usp_attn_forward)
    for block in self.vace_blocks:
        c = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            c, x, context, t_mod, freqs
        )

    # Hints are already sharded per-rank
    hints = torch.unbind(c)[:-1]
    return hints


def usp_attn_forward(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    attn_type = AttnType.FA
    ring_impl_type = "basic"
    if IS_NPU_AVAILABLE:
        attn_type = AttnType.NPU
        ring_impl_type = "basic_npu"
    x = xFuserLongContextAttention(attn_type=attn_type, ring_impl_type=ring_impl_type)(
        None,
        query=q,
        key=k,
        value=v,
    )
    x = x.flatten(2)

    del q, k, v
    getattr(torch, parse_device_type(x.device)).empty_cache()
    return self.o(x)


def get_current_chunk(x, dim=1):
    chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=dim)
    ndims = len(chunks[0].shape)
    pad_list = [0] * (2 * ndims)
    pad_end_index = 2 * (ndims - 1 - dim) + 1
    max_size = chunks[0].size(dim)
    chunks = [
        torch.nn.functional.pad(
            chunk, 
            tuple(pad_list[:pad_end_index] + [max_size - chunk.size(dim)] + pad_list[pad_end_index+1:]), 
            value=0
        ) 
        for chunk in chunks
    ]
    x = chunks[get_sequence_parallel_rank()]
    return x


def gather_all_chunks(x, seq_len=None, dim=1):
    x = get_sp_group().all_gather(x, dim=dim)
    if seq_len is not None:
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(0, seq_len)
        x = x[tuple(slices)]
    return x
