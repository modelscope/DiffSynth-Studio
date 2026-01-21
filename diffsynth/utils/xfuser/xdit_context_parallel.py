import torch
from typing import Optional
from einops import rearrange
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from ...core.device import parse_nccl_backend, parse_device_type
import logging

logger = logging.getLogger(__name__)

def initialize_usp(device_type, sp_size):
    import torch.distributed as dist
    from xfuser.core.distributed import (
        initialize_model_parallel,
        init_distributed_environment,
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_data_parallel_world_size,
        get_data_parallel_rank,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend=parse_nccl_backend(device_type), init_method="env://")

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

    sp_degree = sp_size
    dp_degree = int(dist.get_world_size() / sp_degree)
    initialize_model_parallel(
        data_parallel_degree=dp_degree,
        sequence_parallel_degree=sp_degree,
        ring_degree=1,
        ulysses_degree=sp_degree,
    )
    logger.info(f"[init usp] rank: {dist.get_rank()}, world_size: {dist.get_world_size()}, "
                f"sp world size: {get_sequence_parallel_world_size()}, "
                f"sp rank: {get_sequence_parallel_rank()}, "
                f"dp world size: {get_data_parallel_world_size()}, "
                f"dp rank: {get_data_parallel_rank()}")

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
    freqs_rank = freqs_rank.to(torch.complex64) if freqs_rank.device == "npu" else freqs_rank
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
    chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
    pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
    chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
    x = chunks[get_sequence_parallel_rank()]

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
    x = x[:, :-pad_shape] if pad_shape > 0 else x

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

    '''
    Refer to commit https://github.com/xdit-project/xDiT/pull/598 for the xfuser backward error.
    xFuserRingFlashAttnFunc has 17 inputs (including ctx), but it inherits the backward() method from RingFlashAttnFunc which only returns 16 values (3 gradients + 13 Nones)!

The Math
Parent class (RingFlashAttnFunc): 14 forward inputs → backward returns 3 gradients + 11 Nones = 14 returns xFuser class (xFuserRingFlashAttnFunc): 17 forward inputs → backward should return 3 gradients + 14 Nones = 17 returns Actual: backward only returns 14 returns (inherited from parent without override) Error: PyTorch expects 17 gradients but gets only 14 → expected 17, got 13 (13 = 14 - 1 for ctx)
    '''
    x = xFuserLongContextAttention()(
        None,
        query=q,
        key=k,
        value=v,
    )
    x = x.flatten(2)

    del q, k, v
    getattr(torch, parse_device_type(x.device)).empty_cache()
    return self.o(x)
