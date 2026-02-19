import torch
import torch.nn as nn
from functools import wraps
import time
from collections import defaultdict
import flash_attn
from einops import rearrange
from torch.utils.flop_counter import conv_flop_count

def get_dit_flops(model):
    def get_dit_flops(dit_block_model):
        total_flops = 0
        for sub_model in dit_block_model.modules():
            total_flops += getattr(sub_model, '__flops__', 0)
        return total_flops

    total_flops = 0
    total_duration = 0
    for sub_module in model.modules():
        if sub_module.__class__.__name__ == 'DiTBlock':
            total_flops += get_dit_flops(sub_module)
            total_duration += getattr(sub_module, '__duration__', 0)

    Tflops = total_flops / 1e12
    return Tflops

def get_flops(model):
    def get_module_flops(module):
        if not hasattr(module, "__flops__"):
            module.__flops__ = 0
        
        flops = module.__flops__
        # iterate over immediate children modules
        for child in module.children():
            flops += get_module_flops(child)
        return flops

    t5_flops = 0
    wan_flops = 0
    vae_flops = 0
    for module in model.modules():
        if module.__class__.__name__ == 'WanTextEncoder':
            t5_flops = get_module_flops(module)
        if module.__class__.__name__ == 'WanModel':
            wan_flops = get_module_flops(module)
        if module.__class__.__name__ == 'WanVideoVAE38':
            vae_flops = get_module_flops(module)
    return t5_flops / 1e12, wan_flops / 1e12, vae_flops / 1e12

def print_model_profile(model):
    def get_module_flops(module):
        if not hasattr(module, "__flops__"):
            module.__flops__ = 0
        
        flops = module.__flops__
        # iterate over immediate children modules
        for child in module.children():
            flops += get_module_flops(child)
        return flops

    def get_module_duration(module):
        if not hasattr(module, "__duration__"):
            module.__duration__ = 0
 
        duration = module.__duration__
        if duration == 0:  # e.g. ModuleList
            for m in module.children():
                duration += get_module_duration(m)
        return duration 

    def flops_repr(module):
        flops = get_module_flops(module)
        duration = get_module_duration(module) * 1000
        items = [
            "{:,} flops".format(flops),
            "{:.3f} ms".format(duration),
        ]
        original_extra_repr = module.original_extra_repr()
        if original_extra_repr:
            items.append(original_extra_repr)
        return ", ".join(items)

    def add_extra_repr(module):
        flops_extra_repr = flops_repr.__get__(module)
        if module.extra_repr != flops_extra_repr:
            module.original_extra_repr = module.extra_repr
            module.extra_repr = flops_extra_repr
            assert module.extra_repr != module.original_extra_repr
    
    def del_extra_repr(module):
        if hasattr(module, "original_extra_repr"):
            module.extra_repr = module.original_extra_repr
            del module.original_extra_repr
    
    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)

def get_module_flops(module, *args, result=None, **kwargs):
    module_type = module.__class__.__name__
    module_original_fwd = module._original_forward.__name__

    if module_type == 'RMSNorm':
        x = args[0]
        return x.numel() * 4

    elif module_type == 'RMS_norm':
        x = args[0]
        return x.numel() * 4

    elif module_type == 'Dropout':
        x = args[0]
        return x.numel() * 2

    elif module_type == 'LayerNorm':
        x = args[0]
        has_affine = module.weight is not None
        return x.numel() * (5 if has_affine else 4)
    
    elif module_type == 'Linear':
        x = args[0]
        return x.numel() * module.weight.size(0) * 2
    
    elif module_type == 'ReLU':
        x = args[0]
        return x.numel()
    
    elif module_type == 'GELU':
        x = args[0]
        return x.numel()

    elif module_type == 'SiLU':
        x = args[0]
        return x.numel()

    elif module_type == 'Conv3d' or module_type == 'CausalConv3d' or module_type == 'Conv2d':
        x_shape = args[0].shape
        weight = getattr(module, 'weight', None)
        w_shape = weight.shape
        out_shape = result.shape

        flops = conv_flop_count(
            x_shape=x_shape,
            w_shape=w_shape,
            out_shape=out_shape,
            transposed=False
        )
        return flops

    # AttentionModule input is 3D shape, USP input is 4D shape.
    #
    # 3D shape:
    # q [batch, target_seq_len, Dim]
    # k [batch, source_seq_len, Dim]
    # v [batch, source_seq_len, Dim]
    # flops = (batch * target_seq_len * source_seq_len) * Dim * 2
    #         + (batch * target_seq_len * Dim) * source_seq_len * 2
    #       = 4 * (batch * target_seq_len * source_seq_len * Dim)
    #
    # 4D shape:
    # q [batch, target_seq_len, head, dim]
    # k [batch, source_seq_len, head, dim]
    # v [batch, source_seq_len, head, dim]
    # flops = 4 * (batch * target_seq_len * source_seq_len * head * dim)
    # 
    elif module_type == 'AttentionModule':
        q = args[0]
        k = args[1]
        v = args[2]

        b, ts, dq = q.shape
        _, ss,  _ = k.shape
        _,  _, dv = v.shape
        flops = (b * ts * ss * dq) * 2 + (b * ts * ss * dv) * 2
        return flops

    elif module_original_fwd == 'usp_attn_forward' or module_type == 'T5Attention':
        q_shape = module.q_shape
        k_shape = module.k_shape
        v_shape = module.v_shape

        b, ts, n, dq = q_shape
        _, ss, _,  _ = k_shape
        _,  _, _, dv = v_shape
        flops = (b * ts * ss * n * dq) * 2 + (b * ts * ss * n * dv) * 2
        return flops

    elif module_type == 'GateModule':
        x = args[0]
        return x.numel() * 2

    elif module_type == 'T5LayerNorm':
        x = args[0]
        return x.numel() * 4

    elif module_type == 'T5RelativeEmbedding':
        lq = args[0]
        lk = args[1]
        return lq * lk * 10

    else:
        return 0

def flops_counter(flops_func=None):
    def decorator(forward_func):
        @wraps(forward_func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()

            result = forward_func(self, *args, **kwargs)

            self.__flops__ = get_module_flops(self, *args, result=result, **kwargs)

            end_time = time.perf_counter()
            self.__duration__ = (end_time - start_time)

            return result
        return wrapper
    return decorator


def wrap_existing_module(module, verbose_profiling=False):
    # save original fwd
    module.verbose_profiling = verbose_profiling
    module._original_forward = module.forward

    @flops_counter()
    def profiled_forward(self, x, *args, **kwargs):
        return module._original_forward(x, *args, **kwargs)

    module.forward = profiled_forward.__get__(module, type(module))
    return module

def profile_entire_model(model, verbose_profiling=True):
    for name, module in model.named_modules():
        wrap_existing_module(module, verbose_profiling)
    return model

def unwrap_existing_module(module):
    if hasattr(module, "_original_forward"):
        module.forward = module._original_forward
        del module._original_forward

    if hasattr(module, "verbose_profiling"):
        del module.verbose_profiling
    return module

def unprofile_entire_model(model):
    for name, module in model.named_modules():
        unwrap_existing_module(module)
    return model

