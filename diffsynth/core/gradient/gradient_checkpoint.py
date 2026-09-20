# Modified by Hygon Information Technology Co., Ltd., 2026.
import torch


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def create_custom_forward_use_reentrant(module, num_positional_args, kwarg_keys):
    def custom_forward(*inputs):
        positional_args = inputs[:num_positional_args]
        keyword_args = dict(zip(kwarg_keys, inputs[num_positional_args:]))
        return module(*positional_args, **keyword_args)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        kwarg_keys = tuple(kwargs)
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(
                    model,
                    len(args),
                    kwarg_keys,
                ),
                *all_args,
            )
        return model_output
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
