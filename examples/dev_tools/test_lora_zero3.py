"""Run with torchrun --standalone --nproc_per_node=2 and two CUDA devices.

Uses tiny layers and no pretrained checkpoints. Requires the training extra.
"""
import argparse
from datetime import timedelta
import gc
import itertools
import json
import os
from pathlib import Path
import platform

import deepspeed
from deepspeed.utils import safe_get_full_grad
import torch
import torch.distributed as dist

from diffsynth.utils.lora.general import GeneralLoRALoader


def rank_errors(error):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    return errors


def check_case(sharded, dtype, conv, bias):
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    config = {
        'train_micro_batch_size_per_gpu': 2,
        'gradient_accumulation_steps': 1,
        'zero_optimization': {
            'stage': 3 if sharded else 0,
            'reduce_bucket_size': 128,
            'stage3_prefetch_bucket_size': 128,
            'stage3_param_persistence_threshold': 0,
            'overlap_comm': False,
        },
        'zero_allow_untested_optimizer': True,
        'fp16': {'enabled': dtype == torch.float16, 'loss_scale': 1.0},
        'bf16': {'enabled': dtype == torch.bfloat16},
        'steps_per_print': 100000,
    }
    def make_layer():
        return torch.nn.Conv2d(4, 6, 1, bias=bias) if conv else torch.nn.Linear(4, 6, bias=bias)

    with deepspeed.zero.Init(enabled=sharded, config_dict_or_path=config, dtype=dtype):
        model = torch.nn.Sequential(make_layer())
    if not sharded:
        model.to(device=device, dtype=dtype)
    params = list(model.parameters())
    shape = (6, 4, 1, 1) if conv else (6, 4)
    base = torch.arange(24, device=device, dtype=dtype).reshape(shape) / 32
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0, enabled=sharded):
        with torch.no_grad():
            model[0].weight.copy_(base)
            if bias:
                model[0].bias.fill_(0.25)
    a = torch.arange(8, dtype=dtype).reshape(2, 4) / 16
    b = torch.arange(12, dtype=dtype).reshape(6, 2) / 16
    expected_weight = base + (0.5 * (b.to(device) @ a.to(device))).reshape(shape)
    state = {
        '0.lora_A.weight': a[:, :, None, None] if conv else a,
        '0.lora_B.weight': b[:, :, None, None] if conv else b,
    }
    error = None
    try:
        GeneralLoRALoader(device=device, torch_dtype=dtype).fuse_lora_to_base_model(model, state, alpha=0.5)
        if sharded:
            assert model[0].weight.numel() == 0, 'Weight was not repartitioned'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = rank_errors(error)
    if any(errors):
        return {'phase': 'fusion', 'errors_by_rank': errors}

    tolerance = {torch.float32: 1e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    reference = torch.nn.Sequential(make_layer()).to(device=device, dtype=dtype)
    with torch.no_grad():
        reference[0].weight.copy_(expected_weight)
        if bias:
            reference[0].bias.fill_(0.25)
    errors = None
    with deepspeed.zero.GatheredParameters(params, enabled=sharded):
        try:
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except Exception as exc:
            errors = f'{type(exc).__name__}: {exc}'
    gathered_errors = rank_errors(errors)
    if any(gathered_errors):
        return {'phase': 'fused_weights', 'errors_by_rank': gathered_errors}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    inputs = torch.arange(32 if conv else 8, device=device, dtype=dtype).reshape((2, 4, 2, 2) if conv else (2, 4)) / 64
    inputs = inputs + dist.get_rank() / 32
    expected_output = reference(inputs)
    expected_loss = expected_output.float().square().mean()
    expected_loss.backward()
    expected_grads = []
    for param in reference.parameters():
        grad = param.grad.float()
        dist.all_reduce(grad)
        expected_grads.append(grad / dist.get_world_size())
    actual_output = engine(inputs)
    loss = actual_output.float().square().mean()
    engine.backward(loss)
    actual_grads = [safe_get_full_grad(param) if sharded else param.grad for param in model.parameters()]
    error = None
    try:
        torch.testing.assert_close(actual_output, expected_output, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(loss).item()
        for actual, expected in zip(actual_grads, expected_grads):
            assert actual is not None and torch.isfinite(actual).all().item()
            assert actual.abs().max().item() > 0
            torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = rank_errors(error)
    engine.step()
    with deepspeed.zero.GatheredParameters(params, enabled=sharded):
        try:
            for actual, initial, grad in zip(model.parameters(), reference.parameters(), expected_grads):
                expected = (initial.float() - 0.05 * grad).to(dtype)
                assert not torch.equal(actual, initial), 'Optimizer did not update the parameter'
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    update_errors = rank_errors(error)
    engine.destroy()
    del engine, optimizer, model, reference
    gc.collect()
    torch.cuda.empty_cache()
    return {'phase': 'training', 'errors_by_rank': [a or b for a, b in zip(errors, update_errors)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 2, 'Run this regression with two processes'
    results = []
    for sharded, dtype, conv, bias in itertools.product((False, True), (torch.float32, torch.float16, torch.bfloat16), (False, True), (False, True)):
        result = check_case(sharded, dtype, conv, bias)
        result.update(sharded=sharded, dtype=str(dtype), conv=conv, bias=bias)
        result['passed'] = not any(result['errors_by_rank'])
        results.append(result)
        if dist.get_rank() == 0:
            print(json.dumps(result), flush=True)
    report = {'python': platform.python_version(), 'torch': torch.__version__, 'deepspeed': deepspeed.__version__, 'world_size': dist.get_world_size(), 'results': results}
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps(report, indent=2))
    dist.destroy_process_group()
    assert all(row['passed'] for row in results), 'See per-rank failures in the JSON report'


if __name__ == '__main__':
    main()
