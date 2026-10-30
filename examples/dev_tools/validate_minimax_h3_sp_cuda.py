"""CUDA H3 SP validation and synthetic activation benchmark; no model downloads.

Use torchrun with two or four processes. Numeric tolerances are declared here,
before execution. Synthetic benchmark measurements are not pretrained H3 claims.
"""
import argparse
import copy
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, inject_adapter_in_model
from test_minimax_h3_sequence_parallel import (
    cached_inputs, training_module, CachedDataset, TrainingCheckLogger,
)
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder


TOLERANCES = {
    "float32": dict(rtol=5e-4, atol=1e-5, relative_l2=1e-4),
    "bfloat16": dict(rtol=0.06, atol=0.006, relative_l2=0.03),
}


def compare(actual, expected, precision):
    a, b = actual.detach().float(), expected.detach().float()
    limits = TOLERANCES[precision]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, rtol=limits["rtol"], atol=limits["atol"])
    diff = a - b
    relative = (diff.norm() / b.norm().clamp_min(1e-8)).item()
    assert relative <= limits["relative_l2"], f"Relative L2 {relative} exceeds {limits}"
    return dict(max_absolute=diff.abs().max().item(), relative_l2=relative)


def synchronize_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ds_config(precision, accumulation=2):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": accumulation,
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": 3, "reduce_bucket_size": 65536,
            "stage3_prefetch_bucket_size": 65536,
            "stage3_param_persistence_threshold": 0,
            "overlap_comm": False,
        },
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": precision == "bfloat16"},
        "fp16": {"enabled": False},
        "steps_per_print": 100000,
    }


def make_training_module(device, sp_size, precision, lora):
    module = training_module(device, sp_size, lora=lora)
    dtype = getattr(torch, precision)
    module.to(dtype=dtype)
    module.pipe.torch_dtype = dtype
    # Upstream rebuilds RoPE frequencies; this parameter is unused by forward.
    module.pipe.dit.rope.inv_freq.requires_grad_(False)
    return module


def validation_case(device, engine_kind, precision, lora, checkpoint):
    reference = make_training_module(device, 1, precision, lora)
    parallel = make_training_module(device, dist.get_world_size(), precision, lora)
    parallel.sequence_parallel_size = 1  # manually synchronize both paths' RNG below
    if engine_kind == "zero3":
        import deepspeed
        from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param
        optimizer = torch.optim.SGD([p for p in parallel.parameters() if p.requires_grad], lr=0.05)
        engine, _, _, _ = deepspeed.initialize(model=parallel, optimizer=optimizer, config=ds_config(precision))
    else:
        engine = DDP(parallel, device_ids=[device.index], find_unused_parameters=True)
        optimizer = torch.optim.SGD([p for p in parallel.parameters() if p.requires_grad], lr=0.05)
    ref_params = {name: p for name, p in reference.named_parameters() if p.requires_grad}
    actual_params = {name: p for name, p in parallel.named_parameters() if p.requires_grad}
    assert list(ref_params) == list(actual_params)
    masters = {name: p.detach().float().clone() for name, p in ref_params.items()}
    initial = torch.cat([p.flatten().clone() for p in masters.values()])
    rows = []
    for step in range(2):
        reference.zero_grad(set_to_none=True)
        if engine_kind == "ddp":
            optimizer.zero_grad(set_to_none=True)
        loss_errors = []
        for micro in range(2):
            inputs = cached_inputs(1 + micro * 2, checkpoint=checkpoint)
            synchronize_seed(1200 + step * 2 + micro)
            expected = reference({}, inputs=copy.deepcopy(inputs))
            synchronize_seed(1200 + step * 2 + micro)
            actual = engine({}, inputs=copy.deepcopy(inputs))
            loss_errors.append(compare(actual, expected, precision))
            (expected / 2).backward()
            if engine_kind == "zero3":
                engine.backward(actual)  # DeepSpeed applies accumulation scaling
                if micro == 0:
                    engine.step()  # advances the micro-step, no parameter update
            else:
                (actual / 2).backward()
        ref_grads, actual_grads = [], []
        for name, param in actual_params.items():
            grad = safe_get_full_grad(param) if engine_kind == "zero3" else param.grad
            assert grad is not None and ref_params[name].grad is not None, name
            actual_grads.append(grad.detach().float().flatten())
            ref_grads.append(ref_params[name].grad.detach().float().flatten())
        expected_grad = torch.cat(ref_grads)
        assert expected_grad.norm() > 1e-7, "Vacuous zero-gradient test"
        grad_error = compare(torch.cat(actual_grads), expected_grad, precision)
        if engine_kind == "zero3":
            engine.step()
            # ZeRO keeps FP32 master parameters even for BF16 compute.
            for name, param in ref_params.items():
                masters[name].add_(param.grad.float(), alpha=-0.05)
                param.data.copy_(masters[name].to(param.dtype))
            actual_master = torch.cat([safe_get_full_fp32_param(p).detach().flatten() for p in actual_params.values()])
            expected_master = torch.cat([p.flatten() for p in masters.values()])
            update_error = compare(actual_master - initial, expected_master - initial, precision)
        else:
            optimizer.step()
            with torch.no_grad():
                for param in ref_params.values():
                    param.add_(param.grad, alpha=-0.05)
            update_error = compare(
                torch.cat([p.detach().float().flatten() for p in actual_params.values()]),
                torch.cat([p.detach().float().flatten() for p in ref_params.values()]), precision,
            )
        rows.append(dict(step=step + 1, losses=loss_errors, gradient=grad_error, update=update_error))
    if engine_kind == "zero3":
        assert engine.global_steps == 2, engine.global_steps
        engine.destroy()
    result = dict(lora=lora, checkpoint=checkpoint, optimizer_steps=2, rows=rows, passed=True)
    del engine, optimizer, parallel, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def launcher_case(device, engine_kind, precision):
    from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
    from diffsynth.diffusion.runner import launch_training_task
    kwargs = dict(gradient_accumulation_steps=2, mixed_precision="bf16" if precision == "bfloat16" else "no")
    if engine_kind == "zero3":
        kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=ds_config(precision), zero3_init_flag=False)
    else:
        kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(**kwargs)
    module = make_training_module(device, dist.get_world_size(), precision, lora=True)
    before = {n: p.detach().float().clone() for n, p in module.named_parameters() if p.requires_grad}
    logger = TrainingCheckLogger()
    launch_training_task(accelerator, CachedDataset(), module, logger,
                         num_workers=0, num_epochs=2, learning_rate=1e-3)
    assert logger.micro_steps == 10 and logger.sync_steps == 6, vars(logger)
    if engine_kind == "zero3":
        from deepspeed.utils import safe_get_full_fp32_param
        engine = accelerator.deepspeed_engine_wrapped.engine
        assert engine.global_steps == 6, engine.global_steps
        after = {n: safe_get_full_fp32_param(p).detach() for n, p in module.named_parameters() if p.requires_grad}
    else:
        after = {n: p.detach().float() for n, p in module.named_parameters() if p.requires_grad}
    assert any(not torch.equal(after[n], p) for n, p in before.items())
    for value in after.values():
        assert torch.isfinite(value).all()
        leader = value.clone()
        dist.broadcast(leader, src=0)
        torch.testing.assert_close(value, leader, rtol=0, atol=0)
    assert accelerator._schedulers[-1].state_dict()["last_epoch"] == 6
    return dict(case="actual_launcher", passed=True, micro_steps=10, optimizer_steps=6, scheduler_steps=6)


def benchmark(device, args):
    module = make_training_module(device, 1, args.precision, lora=False)
    synchronize_seed(314)
    configuration = dict(
        num_layers=6, token_refiner_num_layers=1, hidden_size=512,
        num_attention_heads=8, attention_head_dim=64, ffn_hidden_size=1536,
        text_dim=256, timestep_input_dim=128, time_embed_hidden_size=512,
        time_embed_dim=256, adaln_out_features=9216,
        final_adaln_out_features=1024, rope_inv_freq_len=8,
    )
    model = MiniMaxH3DiT(**configuration)
    model.requires_grad_(False)
    model = inject_adapter_in_model(LoraConfig(
        r=8, lora_alpha=8, target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
    ), model).to(device=device, dtype=getattr(torch, args.precision))
    if args.sp:
        model.enable_sequence_parallel(dist.get_world_size())
    module.pipe.dit = model
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    refs = [dict(kind="image", latent_t=1, latent_h=64, latent_w=64)] * args.references
    packed = builder._build_packed_ref2va(32, 2, 32, 32, 16, refs)
    inputs = (
        dict(input_latents=torch.randn(1, 24, 2, 32, 32),
             audio_input_latents=torch.randn(2, 32, 16), packed=packed,
             ref_visual_anchor=torch.randn(1024 * args.references, 96),
             use_gradient_checkpointing=args.checkpoint),
        dict(prompt_embeds=torch.randn(32, 256)), {},
    )
    module.sequence_parallel_size = 1  # identical cached inputs/seeds on every rank
    engine = DDP(module, device_ids=[device.index], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=1e-4)
    params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    measurements, losses = [], []
    for step in range(args.warmup + args.iterations):
        synchronize_seed(9000 + step)
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize()
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss = engine({}, inputs=inputs)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= args.warmup:
            measurements.append(elapsed)
            losses.append(loss.item())
    local = dict(rank=dist.get_rank(), device=torch.cuda.get_device_name(),
                 peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                 peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                 step_seconds=measurements, losses=losses)
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, local)
    return dict(case="synthetic_h3_activation_benchmark", passed=True, pretrained=False,
                parameters=params, trainable_parameters=trainable, configuration=configuration,
                references=args.references, sequence_length=packed["seq_len"], sp=args.sp,
                checkpoint=args.checkpoint, warmup=args.warmup, iterations=args.iterations,
                unique_samples_per_step=1, ranks=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "benchmark"], default="validate")
    parser.add_argument("--engine", choices=["ddp", "zero3"], default="ddp")
    parser.add_argument("--precision", choices=list(TOLERANCES), default="float32")
    parser.add_argument("--sp", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--references", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.references < 1 or args.warmup < 1 or args.iterations < 1:
        parser.error("references, warmup and iterations must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    allocation_limit = min((12 if args.mode == 'benchmark' else 4) * 2**30, torch.cuda.get_device_properties(device).total_memory)
    torch.cuda.set_per_process_memory_fraction(allocation_limit / torch.cuda.get_device_properties(device).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    report = dict(passed=False, torch=torch.__version__, world_size=dist.get_world_size(),
                  allocation_limit_mib=allocation_limit / 2**20,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  tolerances=TOLERANCES[args.precision], results=[])
    try:
        if args.mode == "benchmark":
            report["results"].append(benchmark(device, args))
        else:
            for lora in (False, True):
                for checkpoint in (False, True):
                    result = validation_case(device, args.engine, args.precision, lora, checkpoint)
                    report["results"].append(result)
                    if dist.get_rank() == 0:
                        print(json.dumps(result), flush=True)
            report["results"].append(launcher_case(device, args.engine, args.precision))
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Per-rank reports preserve the original failure when a peer fails first.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        path = args.report.with_name(args.report.stem + f".rank{dist.get_rank()}.json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        if dist.get_rank() == 0:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"passed": report["passed"], "report": str(args.report)}), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
