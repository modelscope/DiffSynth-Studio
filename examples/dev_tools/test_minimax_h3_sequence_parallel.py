"""Tiny real H3 regressions; no pretrained weights, downloads or GPU required.

Run from the repository root:
    DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch torchrun --standalone \
        --nproc_per_node=2 examples/dev_tools/test_minimax_h3_sequence_parallel.py
Use --device cuda for the same numerical checks with NCCL (not a memory benchmark).
"""
import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, inject_adapter_in_model

from diffsynth.diffusion.loss import FlowMatchSFTMiniMaxH3AudioVideoLoss
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.diffusion.sequence_parallel import sequence_parallel_inputs, prepare_sequence_parallel_dataloader
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, MiniMaxH3Unit_PackedSequenceBuilder,
)
from examples.minimax_h3.model_training.train import MiniMaxH3TrainingModule


def tiny_model(device, lora=False):
    torch.manual_seed(123)
    model = MiniMaxH3DiT(
        num_layers=2, token_refiner_num_layers=1, hidden_size=24,
        num_attention_heads=4, attention_head_dim=8, ffn_hidden_size=48,
        text_dim=16, timestep_input_dim=8, time_embed_hidden_size=24,
        time_embed_dim=12, adaln_out_features=432,
        final_adaln_out_features=48, rope_inv_freq_len=1,
    )
    if lora:
        model.requires_grad_(False)
        model = inject_adapter_in_model(LoraConfig(
            r=2, lora_alpha=2, target_modules=["qkv_proj", "out_proj", "fc1", "fc2"],
        ), model)
    return model.to(device)


def direct_inputs(length, device, checkpoint):
    torch.manual_seed(456)
    tags = torch.tensor([1, 1, 0, 0, 2, 2] + [0] * (length - 8) + [-1, -1], device=device)
    img, audio, text = [(tags == t).nonzero().flatten() for t in (0, 2, 1)]
    return dict(
        x=torch.randn(1, length, 96, device=device),
        audio_x=torch.randn(1, length, 32, device=device),
        img_position_ids=torch.randn(1, length, 3, device=device),
        unique_timesteps=torch.tensor([0.2, 0.8, 1.0], device=device),
        inverse_indices=torch.arange(length, device=device) % 3,
        token_tags=tags, prompt_embeds=torch.randn(2, 16, device=device),
        img_pos_info={"position_ids": img}, audio_pos_info={"position_ids": audio},
        text_pos_info={"position_ids": text}, img_pos_for_infer_output_info={"position_ids": img},
        packed_seq_params={"cu_seqlens_q": torch.tensor([0, length - 2, length], device=device), "max_seqlen_q": length - 2},
        refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, 2, 2], device=device), "max_seqlen_q": 2},
        update_mask=torch.tensor([0, 0] + [1] * (len(img) - 2), device=device),
        update_audio_mask=torch.tensor([0, 1], device=device),
        use_gradient_checkpointing=checkpoint,
    )


def close(a, b):
    torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
    return (a - b).abs().max().item() if a.numel() else 0.0


def compare_parameters(reference, parallel, grads):
    maximum = 0.0
    for (name, a), (other, b) in zip(reference.named_parameters(), parallel.named_parameters()):
        assert name == other
        if grads:
            if a.grad is None or b.grad is None:
                assert a.grad is None and b.grad is None, name
                continue
            a, b = a.grad, b.grad
        try:
            maximum = max(maximum, close(a, b))
        except AssertionError as error:
            raise AssertionError(f"Parameter {name}, gradients={grads}: {error}") from error
    return maximum


def check_direct(device, length, checkpoint, lora):
    reference = tiny_model(device, lora)
    parallel = copy.deepcopy(reference)
    parallel.enable_sequence_parallel(dist.get_world_size())
    wrapped = DDP(parallel, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=True)
    args = direct_inputs(length, device, checkpoint)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.05) for m in (reference, parallel)]
    errors = {"output": 0.0, "gradient": 0.0, "update": 0.0}
    local_lengths = []
    hook = parallel.blocks[0].register_forward_pre_hook(lambda module, args: local_lengths.append(len(args[0])))
    # Two optimizer steps, each accumulating two different micro-batches.
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            args["x"] = args["x"] + 0.03
            with wrapped.no_sync() if micro == 0 else nullcontext():
                expected, actual = reference(**args), wrapped(**args)
                errors["output"] = max(errors["output"], *(close(a, b) for a, b in zip(expected, actual)))
                losses = [sum((x - 0.1 * (step + 1)).square().mean() for x in outputs) / 2 for outputs in (expected, actual)]
                close(*losses)
                losses[0].backward()
                losses[1].backward()
        errors["gradient"] = max(errors["gradient"], compare_parameters(reference, parallel, True))
        for optimizer in optimizers:
            optimizer.step()
        errors["update"] = max(errors["update"], compare_parameters(reference, parallel, False))
    hook.remove()
    assert set(local_lengths) == {(length + dist.get_world_size() - 1) // dist.get_world_size()}
    return dict(case="direct", length=length, checkpoint=checkpoint, lora=lora, max_absolute_error=errors)


def cached_inputs(reference_count, checkpoint=False):
    torch.manual_seed(789)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    refs = [dict(kind="image", latent_t=1, latent_h=4, latent_w=4)] * reference_count
    refs += [dict(kind="audio", ref_audio_t=2)]
    packed = builder._build_packed_ref2va(3, 2, 4, 4, 3, refs)
    return (
        dict(input_latents=torch.randn(1, 24, 2, 4, 4),
             audio_input_latents=torch.randn(2, 32, 3), packed=packed,
             ref_visual_anchor=torch.randn(4 * reference_count, 96),
             ref_audio_anchor=torch.randn(4, 32), use_gradient_checkpointing=checkpoint),
        dict(prompt_embeds=torch.randn(3, 16)),
        dict(prompt_embeds=torch.randn(3, 16), packed=copy.deepcopy(packed)),
    )


def training_module(device, sp_size, lora=False, cfg=1.0, audio_weight=1.0):
    # Construct the real training module with a tiny DiT and cached conditioning;
    # only the pretrained checkpoint/encoder loading step is bypassed.
    module = MiniMaxH3TrainingModule.__new__(MiniMaxH3TrainingModule)
    torch.nn.Module.__init__(module)
    module.sequence_parallel_size = sp_size
    module.pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch.float32)
    module.pipe.dit = tiny_model(device, lora)
    module.pipe.units = []
    module.pipe.scheduler.set_timesteps(1000, training=True)
    module.pipe.scheduler_audio.set_timesteps(1000, training=True)
    if sp_size > 1:
        module.pipe.dit.enable_sequence_parallel(sp_size)
    module.task = "sft:train"
    module.task_to_loss = {module.task: lambda pipe, shared, pos, neg: FlowMatchSFTMiniMaxH3AudioVideoLoss(
        pipe, training_cfg_scale=cfg, audio_loss_weight=audio_weight, inputs_nega=neg, **shared, **pos,
    )}
    return module


def check_flow_loss(device, refs, checkpoint, cfg, audio_weight):
    reference = training_module(device, 1)
    parallel = training_module(device, dist.get_world_size())
    # Wrap outside the real training module, just as Accelerate does.
    parallel.sequence_parallel_size = 1  # synchronize once below for both paths
    for module in (reference, parallel):
        module.task_to_loss[module.task] = lambda pipe, shared, pos, neg: FlowMatchSFTMiniMaxH3AudioVideoLoss(
            pipe, training_cfg_scale=cfg, audio_loss_weight=audio_weight, inputs_nega=neg, **shared, **pos,
        )
    wrapped = DDP(parallel, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=True)
    batch = cached_inputs(refs, checkpoint)
    with sequence_parallel_inputs({}, batch, device) as (data, inputs):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        expected = reference(data, inputs=copy.deepcopy(inputs))
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        actual = wrapped(data, inputs=copy.deepcopy(inputs))
    error = close(expected, actual)
    expected.backward()
    actual.backward()
    grad_error = compare_parameters(reference, parallel, True)
    return dict(case="flow_loss", references=refs, checkpoint=checkpoint, cfg=cfg,
                audio_weight=audio_weight, loss_error=error, gradient_error=grad_error)


def check_inputs(device):
    rank = dist.get_rank()
    torch.manual_seed(100 + rank)
    # The leader consumes one seed; the context restores the state after that draw.
    state = torch.get_rng_state()
    if rank == 0:
        torch.randint(0, 2**31, ())
    expected_state = torch.get_rng_state()
    torch.set_rng_state(state)
    with sequence_parallel_inputs({"rank": rank}, {"value": torch.tensor(rank)}, device) as (data, inputs):
        assert data["rank"] == 0 and inputs["value"].item() == 0
        values = torch.randn(4, device=device)
        gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, values)
        for other in gathered:
            close(values, other)
    assert torch.equal(torch.get_rng_state(), expected_state)
    return dict(case="input_broadcast_and_rng", passed=True)


class CachedDataset(torch.utils.data.Dataset):
    load_from_cache = True

    def __len__(self):
        return 5  # non-divisible epoch checks final accumulation flush

    def __getitem__(self, index):
        return cached_inputs(1 + index % 2, checkpoint=True)


class TrainingCheckLogger:
    def __init__(self):
        self.micro_steps = 0
        self.sync_steps = 0

    def on_step_end(self, accelerator, model, save_steps, loss):
        assert torch.isfinite(loss)
        self.micro_steps += 1
        self.sync_steps += int(accelerator.sync_gradients)

    def on_epoch_end(self, *args):
        pass

    def on_training_end(self, *args):
        pass


def check_launcher(device):
    accelerator = Accelerator(cpu=device.type == "cpu", gradient_accumulation_steps=2,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    model = training_module(device, dist.get_world_size(), lora=True)
    initial = copy.deepcopy(model.state_dict())
    logger = TrainingCheckLogger()
    launch_training_task(accelerator, CachedDataset(), model, logger,
                         num_workers=0, num_epochs=2, learning_rate=1e-3)
    assert logger.micro_steps == 10, logger.micro_steps
    assert logger.sync_steps == 6, logger.sync_steps
    scheduler = accelerator._schedulers[-1]
    assert scheduler.state_dict()["last_epoch"] == 6, scheduler.state_dict()
    changed = False
    for name, param in model.named_parameters():
        leader = param.detach().clone()
        dist.broadcast(leader, src=0)
        close(param, leader)
        changed |= not torch.equal(param, initial[name])
    assert changed
    return dict(case="actual_training_launcher_lora", micro_steps=logger.micro_steps,
                optimizer_steps=logger.sync_steps, scheduler_steps=6, passed=True)


def check_guards(device):
    from accelerate.utils import DistributedType
    checked = []

    def rejected(name, call, message):
        try:
            call()
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"Unsupported configuration accepted: {name}")
        checked.append(name)

    # Constructor checks must run before any checkpoint loading or downloads.
    rejected("nonpositive_size", lambda: MiniMaxH3TrainingModule(sequence_parallel_size=0), "positive")
    rejected("preprocessing", lambda: MiniMaxH3TrainingModule(
        sequence_parallel_size=dist.get_world_size(), task="sft:data_process"), "only supported")
    rejected("world_mismatch", lambda: MiniMaxH3TrainingModule(
        sequence_parallel_size=dist.get_world_size() + 1), "Initialize exactly")
    model = tiny_model(device)
    rejected("head_count", lambda: model.enable_sequence_parallel(3), "head count")
    model.blocks[0].attn.forward = lambda *args, **kwargs: None
    rejected("custom_attention", lambda: model.enable_sequence_parallel(dist.get_world_size()), "Custom H3")
    model = tiny_model(device)
    model.enable_sequence_parallel(dist.get_world_size())
    rejected("controlnet", lambda: model(**direct_inputs(13, device, False), control_hints={}), "ControlNet")
    accelerator = SimpleNamespace(distributed_type=DistributedType.NO)
    rejected("unsupported_backend", lambda: prepare_sequence_parallel_dataloader(accelerator, None), "requires DDP")
    accelerator.distributed_type = DistributedType.MULTI_CPU
    accelerator.parallelism_config = object()
    rejected("hybrid_parallelism", lambda: prepare_sequence_parallel_dataloader(accelerator, None), "parallelism_config")
    accelerator.parallelism_config = None
    accelerator.state = SimpleNamespace(deepspeed_plugin=SimpleNamespace(
        deepspeed_config={"train_micro_batch_size_per_gpu": 2}))
    rejected("micro_batch_size", lambda: prepare_sequence_parallel_dataloader(accelerator, None), "micro_batch_size")
    return dict(case="unsupported_configuration_guards", checked=checked, passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if os.environ.get("H3_SP_TEST_MAX_ALLOC_GIB"):
            limit = float(os.environ["H3_SP_TEST_MAX_ALLOC_GIB"]) * 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit / torch.cuda.get_device_properties(device).total_memory))
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=120))
    try:
        assert dist.get_world_size() in (2, 4), "Tiny model has four heads; use two or four ranks."
        results = [check_inputs(device), check_guards(device)]
        for lora in (False, True):
            for checkpoint in (False, True):
                for length in (12, 13):
                    results.append(check_direct(device, length, checkpoint, lora))
        for refs, checkpoint, cfg, audio_weight in ((1, False, 1.0, 1.0), (3, True, 1.0, 0.0), (2, True, 2.0, 1.0)):
            results.append(check_flow_loss(device, refs, checkpoint, cfg, audio_weight))
        results.append(check_launcher(device))
        if dist.get_rank() == 0:
            report = dict(passed=True, world_size=dist.get_world_size(), device=str(device), torch=torch.__version__, results=results)
            print(json.dumps(report, indent=2), flush=True)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
