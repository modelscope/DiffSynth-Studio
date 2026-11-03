"""Exercise H3 SP's real LoRA save/load path, without pretrained weights.

Use torchrun with two ranks. --device cpu runs DDP/FP32; CUDA also supports
--engine zero3 --precision bfloat16. --output is required for checkpoint files.
This checks a weight-only restart with a fresh optimizer, not exact job resume.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from safetensors.torch import load_file

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_training_task
from test_minimax_h3_sequence_parallel import cached_inputs, CachedDataset
from validate_minimax_h3_sp_cuda import compare, ds_config, make_training_module, TOLERANCES


class CheckpointLogger(ModelLogger):
    def __init__(self, output):
        super().__init__(str(output), remove_prefix_in_ckpt="pipe.dit.")
        self.sync_steps = 0
        self.last_model = None

    def on_step_end(self, accelerator, model, save_steps, loss):
        assert torch.isfinite(loss)
        self.sync_steps += int(accelerator.sync_gradients)
        self.last_model = model
        super().on_step_end(accelerator, model, save_steps, loss=loss)


def load_adapter(device, precision, checkpoint, sp_size):
    # Use the same adapter-loading method as train.py's --lora_checkpoint.
    model = make_training_module(device, 1, precision, lora=False)
    model.switch_pipe_to_training_mode(
        model.pipe, lora_base_model="dit", lora_target_modules="qkv_proj,out_proj,fc1,fc2",
        lora_rank=2, lora_checkpoint=str(checkpoint), task="sft:train",
    )
    if sp_size > 1:
        model.pipe.dit.enable_sequence_parallel(sp_size)
    # Both paths use manually synchronized seeds in the comparison below.
    model.sequence_parallel_size = 1
    return model


def check(device, args):
    kwargs = dict(cpu=device.type == "cpu", gradient_accumulation_steps=2,
                  mixed_precision="bf16" if args.precision == "bfloat16" else "no")
    if args.engine == "zero3":
        config = ds_config(args.precision)
        # Matches the repository's full/accelerate_config_zero3.yaml.
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=config, zero3_init_flag=False)
    else:
        kwargs["kwargs_handlers"] = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(**kwargs)
    model = make_training_module(device, dist.get_world_size(), args.precision, lora=True)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    logger = CheckpointLogger(args.output)
    launch_training_task(accelerator, CachedDataset(), model, logger,
                         num_workers=0, num_epochs=2, learning_rate=1e-3)
    assert logger.num_steps == 10 and logger.sync_steps == 6
    # Collective on ZeRO-3: all ranks must participate, not just the writer.
    state = accelerator.get_state_dict(logger.last_model)
    # ZeRO-3 materializes the result only on rank zero, even though every rank
    # must enter get_state_dict. Broadcast this tiny test model for comparison.
    payload = [{n: p.detach().cpu().clone() for n, p in state.items()} if accelerator.is_main_process else None]
    dist.broadcast_object_list(payload, src=0)
    expected = payload[0]
    accelerator.wait_for_everyone()
    assert (args.output / "epoch-0.safetensors").is_file()
    checkpoint = args.output / "epoch-1.safetensors"
    saved = load_file(str(checkpoint))
    expected_adapter = model.export_trainable_state_dict(expected, remove_prefix="pipe.dit.")
    assert saved.keys() == expected_adapter.keys() and saved
    assert not any("sequence_parallel" in name for name in expected)
    for name in saved:
        torch.testing.assert_close(saved[name], expected_adapter[name], rtol=0, atol=0)
    assert any(not torch.equal(expected[n], value) for n, value in initial.items())

    reference = load_adapter(device, args.precision, checkpoint, 1)
    parallel = load_adapter(device, args.precision, checkpoint, dist.get_world_size())
    for loaded in (reference, parallel):
        actual = loaded.state_dict()
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=0)

    wrapped = DDP(parallel, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=True)
    inputs = cached_inputs(3, checkpoint=True)
    torch.manual_seed(2026)
    expected_loss = reference({}, inputs=copy.deepcopy(inputs))
    torch.manual_seed(2026)
    actual_loss = wrapped({}, inputs=copy.deepcopy(inputs))
    loss_error = compare(actual_loss, expected_loss, args.precision)
    expected_loss.backward()
    actual_loss.backward()
    pairs = [(a, b) for a, b in zip(reference.parameters(), parallel.parameters()) if a.requires_grad]
    assert pairs and all(a.grad is not None and b.grad is not None for a, b in pairs)
    gradient_error = compare(torch.cat([b.grad.flatten() for a, b in pairs]),
                             torch.cat([a.grad.flatten() for a, b in pairs]), args.precision)
    # Explicit fresh optimizers: saved adapters contain no Adam/RNG/scheduler state.
    for loaded in (reference, parallel):
        torch.optim.SGD([p for p in loaded.parameters() if p.requires_grad], lr=0.05).step()
    update_error = compare(torch.cat([b.detach().flatten() for a, b in pairs]),
                           torch.cat([a.detach().flatten() for a, b in pairs]), args.precision)
    return dict(passed=True, micro_steps=10, optimizer_steps=6, saved_tensors=len(saved),
                checkpoint=str(checkpoint), all_reloaded_weights_exact=True,
                restart="weights_only_fresh_optimizer", continuation_loss=loss_error,
                continuation_gradient=gradient_error, continuation_update=update_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--engine", choices=["ddp", "zero3"], default="ddp")
    parser.add_argument("--precision", choices=list(TOLERANCES), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cpu" and (args.engine != "ddp" or args.precision != "float32"):
        parser.error("CPU checkpoint checks require DDP/float32")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 2**30 / total))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=180))
    report = dict(passed=False, engine=args.engine, precision=args.precision, device=str(device),
                  torch=torch.__version__, world_size=dist.get_world_size(), tolerances=TOLERANCES[args.precision])
    try:
        report.update(check(device, args))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"report.rank{dist.get_rank()}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        if dist.get_rank() == 0:
            print(json.dumps(report, indent=2), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
