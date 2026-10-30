"""Input synchronization for single-group H3 sequence-parallel training."""
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def sequence_parallel_inputs(data, inputs, device):
    """Broadcast rank zero's batch and seed; restore RNG after the leader's seed draw.

    Batches must still be on CPU. This deliberately broadcasts the original
    batch (including reference media), favoring correctness over input I/O
    efficiency. Precomputed CPU conditioning caches avoid repeating encoders.
    """
    seed = torch.randint(0, 2**31, ()).item() if dist.get_rank() == 0 else None
    payload = [data, inputs, seed] if dist.get_rank() == 0 else [None, None, None]
    dist.broadcast_object_list(payload, src=0)
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(payload[2])
        if device.type == "cuda":
            torch.cuda.manual_seed(payload[2])
        yield payload[0], payload[1]


def prepare_sequence_parallel_dataloader(accelerator, dataloader):
    """Keep one full, unsharded epoch on every SP rank, with accumulation hooks."""
    from accelerate.data_loader import prepare_data_loader
    from accelerate.utils import DistributedType
    if accelerator.distributed_type not in (
        DistributedType.MULTI_CPU, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED,
    ):
        raise ValueError("H3 sequence parallel training requires DDP or DeepSpeed across all ranks.")
    if getattr(accelerator, "parallelism_config", None) is not None:
        raise ValueError("Do not combine H3's sequence_parallel_size with Accelerate parallelism_config.")
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is not None:
        # prepare(model, optimizer, scheduler) has no dataloader from which to
        # infer an automatic micro-batch size. This launcher uses batch size 1.
        config = plugin.deepspeed_config
        if config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
            config["train_micro_batch_size_per_gpu"] = 1
        elif config["train_micro_batch_size_per_gpu"] != 1:
            raise ValueError("H3's SP launcher requires train_micro_batch_size_per_gpu=1 (or auto).")
    return prepare_data_loader(
        dataloader, device=accelerator.device, num_processes=1, process_index=0,
        put_on_device=False, rng_types=[],
    )
