import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
import time
import logging

logger = logging.getLogger(__name__)

def build_dataloader(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    num_workers: int = 1,
    sp_size: int = 1,
):
    if sp_size > 1:
        # When using sequence parallel, it is necessary to ensure that when the sampler uses iter to
        # fetch data from the dataloader, each rank within the same SP group obtains the same sample.
        if accelerator is not None:
            world_size = accelerator.num_processes
            rank = accelerator.process_index
        else:
            raise ValueError(f"Accelerator is None.")

        dp_size = world_size // sp_size
        if dp_size * sp_size != world_size:
            raise ValueError(
                    f"world_size={world_size}, sp_size={sp_size}, world_size should be diviaible by sp_size"
            )

        dp_rank = rank // sp_size
        sp_rank = rank % sp_size
        logger.info(f"accelerator.processid={rank}, accelerator.num_processes={world_size}, "
                    f"sp_size={sp_size}, dp_size={dp_size}, dp_rank={dp_rank}")
    else:
        if accelerator is not None:
            dp_size = accelerator.num_processes
            dp_rank = accelerator.process_index
        else:
            raise ValueError(f"Accelerator is None.")
        logger.info(f"dp_size={dp_size}, dp_rank={dp_rank}")

    sampler = torch.utils.data.DistributedSampler(dataset=dataset, num_replicas=dp_size, rank=dp_rank)

    dataloader_kwargs = dict(
        dataset=dataset,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )
    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    return dataloader

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    sp_size: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        sp_size = args.sp_size

    train_step = 0

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = build_dataloader(accelerator, dataset, num_workers, sp_size)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    for epoch_id in range(num_epochs):
        progress = tqdm(
            dataloader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch_id + 1}/{num_epochs}",
        )

        for data in progress:
            logger.info(f"[train] id{accelerator.process_index}, step{train_step}, prompt: {data['prompt']}")

            iter_start = time.time()
            timing = {}
            if data is None:
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                forward_start = time.time()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                torch.cuda.synchronize()
                timing["forward"] = time.time() - forward_start

                backward_start = time.time()
                accelerator.backward(loss)
                torch.cuda.synchronize()
                timing["backward"] = time.time() - backward_start

                optim_start = time.time()
                optimizer.step()
                torch.cuda.synchronize()
                timing["optimizer"] = time.time() - optim_start

                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

            torch.cuda.synchronize()
            iter_end = time.time()
            timing["step"] = iter_end - iter_start
            train_step += 1

            if accelerator.is_main_process:
                def format_time(key: str) -> str:
                    value = timing.get(key, 0.0)
                    return f"{value:.3f}s"

                postfix_dict = {
                    "loss": f"{loss.item():.5f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.5e}",
                    "step/t": format_time("step"),
                    "fwd/t": format_time("forward"),
                    "bwd/t": format_time("backward"),
                    "opt/t": format_time("optimizer"),
                }
                progress.set_postfix(postfix_dict)
                log_msg = f"[Step {train_step:6d}] | " + " | ".join(f"{k}: {v}" for k, v in postfix_dict.items())
                progress.write(log_msg)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)

def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
