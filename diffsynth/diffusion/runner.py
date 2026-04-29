import os, torch, time
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


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
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        cpu_offload = args.cpu_offload

    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    optimize_on_cpu = True
    offload_manager = None
    if cpu_offload:
        from diffsynth.core import setup_layer_offload
        from diffsynth.core.offload_training.layer import move_gradients_to_cpu
        optimize_on_cpu = getattr(args, 'optimize_on_cpu', False)
        param_size_threshold = getattr(args, 'param_size_threshold', 500)
        offload_manager = setup_layer_offload(
            model, target_device=accelerator.device,
            optimize_on_cpu=optimize_on_cpu,
            param_size_threshold=param_size_threshold,
        )
        if hasattr(model, 'pipe') and hasattr(model.pipe, 'device'):
            model.pipe.device = accelerator.device

    for epoch_id in range(num_epochs):
        step_times = []
        for step_idx, data in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
            step_start = time.time()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                mem_after_forward = torch.cuda.max_memory_allocated() / (1024*1024)
                accelerator.backward(loss)
                if optimize_on_cpu and cpu_offload:
                    move_gradients_to_cpu(model)
                optimizer.step()
                if offload_manager:
                    offload_manager.reset_in_recompute()
                torch.cuda.reset_peak_memory_stats()
                mem_after_optimizer = torch.cuda.max_memory_allocated() / (1024*1024)
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
            step_time = time.time() - step_start
            step_times.append(step_time)
            if accelerator.is_local_main_process:
                accelerator.print(f"Step {step_idx}: {step_time:.2f}s, fwd_mem={mem_after_forward:.1f}MB, opt_mem={mem_after_optimizer:.1f}MB")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

        if accelerator.is_local_main_process:
            avg_time = sum(step_times) / len(step_times) if step_times else 0
            peak_mem = torch.cuda.max_memory_allocated() / (1024*1024)
            accelerator.print(f"Epoch {epoch_id}: avg step time={avg_time:.2f}s, peak memory={peak_mem:.1f} MB")

    model_logger.on_training_end(accelerator, model, save_steps)

    if accelerator.is_local_main_process:
        final_peak = torch.cuda.max_memory_allocated() / (1024*1024)
        accelerator.print(f"FINAL PEAK MEMORY: {final_peak:.1f} MB")


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
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
