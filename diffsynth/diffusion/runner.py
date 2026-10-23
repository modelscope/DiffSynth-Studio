import os, json, torch, importlib
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import DistributedType
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager
from diffsynth.core.fsdp_training import apply_fsdp_to_model


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


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
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer
    enable_fsdp = bool(getattr(args, "enable_fsdp", False))

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer_class = get_optimizer_class(customized_optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_fsdp:
        if enable_model_cpu_offload:
            raise ValueError("--enable_fsdp and --enable_model_cpu_offload are mutually exclusive.")
        # FSDP2 replaces module parameters with DTensor params, so the optimizer must be
        # created AFTER sharding. Shard every trainable pipeline model (frozen models are
        # not loaded in the train task).
        for name, child in model.pipe.named_children():
            if not isinstance(child, torch.nn.Module):
                continue
            if any(p.requires_grad for p in child.parameters()):
                print(f"FSDP: sharding pipeline model `{name}`.")
                apply_fsdp_to_model(child, offload_optimizer=True)
        model.pipe.device = accelerator.device
        optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    elif enable_model_cpu_offload:
        optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    elif getattr(accelerator, "distributed_type", None) == DistributedType.DEEPSPEED:
        # DeepSpeed ZeRO-3: the model must stay on CPU until deepspeed.initialize partitions
        # it across ranks (a full 33B-param copy does not fit on one 64GB NPU). Do NOT call
        # model.to(device) here - the engine handles placement.
        optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
    else:
        optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
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
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
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
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

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
