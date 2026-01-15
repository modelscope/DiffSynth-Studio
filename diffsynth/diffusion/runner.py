import os, torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _pad_frames(frames, target_frames):
    if target_frames is None:
        return frames
    if len(frames) >= target_frames:
        return frames[:target_frames]
    if len(frames) == 0:
        raise ValueError("Cannot pad empty frame list.")
    pad_frame = frames[-1]
    return frames + [pad_frame] * (target_frames - len(frames))


def _frame_to_tensor(frame, min_value=-1.0, max_value=1.0):
    if isinstance(frame, torch.Tensor):
        tensor = frame
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor
    array = np.array(frame, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * ((max_value - min_value) / 255.0) + min_value
    return tensor


def _frames_to_tensor(frames, min_value=-1.0, max_value=1.0):
    frame_tensors = [_frame_to_tensor(frame, min_value=min_value, max_value=max_value) for frame in frames]
    return torch.stack(frame_tensors, dim=1)


def _collate_batch(batch, data_file_keys, num_frames):
    if len(batch) == 1:
        return batch[0]
    single_frame_keys = {"reference_image", "vace_reference_image"}
    output = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample.get(key) for sample in batch]
        if key in data_file_keys:
            is_mask = "mask" in key
            min_value = 0.0 if is_mask else -1.0
            max_value = 1.0 if is_mask else 1.0
            if any(value is None for value in values):
                raise ValueError(f"Missing key '{key}' in one or more batch samples.")
            if key in single_frame_keys:
                frames = []
                for value in values:
                    if isinstance(value, list):
                        if len(value) == 0:
                            raise ValueError(f"Key '{key}' has empty frame list.")
                        frames.append(value[0])
                    else:
                        frames.append(value)
                tensors = [_frame_to_tensor(frame, min_value=min_value, max_value=max_value) for frame in frames]
                output[key] = torch.stack(tensors, dim=0)
            else:
                tensors = []
                for value in values:
                    if isinstance(value, list):
                        padded = _pad_frames(value, num_frames)
                        tensors.append(_frames_to_tensor(padded, min_value=min_value, max_value=max_value))
                    elif isinstance(value, torch.Tensor):
                        tensors.append(value)
                    else:
                        raise ValueError(f"Unsupported value type for key '{key}': {type(value)}")
                output[key] = torch.stack(tensors, dim=0)
        else:
            output[key] = values
    return output


def run_validation(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    num_workers: int,
    batch_size: int,
    data_file_keys: list[str],
    num_frames: int,
    max_batches: int = None,
):
    if dataset is None:
        return None
    if batch_size > 1:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: _collate_batch(batch, data_file_keys, num_frames),
            num_workers=num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader = accelerator.prepare(dataloader)
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for step, data in enumerate(tqdm(dataloader, desc="Eval")):
            if max_batches is not None and step >= max_batches:
                break
            if dataset.load_from_cache:
                loss = model({}, inputs=data)
            else:
                loss = model(data)
            loss = loss.detach().float()
            loss = accelerator.gather(loss)
            losses.append(loss.flatten())
    if was_training:
        model.train()
    if not losses:
        return None
    mean_loss = torch.cat(losses).mean().item()
    if accelerator.is_main_process:
        print(f"Eval loss: {mean_loss:.6f}")
    return mean_loss


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
    val_dataset: torch.utils.data.Dataset = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        data_file_keys = args.data_file_keys.split(",")
        num_frames = getattr(args, "num_frames", None)
        val_num_workers = args.val_dataset_num_workers
        val_batch_size = args.val_batch_size or batch_size
        val_data_file_keys = (args.val_data_file_keys or args.data_file_keys).split(",")
        eval_every_n_epochs = args.eval_every_n_epochs
        eval_max_batches = args.eval_max_batches
    else:
        batch_size = 1
        data_file_keys = []
        num_frames = None
        val_num_workers = 0
        val_batch_size = 1
        val_data_file_keys = []
        eval_every_n_epochs = 0
        eval_max_batches = None
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if batch_size > 1:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: _collate_batch(batch, data_file_keys, num_frames),
            num_workers=num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        epoch_loss_sum = None
        epoch_steps = 0
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                loss_value = loss.detach().float()
                if epoch_loss_sum is None:
                    epoch_loss_sum = loss_value
                else:
                    epoch_loss_sum = epoch_loss_sum + loss_value
                epoch_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if epoch_loss_sum is None:
            epoch_loss_sum = torch.tensor(0.0, device=accelerator.device)
        steps_tensor = torch.tensor(float(epoch_steps), device=epoch_loss_sum.device)
        loss_stats = torch.stack([epoch_loss_sum, steps_tensor]).unsqueeze(0)
        gathered_stats = accelerator.gather(loss_stats)
        if accelerator.is_main_process:
            total_loss = gathered_stats[:, 0].sum().item()
            total_steps = gathered_stats[:, 1].sum().item()
            avg_loss = total_loss / total_steps if total_steps > 0 else float("nan")
            print(f"Train loss (epoch {epoch_id}): {avg_loss:.6f}")
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if val_dataset is not None and eval_every_n_epochs > 0 and (epoch_id + 1) % eval_every_n_epochs == 0:
            run_validation(
                accelerator,
                val_dataset,
                model,
                val_num_workers,
                val_batch_size,
                val_data_file_keys,
                num_frames,
                max_batches=eval_max_batches,
            )
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
