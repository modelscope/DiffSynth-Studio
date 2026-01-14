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
        batch_size = args.batch_size
        data_file_keys = args.data_file_keys.split(",")
        num_frames = getattr(args, "num_frames", None)
    else:
        batch_size = 1
        data_file_keys = []
        num_frames = None
    
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
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
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
