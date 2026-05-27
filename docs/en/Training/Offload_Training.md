# Offload Training

This document introduces the Offload Training feature in DiffSynth-Studio, which significantly reduces GPU memory usage during training by moving model weights layer-by-layer between CPU and GPU.

> **Note**: Offload Training currently supports single-GPU training only and is not compatible with multi-GPU (DDP) setups.

## What is Offload Training

When training large-scale models (e.g., Qwen-Image with 60 layers, Wan2.1-14B with 40 layers), all layer weights must reside on the GPU simultaneously, consuming tens of GB of memory for weights alone. The core idea of Offload Training is: **at any given moment, only load the weights of the currently computing module onto the GPU, and immediately offload them back to CPU after computation**, reducing memory usage from O(N × params_per_layer) to O(1 × params_per_layer).

This feature is implemented via PyTorch's Module Hook mechanism and requires no modifications to model code.

## How It Works

### Core Mechanism

`OffloadTrainingManager` scans the model and registers 4 hooks for each managed module:

```
forward_pre_hook   → Load module weights from CPU to GPU (onload)
module.forward()   → Normal forward computation
forward_hook       → Offload module weights from GPU back to CPU (offload)

backward_pre_hook  → Reload module weights from CPU to GPU (onload)
module.backward()  → Compute gradients
backward_hook      → Offload module weights back to CPU (offload)
```

### Parameter and Buffer Classification

Different offload strategies are applied depending on whether parameters are trainable and for buffer types:

| Type | Offloader Class | Behavior |
|---------------|----------------|----------|
| Non-trainable (`requires_grad=False`) | `StaticParamOffloader` | Copies weights to pre-allocated pinned memory at init, maintaining a permanent CPU copy, and replaces `param.data` with an empty GPU placeholder (freeing GPU memory); onload asynchronously copies from CPU to GPU, offload reassigns `param.data` to the placeholder (no PCIe transfer back) |
| Trainable + `enable_optimizer_cpu_offload=True` | `TrainableParamOffloader` | Weights change during training, so no static copy is kept; onload/offload via `param.data.to(device)` with actual data transfer; also moves `param.grad` to CPU after backward |
| Trainable + `enable_optimizer_cpu_offload=False` | `AlwaysOnGPUParamOffloader` | Moves parameters to GPU at init and never offloads; suitable for LoRA training (small number of trainable params) |
| Module Buffers (e.g., BatchNorm's `running_mean`/`running_var`) | `BufferOffloader` | Similar to `StaticParamOffloader`: copies buffer to pinned memory at init; onload asynchronously copies from CPU to GPU, offload reassigns `module._buffers[name]` back to the CPU copy |

### Pinned Memory Pool

`StaticParamOffloader` and `BufferOffloader` need to allocate a pinned memory copy on CPU for each non-trainable parameter/buffer (pinned memory enables asynchronous non-blocking CPU→GPU transfers, much faster than regular pageable memory).

**Problem**: PyTorch's `pin_memory()` allocates memory through `CachingHostAllocator`, which rounds up each allocation size to the next power of two. For example, a 17MB tensor actually allocates 32MB. Large models have thousands of parameter tensors, and allocating each independently via `pin_memory()` leads to massive memory waste (measured inflation of 50%~100%).

**Solution**: `PinnedArenaPool` pre-allocates a few large blocks of pinned memory (i.e., arenas — large pre-allocated memory regions from which all small objects are carved out), then uses bump-pointer allocation to compactly carve out space for each tensor, avoiding the per-tensor rounding waste:

- `from_model()` scans all non-trainable parameters and buffers in the model, computing total size
- Decomposes total size into several power-of-two sized chunks (each chunk is a `PinnedBuffer`)
- Allocation sequentially probes chunks for remaining space; bump-pointer advances to complete allocation (only 64-byte alignment, no rounding waste)
- Automatically grows new chunks when space is insufficient
- Falls back to per-tensor `pin_memory()` on exceptions

### Gradient Checkpointing Compatibility

Gradient Checkpointing re-executes forward during backward (recomputing activations), which re-triggers `forward_hook`. This is solved via the `_in_recompute` set:

- First forward: normal offload, module added to `_in_recompute`
- Recomputed forward (during backward): detects module in `_in_recompute`, skips offload, keeps weights on GPU for backward
- When `after_backward()` is called: clears `_in_recompute`, preparing for the next step

### Hook Registration Granularity

`OffloadTrainingManager` registers hooks at leaf module granularity by default (`nn.Linear`, `nn.LayerNorm`, etc.), meaning each leaf module is independently onloaded/offloaded. Additionally, "orphan parameters" and "orphan buffers" not managed by any leaf module are automatically collected and hooked.

**Experimental**: The `cpu_offload_split_threshold` parameter (unit: MB) allows adjusting hook registration granularity. When set, modules with total parameters exceeding the threshold are recursively split into children, while modules below the threshold are hooked as a whole. This feature may not be compatible with all model architectures in the current version and is disabled by default.

### Training Loop Integration

Execution flow in `runner.py`:

```python
# When enable_model_cpu_offload=True:
# 1. Model does NOT call model.to(device), stays on CPU
# 2. Only prepare optimizer, dataloader, scheduler (model is NOT prepared)
# 3. Create OffloadTrainingManager, which auto-registers hooks on the model

# Training loop:
loss = model(data)
accelerator.backward(loss)
offload_manager.after_backward()  # Clear recompute marks + move gradients to CPU
optimizer.step()
optimizer.zero_grad()
```

## Usage

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_model_cpu_offload` | False | Enable layer-wise offload training |
| `--enable_optimizer_cpu_offload` | False | Used with `--enable_model_cpu_offload`; moves trainable params and optimizer to CPU |
| `--cpu_offload_split_threshold` | None | Experimental (unit: MB); modules above this threshold are recursively split |

### Parameter Combinations

| Scenario | `--enable_model_cpu_offload` | `--enable_optimizer_cpu_offload` | Effect |
|----------|:---------------:|:-------------------:|--------|
| Default training | ❌ | ❌ | All weights and optimizer on GPU |
| Offload non-trainable params | ✅ | ❌ | Non-trainable params offloaded layer-by-layer; trainable params and optimizer stay on GPU |
| Offload all params | ✅ | ✅ | All params offloaded layer-by-layer; gradients and optimizer run on CPU |

### Example

Simply add `--enable_model_cpu_offload` to your existing training command. Example with Qwen-Image LoRA training:

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_dataset \
  --dataset_metadata_path data/example_dataset/metadata.json \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --enable_model_cpu_offload
```

For full offload (optimizer also on CPU), add `--enable_optimizer_cpu_offload`:

```bash
  --enable_model_cpu_offload \
  --enable_optimizer_cpu_offload
```

### Compatibility

| Feature | Compatible | Notes |
|---------|:----------:|-------|
| Gradient Checkpointing | ✅ | `_in_recompute` mechanism handles recomputation |
| Accelerate DDP (multi-GPU) | ⚠️ | In enable_model_cpu_offload mode, model is not wrapped by DDP (no `accelerator.prepare(model)`), so **gradient allreduce is not performed**. Multi-GPU training compatibility is not guaranteed; each GPU trains independently without gradient synchronization |
| Split Training | ✅ | `launch_data_process_task` also supports `--enable_model_cpu_offload` |
| DeepSpeed | ❌ | ZeRO's parameter gathering conflicts with hooks |

### Notes

- With `--enable_model_cpu_offload` enabled, the model never calls `model.to(device)`; weights are managed entirely by hooks
- Training speed decreases due to CPU↔GPU transfers (typically 2-10x slower); larger models see greater slowdown; suitable for memory-constrained scenarios
- Recommended to use with `--use_gradient_checkpointing` to further reduce activation memory
- `--enable_optimizer_cpu_offload` only supports gradient accumulation steps of 1 (`--gradient_accumulation_steps 1`)

## Integrating Offload Training Module in Other Codebases

The Offload Training module is relatively independent, so developers can integrate it into other codebases. Below is a code example with 4GB VRAM usage.

```python
import torch
from tqdm import tqdm

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4096, 4096) for _ in range(10))
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(torch.nn.functional.layer_norm(x, (4096,)))
        return x

model = ToyModel().to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
pbar = tqdm(range(100))
for i in pbar:
    x = torch.randn((512, 4096), device="cuda")
    y = x + 1
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_postfix(loss=f"{loss.item():.4f}")
```

With Offload Training enabled, VRAM usage drops to 1.4GB:

```python
import torch
from tqdm import tqdm
from diffsynth.core import OffloadTrainingManager

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(4096, 4096) for _ in range(10))
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(torch.nn.functional.layer_norm(x, (4096,)))
        return x

model = ToyModel().to("cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
offload_manager = OffloadTrainingManager(model, target_device="cuda", enable_optimizer_cpu_offload=True)
pbar = tqdm(range(100))
for i in pbar:
    x = torch.randn((512, 4096), device="cuda")
    y = x + 1
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    loss.backward()
    offload_manager.after_backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_postfix(loss=f"{loss.item():.4f}")
```
