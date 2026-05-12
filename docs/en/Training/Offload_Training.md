# Offload Training

This document introduces the Offload Training feature in DiffSynth-Studio, which significantly reduces GPU memory usage during training by moving model weights layer-by-layer between CPU and GPU.

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

### Parameter Classification

Different offload strategies are applied depending on whether parameters are trainable:

| Parameter Type | Offloader Class | Behavior |
|---------------|----------------|----------|
| Non-trainable (`requires_grad=False`) | `StaticParamOffloader` | Copies weights to pre-allocated pinned memory at init, maintaining a permanent CPU copy; onload copies from CPU to GPU, offload reassigns `param.data` back to the CPU copy (no PCIe transfer back) |
| Trainable + `optimize_on_cpu=True` | `TrainableParamOffloader` | Weights change during training, so no static copy is kept; onload/offload via `param.data.to(device)` with actual data transfer; also moves `param.grad` to CPU after backward |
| Trainable + `optimize_on_cpu=False` | `AlwaysOnGPUParamOffloader` | Moves parameters to GPU at init and never offloads; suitable for LoRA training (small number of trainable params) |

### Pinned Memory Pool

`StaticParamOffloader` needs to allocate a pinned memory copy on CPU for each non-trainable parameter (pinned memory enables asynchronous non-blocking CPU→GPU transfers, much faster than regular pageable memory).

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

### Module Discovery Strategy (Experimental)

`OffloadTrainingManager` controls module granularity via the `param_size_threshold` parameter:

- `param_size_threshold=None` (default): offloads every leaf module (`nn.Linear`, `nn.LayerNorm`, etc.)
- `param_size_threshold=N` (MB): modules with total parameters exceeding the threshold are recursively split into children; modules below the threshold are offloaded as a whole

Additionally, "orphan parameters" and buffers not managed by any unit are automatically collected and hooked.

### Training Loop Integration

Execution flow in `runner.py`:

```python
# When cpu_offload=True:
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
| `--cpu_offload` | False | Enable layer-wise offload training |
| `--optimize_on_cpu` | False | Used with `--cpu_offload`; moves trainable params and optimizer to CPU |
| `--param_size_threshold` | None | Experimental (unit: MB); modules above this threshold are recursively split |

### Parameter Combinations

| Scenario | `--cpu_offload` | `--optimize_on_cpu` | Effect |
|----------|:---------------:|:-------------------:|--------|
| Default training | ❌ | ❌ | All weights and optimizer on GPU |
| Weight offload only | ✅ | ❌ | Non-trainable params offloaded layer-by-layer; trainable params (e.g., LoRA) and optimizer stay on GPU |
| Full offload | ✅ | ✅ | All params offloaded layer-by-layer; gradients and optimizer run on CPU |

**Recommendations**:
- LoRA training: use `--cpu_offload` (trainable params are small, fine to keep on GPU)
- Full fine-tuning: use `--cpu_offload --optimize_on_cpu` (optimizer states are large, need to offload to CPU)

### Example

Simply add `--cpu_offload` to your existing training command. Example with Qwen-Image LoRA training:

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
  --cpu_offload
```

For full offload (optimizer also on CPU), add `--optimize_on_cpu`:

```bash
  --cpu_offload \
  --optimize_on_cpu
```

### Compatibility

| Feature | Compatible | Notes |
|---------|:----------:|-------|
| LoRA Training | ✅ | Trainable params stay on GPU via `AlwaysOnGPUParamOffloader` |
| Gradient Checkpointing | ✅ | `_in_recompute` mechanism handles recomputation |
| Accelerate DDP | ✅ | In cpu_offload mode, model is not prepared; only optimizer/dataloader are |
| Split Training | ✅ | `launch_data_process_task` also supports `--cpu_offload` |
| DeepSpeed | ❌ | ZeRO's parameter gathering conflicts with hooks |

### Notes

- With `--cpu_offload` enabled, the model never calls `model.to(device)`; weights are managed entirely by hooks
- Training speed decreases due to CPU↔GPU transfers (typically 2-10x slower); larger models see greater slowdown; suitable for memory-constrained scenarios
- Recommended to use with `--use_gradient_checkpointing` to further reduce activation memory
