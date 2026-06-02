# Offload Training

本文档介绍 DiffSynth-Studio 中的 Offload Training 功能，通过将模型权重逐层在 CPU 和 GPU 间搬运，大幅降低训练时的 GPU 显存占用。

> **注意**：当前 Offload Training 仅支持单卡训练，暂不兼容多卡（DDP）场景。

## 什么是 Offload Training

训练大规模模型（如 Qwen-Image 60 层、Wan2.1-14B 40 层）时，所有层的权重需同时驻留 GPU，仅权重就占用数十 GB 显存。Offload Training 的核心思想是：**任一时刻只将当前正在计算的模块权重加载到 GPU，计算完毕后立即卸载回 CPU**，从而将显存占用从 O(N × 每层参数量) 降低到 O(1 × 每层参数量)。

该功能基于 PyTorch 的 Module Hook 机制实现，不需要修改任何模型代码。

## 方案原理

### 核心机制

`OffloadTrainingManager` 会扫描模型，为每个需要管理的模块注册 4 个 Hook：

```
forward_pre_hook   → 将模块权重从 CPU 加载到 GPU (onload)
module.forward()   → 正常前向计算
forward_hook       → 将模块权重从 GPU 卸载回 CPU (offload)

backward_pre_hook  → 将模块权重从 CPU 重新加载到 GPU (onload)
module.backward()  → 计算梯度
backward_hook      → 将模块权重卸载回 CPU (offload)
```

### 参数与 Buffer 分类管理

根据参数是否可训练以及 buffer 类型，采用不同的 offload 策略：

| 类型 | Offloader 类 | 行为 |
|---------|-------------|------|
| 非可训练参数 (`requires_grad=False`) | `StaticParamOffloader` | 初始化时将权重拷贝到预分配的 pinned memory 中保持永久 CPU 副本，并将 `param.data` 替换为 GPU 端空 placeholder（释放 GPU 显存）；onload 时从 CPU 副本异步拷贝到 GPU，offload 时将 `param.data` 重新指向 placeholder（无需 PCIe 回传） |
| 可训练参数 + `enable_optimizer_cpu_offload=True` | `TrainableParamOffloader` | 权重在训练中会变化，无法保持静态副本；onload/offload 通过 `param.data.to(device)` 实际搬运；backward 后将 `param.grad` 也移到 CPU |
| 可训练参数 + `enable_optimizer_cpu_offload=False` | `AlwaysOnGPUParamOffloader` | 初始化时直接将参数移到 GPU，之后不再搬运；适用于 LoRA 训练（可训练参数量小） |
| 模块 Buffer（如 BatchNorm 的 `running_mean`/`running_var`） | `BufferOffloader` | 与 `StaticParamOffloader` 类似：初始化时将 buffer 拷贝到 pinned memory；onload 时从 CPU 副本异步拷贝到 GPU，offload 时将 `module._buffers[name]` 重新指向 CPU 副本 |

### Pinned Memory Pool

`StaticParamOffloader` 和 `BufferOffloader` 在初始化时需要为每个非可训练参数/buffer 分配一份 CPU 端的 pinned memory 副本（pinned memory 可以实现 CPU→GPU 的异步非阻塞传输，比普通 pageable memory 快得多）。

**问题**：PyTorch 的 `pin_memory()` 底层通过 `CachingHostAllocator` 分配内存，该分配器会将每次申请的大小向上取整到 2 的整数次方。例如一个 17MB 的 tensor 会实际分配 32MB。大模型有数千个参数 tensor，每个都独立 `pin_memory()` 会导致大量内存浪费（实测可能膨胀 50%~100%）。

**解决方案**：`PinnedArenaPool` 预先一次性分配少量大块 pinned memory（即 arena，一块预分配的大内存区域，所有小对象从中切分），然后用 bump-pointer 方式从大块中紧凑地切分出每个 tensor 需要的空间，避免逐 tensor 取整带来的浪费：

- `from_model()` 扫描模型所有非可训练参数和 buffer，计算总大小
- 将总大小分解为若干 power-of-two 大小的 chunk（每个 chunk 是一个 `PinnedBuffer`）
- 分配时顺序查找有剩余空间的 chunk，bump-pointer 前进即完成分配（仅 64 字节对齐，无取整浪费）
- 空间不足时自动 grow 新 chunk
- 异常情况下回退到逐 tensor 的 `pin_memory()`

### Gradient Checkpointing 兼容

Gradient Checkpointing 在 backward 时会重新执行 forward（重算激活），这会再次触发 `forward_hook`。方案通过 `_in_recompute` 集合解决：

- 首次 forward：正常 offload，并将模块加入 `_in_recompute`
- 重算 forward（backward 期间）：检测到模块在 `_in_recompute` 中，跳过 offload，保持权重在 GPU 供 backward 使用
- `after_backward()` 调用时：清空 `_in_recompute`，为下一个 step 做准备

### Hook 注册粒度

`OffloadTrainingManager` 默认以每个叶子模块（`nn.Linear`、`nn.LayerNorm` 等）为单位注册 hook，即每个叶子模块独立进行 onload/offload。此外，未被任何叶子模块管理到的「孤儿参数」和「孤儿 buffer」也会被自动收集并注册 hook。

**实验性功能**：通过 `cpu_offload_split_threshold`（单位 MB）可以调整 hook 的注册粒度。设置后，参数总量超过阈值的模块会被递归拆分为子模块，未超过阈值的模块则作为整体注册 hook。该功能当前版本可能无法兼容所有模型结构，默认不启用。

### 训练流程集成

在 `runner.py` 中的执行流程：

```python
# enable_model_cpu_offload=True 时：
# 1. 模型不调用 model.to(device)，保持在 CPU
# 2. 只 prepare optimizer、dataloader、scheduler（不 prepare model）
# 3. 创建 OffloadTrainingManager，自动为模型注册 hook

# 训练循环：
loss = model(data)
accelerator.backward(loss)
offload_manager.after_backward()  # 清空 recompute 标记 + 梯度移到 CPU
optimizer.step()
optimizer.zero_grad()
```

## 如何使用

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_model_cpu_offload` | False | 启用逐层 offload 训练 |
| `--enable_optimizer_cpu_offload` | False | 配合 `--enable_model_cpu_offload`，将可训练参数和 optimizer 也放在 CPU |
| `--cpu_offload_split_threshold` | None | 实验性参数（单位 MB），超过此阈值的模块会被递归拆分 |

### 参数组合

| 场景 | `--enable_model_cpu_offload` | `--enable_optimizer_cpu_offload` | 效果 |
|------|:---------------:|:-------------------:|------|
| 默认训练 | ❌ | ❌ | 所有权重和 optimizer 在 GPU |
| 仅 offload 非可训练参数 | ✅ | ❌ | 非可训练参数逐层 offload，可训练参数和 optimizer 留在 GPU |
| offload 所有参数 | ✅ | ✅ | 所有参数逐层 offload，梯度和 optimizer 在 CPU 执行 |

### 使用示例

在现有训练命令中添加 `--enable_model_cpu_offload` 即可启用，以 Qwen-Image LoRA 训练为例：

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

如需完整 offload（optimizer 也在 CPU），添加 `--enable_optimizer_cpu_offload`：

```bash
  --enable_model_cpu_offload \
  --enable_optimizer_cpu_offload
```

### 兼容性

| 特性 | 兼容 | 说明 |
|------|:----:|------|
| Gradient Checkpointing | ✅ | `_in_recompute` 机制兼容 |
| Accelerate DDP（多卡训练） | ⚠️ | enable_model_cpu_offload 模式下不会对 model 进行 DDP 包装（不调用 `accelerator.prepare(model)`），因此**不会执行梯度 allreduce**。无法保证与多卡训练的兼容性，各卡独立计算梯度而无同步 |
| 拆分训练 | ✅ | `launch_data_process_task` 同样支持 `--enable_model_cpu_offload` |
| DeepSpeed | ❌ | ZeRO 的参数聚集机制与 hook 冲突 |

### 注意事项

- 开启 `--enable_model_cpu_offload` 后，模型不会调用 `model.to(device)`，权重始终由 hook 管理
- 训练速度会因 CPU↔GPU 传输而下降（典型约 2-10 倍），模型越大，速度下降越多，适合显存受限场景
- 建议配合 `--use_gradient_checkpointing` 使用以进一步降低激活值的显存占用
- `--enable_optimizer_cpu_offload` 仅支持梯度累积步数为 1（`--gradient_accumulation_steps 1`）

## 在其他代码库中集成 Offload Training 模块

Offload Training 模块是相对独立的，因此开发者可以将其集成到其他代码库中，以下是一个代码样例，显存占用 4G。

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

启用 Offload Training，显存占用降低到 1.4G：

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
