# Scheduler 接入指南

**适用场景**：Pipeline 接入时，目标模型使用与已有系列不同的 timestep 调度策略。

## 决策树：Scheduler 应该如何接入

```
目标模型的调度策略是什么？
│
├── Flow Matching（sigma-based，timesteps = sigmas * 1000）
│   │
│   ├── 调度公式与已有 template 相同？（完整清单见下方 §已注册 template）
│   │   ├── 是 → 方案 A：复用已有 template
│   │   └── 否 → 方案 B：在 FlowMatchScheduler 中新增 set_timesteps_{series} 方法
│   │
│   └── 是否有模型特异参数或特殊方案（如 distilled、turbo）？
│       ├── 有 → 通过 special_case 参数区分方案，在 set_timesteps 内部返回固定 sigmas
│       └── 无 → 只传 num_inference_steps + denoising_strength
│
└── 非 Flow Matching（如 DDPM、DDIM、Euler 等）
    └── 在 FlowMatchScheduler 中新增方法，按该采样器逻辑实现 set_timesteps_{series}
```

**调度策略默认统一在 `FlowMatchScheduler` 中实现**，通过 template 分发，不新建 Scheduler 类。仅当调度逻辑无法用 `set_timesteps_*` 表达时才继承 `FlowMatchScheduler`（上游仅有一例：`HiDreamO1FlashScheduler`）。

### 已注册 template

`FlowMatchScheduler.__init__` 的 `template` 参数当前接受以下值（`diffsynth/diffusion/flow_match.py`）：

`FLUX.1`、`Wan`、`Qwen-Image`、`FLUX.2`、`Z-Image`、`LTX-2`、`Qwen-Image-Lightning`、`ERNIE-Image`、`ACE-Step`、`HiDream-O1-Image`、`Ideogram4`、`Krea-2`、`Boogu`、`MiniMax-H3`、`MiniMax-Music3`、`LingBot-Video`

**接入前先读一遍 `flow_match.py` 的注册字典**，确认是否已有可复用的 template——该列表随上游新模型接入持续增长。传入未注册的名称会静默回退到 `set_timesteps_flux`。

---

## 方案 A：复用已有 FlowMatchScheduler template

如果目标模型的 timestep 调度公式与已有 template 完全相同：

```python
# __init__ 中
self.scheduler = FlowMatchScheduler("Wan")  # 传入已有 template 名称
```

`__call__` 中无需额外参数：

```python
self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)
```

---

## 方案 B：在 FlowMatchScheduler 中新增 template 方法

如果目标模型使用 flow matching 但调度公式与已有 template 不同，或有多个调度方案。

### 步骤 1：在 `flow_match.py` 中添加新的 `set_timesteps_{series}` 方法

```python
# diffsynth/diffusion/flow_match.py

@staticmethod
def set_timesteps_{series}(num_inference_steps=100, denoising_strength=1.0, special_case=None, ...其他特异参数...):
    """{Series} 的 timestep 调度策略。
    
    special_case 用于区分同一模型的不同调度方案（如 distilled、turbo）。
    这些方案通常有固定的 step 数和固定的 sigmas，不依赖 num_inference_steps。
    """
    num_train_timesteps = 1000
    
    # 特殊方案：固定 sigmas（distilled / turbo 等）
    if special_case == "distilled":
        sigmas = torch.Tensor([...])  # 固定的 sigmas
    elif special_case == "turbo":
        sigmas = torch.Tensor([...])  # 固定的 sigmas
    else:
        # 默认方案：动态计算 sigmas
        # ... 调度公式实现 ...
        sigmas = torch.linspace(...)
    
    timesteps = sigmas * num_train_timesteps  # 必须乘以 1000，保持训练兼容性
    return sigmas, timesteps
```

**重要**：`timesteps` 必须乘以 `num_train_timesteps`（1000），这是为了与训练脚本中的 timestep 权重计算保持一致。

### 步骤 2：在 `__init__` 的 `set_timesteps_fn` 字典中注册

```python
def __init__(self, template: Literal["FLUX.1", "Wan", ..., "{Series}"] = "FLUX.1"):
    self.set_timesteps_fn = {
        "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
        # ... 已有 entries ...
        "{Series}": FlowMatchScheduler.set_timesteps_{series},  # 新增
    }.get(template, FlowMatchScheduler.set_timesteps_flux)
```

### 步骤 3：Pipeline 中传入 template 名称和模型特异参数

```python
# __init__
self.scheduler = FlowMatchScheduler("{Series}")

# __call__
self.scheduler.set_timesteps(
    num_inference_steps,
    denoising_strength=denoising_strength,
    special_case=special_case,          # 特殊方案（distilled / turbo 等）
    dynamic_shift_len=dynamic_shift_len, # 其他模型特异参数
)
```

### 步骤 4：在 model_fn 中处理 timestep 范围（如需要）

如果目标模型的 DiT 期望 timestep 范围为 `[0, 1]` 而不是 `[0, 1000]`，需要在 `model_fn` 中除以 1000：

```python
def {series}_model_fn(dit, timestep, latents, ...):
    timestep_tensor = (timestep / 1000).expand(batch_size).to(dtype=latents.dtype, device=latents.device)
    output = dit(hidden_states=latents, timestep=timestep_tensor, ...)
    return output
```

**判断是否需要除以 1000**：检查目标库原始代码中传入 DiT 的 timestep 范围。如果原始库使用 `t_schedule` 值在 `[0, 1]` 范围内，则需要；如果原始库使用 `t * num_train_timesteps` 或值在 `[0, 1000]` 范围内，则不需要。

### LTX-2 完整示例

LTX-2 的 `set_timesteps_ltx2` 展示了如何处理同一模型的多个调度方案：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dynamic_shift_len` | int | 4096 | 序列长度，用于计算 sigma_shift |
| `terminal` | float | 0.1 | terminal shift，控制最终 sigma 的衰减 |
| `special_case` | str | None | 调度方案标记：`None`（默认）/ `"stage2"` / `"distilled_stage1"` |

```python
# flow_match.py 中的实现
@staticmethod
def set_timesteps_ltx2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None):
    num_train_timesteps = 1000
    
    # 特殊方案：固定 sigmas，不依赖 num_inference_steps
    if special_case == "stage2":
        sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
    elif special_case == "distilled_stage1":
        sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
    else:
        # 默认方案：动态计算
        dynamic_shift_len = dynamic_shift_len or 4096
        sigma_shift = FlowMatchScheduler._calculate_shift_qwen_image(
            image_seq_len=dynamic_shift_len, base_seq_len=1024, max_seq_len=4096,
            base_shift=0.95, max_shift=2.05,
        )
        sigma_min, sigma_max = 0.0, 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1 - terminal)
        sigmas = 1.0 - (one_minus_z / scale_factor)
    
    timesteps = sigmas * num_train_timesteps
    return sigmas, timesteps
```

Pipeline 中的使用方式：

```python
class LTX2Pipeline(BasePipeline):
    def __init__(self, ...):
        self.scheduler = FlowMatchScheduler("LTX-2")

    @torch.no_grad()
    def __call__(
        self,
        # ... 常规参数 ...
        num_inference_steps: int = 30,
        # Scheduler 特异参数
        dynamic_shift_len: int = 4096,
        terminal: float = 0.1,
        special_case: str = None,  # 特殊调度方案
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=dynamic_shift_len,
            terminal=terminal,
            special_case=special_case,
        )
```

---

## 特殊调度方案设计原则

当目标模型有 distilled、turbo 等变体时，遵循以下规则：

1. **统一在 `set_timesteps_{series}` 中处理**，不创建额外的 scheduler 类或方法
2. **使用 `special_case` 参数区分方案**，默认值为 `None`（标准方案）
3. **特殊方案通常有固定的 sigmas**，忽略 `num_inference_steps` 参数
4. **在蓝图中明确列出所有方案**：每个方案的名称、固定 sigmas、对应 step 数
5. **Pipeline `__call__` 中暴露 `special_case` 参数**，让用户可以选择方案

### 蓝图中的特殊方案描述格式

```markdown
| 方案名称 | special_case 值 | 固定 sigmas | Step 数 | 说明 |
|---------|----------------|------------|--------|------|
| 默认 | `None` | 动态计算 | 用户指定 | 标准推理 |
| Distilled Stage 2 | `"stage2"` | `[0.909, 0.725, 0.422]` | 3 | 蒸馏模型 |
| Distilled Stage 1 | `"distilled_stage1"` | `[1.0, 0.994, ...]` | 8 | 蒸馏模型 |
```

---

## Scheduler 参数透传规则

| 场景 | `__call__` 中是否暴露 | `set_timesteps` 中是否透传 |
|------|----------------------|--------------------------|
| `num_inference_steps` | ✅ 必须 | ✅ 必须 |
| `denoising_strength` | ✅ 必须（如果有 img2x 功能） | ✅ 必须 |
| 模型特异参数（如 `dynamic_shift_len`） | ✅ 必须，有默认值 | ✅ 必须 |
| 特殊方案参数（`special_case`） | ✅ 必须，默认 `None` | ✅ 必须 |
| 训练阶段参数 | ❌ 推理时不暴露 | 推理时传 `None`，训练时由训练脚本传入 |

**默认值设置原则**：模型特异参数的默认值应设为该系列最常见/基准配置的值。

## 蓝图中的 Scheduler 信息要求

蓝图报告必须在「Pipeline 功能规划」章节中明确：

1. **调度策略类型**：Flow Matching / DDPM / DDIM / 其他
2. **如果是 Flow Matching**：
   - 是否与已有 template 的调度公式相同？
   - 如果不同，列出完整的 sigma 计算公式
   - 列出所有模型特异参数（名称、类型、默认值、用途）
3. **特殊调度方案**（如有）：每个方案的名称、`special_case` 值、固定 sigmas、step 数
4. **目标库原始 scheduler 类名和文件路径**
5. **训练时是否有特殊的 timestep 权重策略**（如 LTX-2 的 `set_training_weight`）
