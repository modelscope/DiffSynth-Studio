# Gradient Checkpointing 集成指南

**适用对象**：DiT 及其他噪声预测模型中的 transformer blocks 循环。

## 核心原则

**完全替换目标库的 gradient checkpointing 实现。** 不在 `__init__` 中保留任何 checkpoint 状态，仅在 `forward` 方法中通过参数控制。

DiffSynth 的 `gradient_checkpoint_forward` 内部已处理所有后端细节（DeepSpeed、CPU offload、use_reentrant），接入时只需按统一模式调用。

## API 签名

```python
from ..core.gradient import gradient_checkpoint_forward

def gradient_checkpoint_forward(
    model,                              # 可调用的 forward 函数或 nn.Module
    use_gradient_checkpointing: bool,   # 是否启用
    use_gradient_checkpointing_offload: bool,  # 是否 offload 到 CPU
    *args, **kwargs                     # 传给 model forward 的参数
) -> torch.Tensor:
```

**内部决策链**：
1. DeepSpeed 已配置 → `deepspeed.checkpointing.checkpoint`
2. offload 开启 → `torch.utils.checkpoint` + `save_on_cpu`
3. 普通 checkpoint → `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
4. 不启用 → 直接 `model(*args, **kwargs)`

## 集成步骤

### 步骤 1：添加 import

```python
from ..core.gradient import gradient_checkpoint_forward
```

### 步骤 2：删除目标库原有的 checkpointing 实现

删除以下内容：
- `__init__` 中的 `self.gradient_checkpointing = False` 或类似状态变量
- `self._gradient_checkpointing_func` 或类似的自定义 checkpoint 函数
- 目标库自行实现的 `torch.utils.checkpoint.checkpoint` 调用
- 任何与 DeepSpeed checkpointing 相关的逻辑

### 步骤 3：在 forward 方法中添加 checkpointing 参数

```python
def forward(
    self,
    # ... 原有参数 ...
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
```

### 步骤 4：将 block 循环替换为 gradient_checkpoint_forward 调用

```python
# 替换前（目标库原始实现）
for block in self.blocks:
    hidden_states = block(hidden_states, **block_kwargs)

# 替换后
for block in self.blocks:
    hidden_states = gradient_checkpoint_forward(
        block,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        hidden_states=hidden_states,
        **block_kwargs,
    )
```

## 集成模式：Per-Block 包装（唯一模式）

在 DiT 的 `forward` 中遍历 blocks 时逐个包装。适用于所有 DiT 模型。

**这是唯一的集成模式。** 不要在 block 层（Layer）内部做委托——gradient checkpointing 的控制权必须集中在 DiT 的 `forward` 方法中，由调用方（Pipeline/训练脚本）统一传入参数。

```python
class {SeriesName}DiT(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 不在 __init__ 中添加 self.gradient_checkpointing
        self.blocks = nn.ModuleList([...])

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        # ... 前置逻辑 ...
        for block in self.blocks:
            hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                # ... 其他 block 需要的参数
            )
        # ... 后置逻辑 ...
```

**所有模型统一使用 Per-Block 包装模式**，无论目标库的 block 是否有 `_forward_impl` + `forward` 双层结构。如果目标库 block 有这种双层结构，保持其 `_forward_impl` 不变，在 DiT 的 `forward` 中对 block 整体做包装即可。

## 注意事项

- `gradient_checkpoint_forward` 内部已处理所有细节，不需要额外逻辑
- 新增的 `use_gradient_checkpointing` 和 `use_gradient_checkpointing_offload` 参数必须有默认值 `False`
- 如果目标库没有 gradient checkpointing 逻辑，直接添加即可，不需要删除任何东西
- VAE 类模型通常不需要集成 gradient checkpointing（除非蓝图报告明确要求）

## 相关参考

- Attention 替换 → [attention-replacement.md](./attention-replacement.md)
- 自有源码接入 → [model-file-own-source.md](./model-file-own-source.md)
