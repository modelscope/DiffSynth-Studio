# 模型文件接入 — diffusers 组件重构

**适用场景**：`new_series` 接入类型，目标库组件依赖 diffusers（如 AutoencoderKL、UNet2DConditionModel 等）。

## 核心原则

**DiffSynth 模型代码不可依赖 diffusers 包。** 所有 diffusers 组件必须重构为独立实现，但这并不意味着不能安装 diffusers 包（安装到 conda 环境是允许的）。

**尽量保持 state_dict key 与原始 diffusers checkpoint 一致。** 重构时不要改变层的命名结构：
- 保持 diffusers 原始的属性名（如 `self.conv_in`、`self.down_blocks`）
- 不要重命名为 DiffSynth 风格的命名
- 如果必须修改（如继承关系变化），在 converter 中处理 key 映射

**代码注释原则**：
- ✅ **保留必要的代码解释注释**（如算法逻辑、复杂实现说明）
- ❌ **不要使用大段分隔线注释块**（如 `# ==== ... ====`）
- ✅ **保留原始许可证头**（如果是 Apache 2.0/MIT 等开源许可证要求）

## 重构流程

### 1. 定位 diffusers 源码

- 在 diffusers 仓库中找到对应组件（如 `src/diffusers/models/autoencoders/autoencoder_kl.py`）
- 识别所有依赖文件（如 `activations.py`、`attention.py`、`resnet.py` 等）

### 2. 创建 DiffSynth 模型文件

- 文件命名：`diffsynth/models/{series}_{component}.py`
- **不要添加代码来源相关的注释**（如 "copied from diffusers"、"original code" 等）

### 3. 重构要点

**从 diffusers 源码复制代码时，保持原样，只处理以下必要的适配点：**

- 移除 `diffusers.models.modeling_utils.ModelMixin` 继承，改为 `torch.nn.Module`
- 移除 `diffusers.configuration_utils.ConfigMixin` 继承
- 将 `diffusers` 内部 import 替换为 DiffSynth 核心模块或直接内联实现
- 保留所有数学实现不变（attention、normalization、activation 等）
- 确保 `forward` 签名符合 DiffSynth 的需求
- **尽量保持属性命名与 diffusers 原始代码一致**（如 `self.conv_in`、`self.down_blocks`），避免不必要的 key 映射

除上述适配点外，不要修改 diffusers 的代码逻辑、不重构结构、不统一风格。

**示例**：
```python
# diffusers 原始代码:
self.conv_in = nn.Conv2d(...)
self.down_blocks = nn.ModuleList([...])

# ✅ 正确：保持命名一致，checkpoint 中的 "conv_in.weight" 直接匹配
self.conv_in = nn.Conv2d(...)
self.down_blocks = nn.ModuleList([...])

# ❌ 错误：重命名会导致需要 converter
self.input_conv = nn.Conv2d(...)  # checkpoint 中是 "conv_in"
```

**Attention 和 Gradient Checkpointing**：
- 如果重构的是 **DiT / 噪声预测模型**（如 UNet2DConditionModel），按照以下独立指南处理：
  - Attention 替换 → [attention-replacement.md](./attention-replacement.md)
  - Gradient Checkpointing → [gradient-checkpointing.md](./gradient-checkpointing.md)
- 如果重构的是 **VAE / 编解码器**（如 AutoencoderKL），通常不需要替换 attention（VAE 内部的 attention 规模小，保持 diffusers 原始实现即可）

### 4. 不要做的

- **不要重命名属性**（如把 `self.conv_in` 改为 `self.input_conv`）——这会改变 state_dict key，导致需要写 converter
- 不要重构代码结构（保持 diffusers 的模块组织方式）
- 不要统一代码风格
- 不要删除原始许可证头

### 5. 参考实现

- `diffsynth/models/flux2_vae.py` — 从 diffusers `AutoencoderKL` 重构，保留 Apache 2.0 许可证头
- `diffsynth/models/flux_vae.py` — 从 diffusers VAE 重构

## 示例

```python
import torch
import torch.nn as nn

# 内联 diffusers 的 activation 定义（而非 import diffusers.models.activations）
ACT2CLS = {"swish": nn.SiLU, "silu": nn.SiLU, "mish": nn.Mish, "gelu": nn.GELU, "relu": nn.ReLU}

class {SeriesName}VAE(nn.Module):  # 继承 nn.Module 而非 ModelMixin
    def __init__(self, ...):
        super().__init__()  # 不使用 super().__init__(config)
        # 复制 diffusers 的 __init__ 逻辑
        ...

    def forward(self, x, ...):
        # 复制 diffusers 的 forward 逻辑
        ...
```

## 参考模板

见 [model-file-template.md](./model-file-template.md)。
