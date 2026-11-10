# 模型文件接入 — 目标库自有源码

**适用场景**：`new_series` 接入类型，目标库组件使用自有源码（非 diffusers/transformers 依赖）。

## 核心原则

**优先直接复制代码，除非要解决依赖问题，否则不建议修改代码。**

拿到目标库代码后，第一步是原样复制，之后只做两类修改：

| 允许的修改 | 说明 |
|-----------|------|
| **解决依赖问题** | 替换无法 import 的模块为 DiffSynth 核心模块或内联实现 |
| **替换 Attention / Gradient Checkpointing** | 使用 DiffSynth 的 `attention_forward` 和 `gradient_checkpoint_forward` |

除上述两类外，不重构、不统一风格、不提取公共函数、不重命名变量。这样做的目的：目标库代码已测试完整，改动最小化 = bug 概率最小化；上游更新时 diff 清晰易合并；原始逻辑不动，forward 输出自然一致。

**直接复制目标库代码，仅做最小必要修改。** 数学实现不变，只替换 DiffSynth 已有的核心模块。

**尽量保持 state_dict key 与原始 checkpoint 一致。** 这是减少 converter 工作量的关键：
- `self.{name}` 的命名直接对应 checkpoint 中的 `{name}.weight`
- 不要重命名目标库的层（如把 `self.proj` 改成 `self.qkv_proj`）
- 如果目标库代码的 key 与 checkpoint 基本匹配，保持原样，不要改动

**代码注释原则**：
- ✅ **保留必要的代码解释注释**（如算法逻辑、复杂实现说明）
- ❌ **不要使用大段分隔线注释块**（如 `# ==== ... ====`）

## 文件命名

`diffsynth/models/{series}_{component}.py`

## 最小必要修改清单

1. **import 路径调整**：将目标库的 attention 实现替换为 DiffSynth 核心模块
2. **删除冗余**：删除目标库中与 DiffSynth 核心模块功能重复的实现
3. **其余不动**：RMSNorm、TimestepEmbeddings、AdaLayerNorm、设备管理、rope 等全部保留目标库原始实现

**关键：尽量保持层命名与 checkpoint key 匹配**

假设目标库的 checkpoint 中有 `"transformer_blocks.0.attn.to_q.weight"`，那么：

```python
# ✅ 正确：尽量保持与 checkpoint key 匹配的命名
self.transformer_blocks = nn.ModuleList([
    TransformerBlock(...) for _ in range(num_layers)
])
# 这样 state_dict 会有 "transformer_blocks.0.attn.to_q.weight"

# ❌ 错误：重命名会导致需要复杂的 converter
self.blocks = nn.ModuleList([...])  # checkpoint 中是 "transformer_blocks" 不是 "blocks"
```

如果必须修改结构（如为了使用 DiffSynth 的 attention_forward），优先在 converter 中做 key 映射，而不是修改模型中的命名。个别 key 的差异通过 converter 处理即可。

## 允许复用的 DiffSynth 核心模块（仅限以下两个）

- attention → `from ..core.attention import attention_forward`
- gradient checkpointing → `from ..core.gradient import gradient_checkpoint_forward`

**不复用其他模块**（RMSNorm、TimestepEmbeddings、AdaLayerNorm、设备管理等）：直接保留目标库的实现，复制到模型文件中。

## 详细操作指南

根据组件类型，查看详细替换步骤：

- **Attention 替换**（DiT / 噪声预测模型）→ [attention-replacement.md](./attention-replacement.md)
  - QKV pattern 选择、GQA/MQA 处理、custom mask 处理
- **Gradient Checkpointing 集成**（DiT / 噪声预测模型）→ [gradient-checkpointing.md](./gradient-checkpointing.md)
  - 删除目标库原有实现、Per-Block 包装模式（唯一模式）

## 不要做的

- **不要修改已复制的代码逻辑**——目标库代码已测试完整，任何非必要修改都可能引入 bug
- 不要重构代码结构
- 不要重命名变量或函数（特别是 `self.{name}`，这会改变 state_dict key）
- 不要统一代码风格
- 不要将通用模块提取到 general_modules.py
- 不要在 `__init__` 中添加 `self.gradient_checkpointing` 状态

## 必须保证

- `forward` 的输入输出签名与目标库**完全一致**（这是后续一致性测试的前提）
- 新增的 `use_gradient_checkpointing` 和 `use_gradient_checkpointing_offload` 参数必须有默认值 `False`

## 参考模板

见 [model-file-template.md](./model-file-template.md)。
