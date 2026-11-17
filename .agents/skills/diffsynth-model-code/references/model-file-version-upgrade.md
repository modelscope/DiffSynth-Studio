# 模型文件接入 — version_upgrade 结构变化

**适用场景**：`version_upgrade` 接入类型，新版本模型结构有变化（如新增 gated_attention、cross_attention_adaln 等模块），或已有模型参数发生变化。

## 核心原则

**前提：不改变已有模型的 forward 逻辑和输出。** 任何修改都必须向后兼容。

**决策流程**：
```
新 checkpoint 的 __init__ / forward 签名是否有变化？
├── 无变化 → 走【完全复用】方案，详见 model-file-reuse.md
└── 有变化 → 继续判断
    ├── 只有参数值变化（如 num_layers 不同） → 通过 extra_kwargs 处理
    └── 有结构变化（新增/删除模块） → 按本文档处理
```

## 修改方式

在已有模型文件中添加新参数和对应的功能模块，通过条件分支控制：

```python
def __init__(self, ..., apply_gated_attention=False, cross_attention_adaln=False, ...):
    super().__init__()
    # 根据 extra_kwargs 决定是否启用新特性
    self.apply_gated_attention = apply_gated_attention
    if apply_gated_attention:
        self.gated_attn = GatedAttention(...)
```

**关键**：新增的参数必须有默认值（`False` 或 `None`），确保已有注册条目在传入旧 extra_kwargs 时行为不变。

## 兼容性要求

| 要求 | 说明 |
|------|------|
| 新增参数必须有默认值 | `False` 或 `None`，确保已有条目行为不变 |
| 不删除已有参数 | 即使新版本不再使用，也保留并标记为 deprecated |
| 不改变已有参数默认值 | 已有的 forward 输出必须保持不变 |
| 不修改已有 forward 逻辑 | 只在条件分支中添加新功能 |
| 修改后需验证 | 用已有注册条目的 extra_kwargs 实例化，验证 forward 输出与修改前一致 |

## 常见结构变化示例

### 示例 1：新增模块

```python
# 新版本有 gated_attention，旧版本没有
class WanModel(nn.Module):
    def __init__(
        self,
        ...,
        apply_gated_attention=False,  # 新增参数，默认 False
    ):
        super().__init__()
        # 原有逻辑
        self.transformer_blocks = nn.ModuleList([...])

        # 新增模块，仅在启用时创建
        if apply_gated_attention:
            self.gated_attn = GatedAttention(...)

    def forward(self, ...):
        # 原有 forward 逻辑不变
        for block in self.transformer_blocks:
            x = block(x, ...)

        # 新增功能，仅在启用时执行
        if hasattr(self, "gated_attn"):
            x = self.gated_attn(x, ...)

        return x
```

### 示例 2：参数值变化

如果只是 `num_layers`、`hidden_size` 等参数值变化，无需修改模型文件，只需在 `model_configs.py` 的 `extra_kwargs` 中传入新值：

```python
# 旧版本：num_layers=30
{"model_hash": "...", "model_name": "...", "model_class": "...", "extra_kwargs": {}}

# 新版本：num_layers=40
{"model_hash": "...", "model_name": "...", "model_class": "...", "extra_kwargs": {"num_layers": 40}}
```

### 示例 3：新增交叉注意力模块

```python
# 新版本引入了 cross_attention_adaln
class WanModel(nn.Module):
    def __init__(
        self,
        ...,
        cross_attention_adaln=False,  # 新增参数
        adaln_hidden_size=None,       # 新增参数
    ):
        super().__init__()
        # 原有模块
        self.transformer_blocks = nn.ModuleList([...])

        # 新增模块
        if cross_attention_adaln:
            self.adaln = AdaLayerNorm(adaln_hidden_size)

    def forward(self, x, timestep, ..., adaln_cond=None):
        # 原有 forward 逻辑
        for block in self.transformer_blocks:
            x = block(x, timestep)

        # 新增逻辑，仅在启用时执行
        if hasattr(self, "adaln") and adaln_cond is not None:
            x = self.adaln(x, adaln_cond)

        return x
```

## 验证步骤

修改完成后，必须执行以下验证：

1. **向后兼容验证**：用已有注册条目的 `extra_kwargs` 实例化模型，验证 forward 输出与修改前一致
2. **新功能验证**：用新注册条目的 `extra_kwargs` 实例化模型，验证新功能正常工作
3. **Forward 一致性测试**：所有版本都必须通过完整的 forward 一致性测试（详见 Step 6）

## 一致性测试影响

version_upgrade 的所有情况都必须执行两种测试：
- **Forward: ✅ 必须** —— 验证修改后的 forward 逻辑正确，不同版本输出与原始模型一致
- **加载: ✅ 必须** —— 验证注册链正确，所有版本的 hash 都能正确匹配

## 相关参考

- 完全复用 → [model-file-reuse.md](./model-file-reuse.md)
- 模型文件模板 → [model-file-template.md](./model-file-template.md)
- 接入流程总览 → [SKILL.md](../SKILL.md)
