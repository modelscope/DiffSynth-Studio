# 模型文件模板

## 新建模型文件

文件位置：`diffsynth/models/{series}_{component}.py`

**接入原则：直接复制目标库代码，仅做最小必要修改。**

**代码注释原则**：
- ✅ **保留必要的代码解释注释**（如算法逻辑、复杂实现说明）
- ❌ **不要使用大段分隔线注释块**（如 `# ==== ... ====`）

```python
import torch
import torch.nn as nn
# 仅复用以下两个核心模块：
from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward
# 其他模块（RMSNorm、TimestepEmbeddings 等）直接保留目标库的实现，复制到本文件中

class {ModelClass}(nn.Module):
    """{Model description}."""

    # 仅 DiT 需要：重复 transformer block 的类名，供 torch.compile 做 regional 编译
    _repeated_blocks = ["{RepeatedBlockClass}"]

    def __init__(self, ...):
        """Initialize the model.
        
        Args:
            ...
        """
        super().__init__()
        # 从目标库复制 __init__ 逻辑
        ...
    
    def forward(self, ...):
        """Forward pass.
        
        Args:
            ...
        Returns:
            ...
        """
        # 从目标库复制 forward 逻辑
        # 确保签名与目标库完全一致
        ...
```

**最小必要修改清单**：
1. **import 路径调整**：将目标库的 attention 实现替换为 `from ..core.attention import attention_forward`
2. **删除冗余**：如果目标库文件中包含了与 attention_forward 相同的 attention 实现，删除它
3. **gradient checkpointing**：如果目标库有 gradient checkpointing 逻辑，替换为 `from ..core.gradient import gradient_checkpoint_forward`
4. **其余不动**：RMSNorm、TimestepEmbeddings、AdaLayerNorm、设备管理、rope 等全部保留目标库原始实现，复制到本文件中

**关键：尽量保持与 checkpoint key 匹配的命名**

`self.{name}` 的命名直接决定 state_dict 中的 `{name}.weight`。尽量保持与原始 checkpoint 一致，减少 converter 工作量：
- 目标库使用 `self.transformer_blocks` → DiffSynth 也使用 `self.transformer_blocks`
- 不要为了风格统一而重命名为 `self.blocks`
- 个别不匹配通过 converter 处理，大量不匹配考虑修改模型命名

**不要做的**：
- 不要重构代码结构
- 不要重命名变量或函数（特别是 `self.{name}`，这会改变 state_dict key）
- 不要统一代码风格
- 不要将通用模块提取到 general_modules.py

## 相关参考

根据不同接入方式，查看详细规则：
- 目标库自有源码 → [model-file-own-source.md](./model-file-own-source.md)
- diffusers 重构 → [model-file-diffusers-refactor.md](./model-file-diffusers-refactor.md)
- transformers 包装 → [model-file-transformers-wrapper.md](./model-file-transformers-wrapper.md)
- version_upgrade 结构变化 → [model-file-version-upgrade.md](./model-file-version-upgrade.md)
- 完全复用 → [model-file-reuse.md](./model-file-reuse.md)
