# 模型文件接入 — transformers 组件包装

**适用场景**：`new_series` 接入类型，目标库组件依赖 transformers（如 Qwen2.5-VL、CLIP 等 TextEncoder）。

## 核心原则

**DiffSynth 可以依赖 transformers**，但需要编写接入包装类。

## 包装流程

### 1. 创建包装类文件

- 文件命名：`diffsynth/models/{series}_{component}.py`

### 2. 实现模式

```python
import torch
from typing import Optional, Union

class {SeriesName}{ComponentType}(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Lazy import transformers（在 __init__ 中而非文件顶部）
        from transformers import {TransformerConfig}, {TransformerModel}

        # 手动构建 config dict（或从预训练配置加载）
        config = {TransformerConfig}(**{
            "hidden_size": ...,
            "num_attention_heads": ...,
            # ... 完整配置
        })

        # 包装 transformers 模型
        self.model = {TransformerModel}(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        # ... 其他参数
    ):
        # 调用 transformers 模型的 forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # ...
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states  # 返回 DiffSynth 需要的格式
```

### 3. Config 验证（关键步骤）

**在做一致性测试之前，必须先从目标库的对应模型读取原始 config，确认包装类中手动构建的 config dict 与原始配置一致。**

为什么必须验证 config：
- transformers 模型的 config dict 参数众多（常达 30+ 个字段）
- 手动构建容易遗漏或写错某个参数
- config 参数错误会导致模型结构或行为偏差，forward 输出不一致

**验证步骤**：

```python
"""
步骤 1: 从目标库加载原始 transformers 模型，读取其 config
步骤 2: 对比包装类中手动构建的 config dict
步骤 3: 确认一致后再进行 forward 一致性测试
"""
import torch
from transformers import {TransformerConfig}, {TransformerModel}

# === 步骤 1: 加载原始 transformers 模型及其 config ===
# 使用目标库的方式加载（或直接从预训练模型加载）
orig_model = {TransformerModel}.from_pretrained("{target_model_path}")
orig_config = orig_model.config

# === 步骤 2: 对比 config ===
from diffsynth.models.{series}_{component} import {SeriesName}{ComponentType}
wrapper = {SeriesName}{ComponentType}()

# 逐字段对比 config
print("=== Config 对比验证 ===")
mismatch_fields = []
for key in orig_config.to_dict():
    orig_val = getattr(orig_config, key, None)
    wrapper_val = getattr(wrapper.config, key, None)
    if orig_val != wrapper_val:
        mismatch_fields.append((key, orig_val, wrapper_val))
        print(f"  ❌ {key}: 原始={orig_val}, 包装={wrapper_val}")

if mismatch_fields:
    print(f"\n⚠️ 发现 {len(mismatch_fields)} 个 config 字段不一致，请先修正包装类！")
    raise ValueError(f"Config mismatch: {[f[0] for f in mismatch_fields]}")
else:
    print("✅ Config 对比通过，所有字段与原始模型一致")

# === 步骤 3: config 通过后才进行 forward 一致性测试 ===
# 后续按照标准的一致性测试流程验证 forward 输出
```

**注意**：
- 如果目标库使用 `from_pretrained` 加载，config 已包含所有默认值，直接对比即可
- 如果手动构建的 config 有遗漏字段，transformers 会使用默认值，这可能与目标库的实际参数不同
- 此步骤必须在 forward 一致性测试 **之前** 执行，config 不通过不允许进行测试

### 4. 参考实现

- `diffsynth/models/qwen_image_text_encoder.py` — Qwen2.5-VL 的 transformers 包装
- `diffsynth/models/step1x_text_encoder.py` — 基于 QwenImageTextEncoder 的进一步包装

### 5. 注意事项

- transformers 版本要求必须在蓝图报告中明确
- config dict 必须完整，不能遗漏关键参数
- `forward` 方法应返回 DiffSynth Pipeline 需要的格式（通常是 hidden_states 或 last_hidden_state）
- 使用 lazy import（在 `__init__` 中 import transformers），避免文件顶部 import

## 参考模板

见 [model-file-template.md](./model-file-template.md)。
