# 模型文件接入 — 完全复用

**适用场景**：新模型的 `__init__` 和 `forward` 与已有组件完全相同，无需任何代码变更。

## 判断标准

- 新组件的 `__init__` 签名与已有组件完全相同
- 新组件的 `forward` 签名与已有组件完全相同
- 权重 key 格式与已有 converter 处理的格式相同

## 执行动作

**跳过模型文件创建/修改步骤。** 无需创建新的模型文件或 converter 文件。

### 注册条目处理

1. **计算新 checkpoint 的 model_hash**
2. **检查 hash 是否已存在**：
   - 如果已存在于 `model_configs.py` 中 → **无需新增条目**，直接使用已有条目
   - 如果不存在 → 在 `model_configs.py` 中添加新条目，复用已有 `model_class` 和 `state_dict_converter`

```python
# 示例：完全复用的注册条目处理
from diffsynth.core.loader.file import hash_model_file
from diffsynth.configs import MODEL_CONFIGS

# 计算新 checkpoint 的 hash
new_hash = hash_model_file("/path/to/new_checkpoint.safetensors")

# 检查是否已存在
existing_config = None
for config in MODEL_CONFIGS:
    if config["model_hash"] == new_hash:
        existing_config = config
        break

if existing_config:
    # Hash 已存在，无需新增条目
    print(f"Hash {new_hash} 已存在，对应 model_name: {existing_config['model_name']}")
else:
    # Hash 不存在，添加新条目（复用已有 model_class 和 converter）
    new_config = {
        "model_hash": new_hash,  # 新的 hash
        "model_name": "wan_video_dit",      # 复用相同的 model_name
        "model_class": "diffsynth.models.wan_video_dit.WanModel",  # 复用相同的 model_class
        "extra_kwargs": {},         # 如有不同参数，在此覆盖
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",  # 复用相同的 converter
    }
    # 追加到对应的 {series}_series 列表中...
```

## 一致性测试影响

完全复用的组件仍需执行两种测试（**所有情况都必须执行，永不跳过**）：
- **ModelConfig + ModelPool 加载测试: ✅ 必须** —— 验证完整注册链能正确加载 checkpoint
- **Forward 一致性测试: ✅ 必须** —— 验证不同 checkpoint 的 forward 输出与原始模型一致

### Hash 已存在 vs 不存在

| 场景 | 处理 | ModelConfig + ModelPool 加载测试 | Forward 一致性测试 |
|-----|------|------------------|-------------------|
| Hash **已存在** | 无需新增条目，直接使用已有 | ✅ 必须执行 | ✅ 必须执行 |
| Hash **不存在** | 添加新条目，复用已有 model_class 和 converter | ✅ 必须执行 | ✅ 必须执行 |

### 为什么两种测试都必须执行？

即使代码完全复用，仍需验证：
1. **Hash 计算正确** —— 确认 checkpoint 文件的 hash 计算无误
2. **注册条目匹配** —— 验证 MODEL_CONFIGS 中存在对应 hash 的条目（已有或新增）
3. **Converter 复用正确** —— 确认已有 converter 能正确处理 checkpoint 的权重 key
4. **模型实例化成功** —— 验证通过 ModelPool 能成功加载并实例化模型
5. **Forward 输出正确** —— 验证不同 checkpoint 的 forward 输出与原始模型一致

## 相关参考

- 模型文件模板 → [model-file-template.md](./model-file-template.md)
- 接入流程总览 → [SKILL.md](../SKILL.md)
