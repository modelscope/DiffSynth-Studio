# State Dict Converter 模板

文件位置：`diffsynth/utils/state_dict_converters/{series}_{component}.py`

## ⚠️ 严格禁止：Converter 必须是函数，不是类

**converter 只能是普通函数（`def`），绝对不能是类（`class`）。**

| ❌ 禁止 | ✅ 正确 |
|--------|--------|
| `class {Series}{Component}Converter:` | `def {Series}{Component}StateDictConverter(state_dict):` |
| `class StateDictConverter:` | 使用顶层函数 |
| 任何 `class` 定义的 converter | 必须是 `def` 函数 |

**原因**：
- `ModelLoader` 通过 `__call__` 协议调用 converter，期望接收一个可调用的函数对象
- 如果定义为类，需要实例化后才能调用，但注册系统直接传入类名路径并当作函数调用
- 类形式的 converter 会在运行时抛出 `TypeError: 'type' object is not callable` 或类似错误

**注册路径示例**：
```python
# ✅ 正确：指向函数
"state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter"
# 实际执行: converter_func = import_path(...)  →  converter_func(state_dict)

# ❌ 错误：如果 WanVideoDiTStateDictConverter 是个 class，调用会失败
```

## 核心原则：尽量保持 key 一致，避免不必要的转换

**最佳情况：不需要 converter。** 如果 DiffSynth 模型的 `state_dict().keys()` 与原始 checkpoint 基本一致，可以省略 `state_dict_converter` 字段。

**次佳情况：简单的 key 映射。** 只有少数几个 key 需要调整，用简单的字符串替换即可。

**避免：复杂的映射逻辑。** 如果需要大量复杂的 key 重命名，考虑是否在模型代码中保持与 checkpoint 一致的命名更好。

## 何时需要

- 权重 key 格式与 DiffSynth 模型的 `state_dict().keys()` 不一致时
- 目标库的 checkpoint key 格式与已有 converter 处理的不同时

## 何时不需要

- 权重 key 格式与已有 converter 处理的格式相同 → 直接复用
- 新版本只是参数变化，key 格式不变 → 复用已有 converter
- 目标库的 key 格式与 DiffSynth 模型一致 → 不需要 converter（注册条目中省略 `state_dict_converter` 字段）

## 重要：DiskMap 类型注意事项

**converter 接收的 `state_dict` 参数是 `DiskMap` 类型，不是普通的 dict。**

### DiskMap 特性

`DiskMap` 是 DiffSynth 的延迟加载机制，用于大模型文件的高效加载：
- **不会一次性加载所有参数到内存**
- **支持 safetensors 格式的延迟读取**
- **converter 被调用时，传入的 dict 中值是 key 名本身**（用于构建 key 映射表）

### 必须遵守的规则

| ❌ 禁止 | ✅ 正确 |
|--------|--------|
| `state_dict.items()` | `for name in state_dict:` |
| `state_dict.values()` | 不要访问值 |
| `state_dict[key].item()` | 不要访问张量数据 |
| `state_dict[key].shape` | 不要访问张量元数据 |
| `isinstance(state_dict, dict)` | 直接当作可迭代对象使用 |

### 正确的 converter 写法

```python
def {Series}{Component}StateDictConverter(state_dict):
    """
    参数 state_dict 是 DiskMap 类型。
    遍历时，name 是 key 名，state_dict[name] 也是 key 名（用于构建映射）。
    """
    state_dict_ = {}
    for name in state_dict:  # ✅ 正确：直接迭代
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            state_dict_[new_name] = state_dict[name]  # 值可以是任意占位符
    return state_dict_
```

**错误示例**：
```python
def WrongConverter(state_dict):
    # ❌ 错误：不能调用 .items()
    for name, param in state_dict.items():
        # ❌ 错误：不能访问 param.shape、param.dtype 等
        if param.shape[0] > 100:
            pass
    return state_dict
```

## 推荐写法（简洁函数形式）

大多数 converter 使用这种形式即可：

```python
def {Series}{Component}StateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            state_dict_[new_name] = state_dict[name]
    return state_dict_
```

## 如何避免复杂的 Converter

如果你发现自己在写一个包含大量 key 映射的复杂 converter，**停下来重新考虑模型代码的命名**。

**检查清单**：
- [ ] 是否可以在模型代码中使用与 checkpoint 一致的属性名？
- [ ] 是否只是个别 key 需要调整？→ 用简单替换
- [ ] 是否大部分 key 都不匹配？→ 考虑修改模型代码而非写复杂 converter

**决策流程**：
```
发现 key 不匹配时：
├── 检查是否可以修改模型代码的命名
│   └── 例如：把 self.blocks 改为 self.transformer_blocks 以匹配 checkpoint
├── 修改模型代码成本更低 → 修改模型
└── 必须保持模型代码不变 → 写 converter
```

**记住**：Converter 是最后的手段，不是首选方案。保持命名一致能省去维护 converter 的工作量。

## 常见转换模式

### 1. 移除前缀

参考：`ltx2_dit.py`

```python
def LTXModelStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:  # ✅ 正确：直接迭代 key
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            # 跳过不需要的层
            if new_name.startswith("audio_embeddings_connector.") or \
               new_name.startswith("video_embeddings_connector."):
                continue
            state_dict_[new_name] = state_dict[name]
    return state_dict_
```

### 2. 替换分隔符 / 部分重命名

参考：`ltx2_text_encoder.py`

```python
def LTX2TextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for key in state_dict:  # ✅ 正确：直接迭代 key
        if key.startswith("language_model.model."):
            new_key = key.replace("language_model.model.", "model.language_model.")
        elif key.startswith("vision_tower."):
            new_key = key.replace("vision_tower.", "model.vision_tower.")
        else:
            continue  # 跳过不匹配的 key
        state_dict_[new_key] = state_dict[key]
    return state_dict_
```

### 3. 简单 key 映射

参考：`flux_text_encoder_t5.py`

```python
def FluxTextEncoderT5StateDictConverter(state_dict):
    # ✅ 正确：{i: state_dict[i] for i in state_dict} 是安全的
    state_dict_ = {i: state_dict[i] for i in state_dict}
    # 添加一个额外的映射
    state_dict_["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    return state_dict_
```

### 4. 复杂 key 映射表

参考：`wan_video_dit.py`

```python
def WanVideoDiTStateDictConverter(state_dict):
    rename_dict = {
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        # ... 更多映射
    }
    state_dict_ = {}
    for name in state_dict:  # ✅ 正确：直接迭代 key
        if name in rename_dict:
            state_dict_[rename_dict[name]] = state_dict[name]
        else:
            # 处理带有数字索引的层（如 blocks.0, blocks.1 等）
            name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if name_ in rename_dict:
                name_ = rename_dict[name_]
                name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                state_dict_[name_] = state_dict[name]
    return state_dict_
```

### 5. 多来源格式支持

参考：`flux_dit.py`

```python
def FluxDiTStateDictConverter(state_dict):
    # 检测 checkpoint 来源（使用 `in` 检查，不是 .items()）
    is_nexus_gen = sum([key.startswith("pipe.dit.") for key in state_dict]) > 0
    if is_nexus_gen:
        dit_state_dict = {}
        for key in state_dict:  # ✅ 正确：直接迭代 key
            if key.startswith('pipe.dit.'):
                new_key = key.replace("pipe.dit.", "")
                dit_state_dict[new_key] = state_dict[key]
        return dit_state_dict
    
    # 其他来源的处理...
    rename_dict = {...}
    # ...
```

### 6. 带张量操作的转换

参考：`flux_dit.py` 中的 concat 操作

```python
# 合并多个张量
def FluxDiTStateDictConverter(state_dict):
    state_dict_ = {}
    # ... 前面的处理 ...
    
    for name in list(state_dict_.keys()):
        if ".a_to_q." in name:
            # 将 q, k, v 三个张量合并
            param = torch.concat([
                state_dict_[name],
                state_dict_[name.replace(".a_to_q.", ".a_to_k.")],
                state_dict_[name.replace(".a_to_q.", ".a_to_v.")],
            ], dim=0)
            state_dict_[name.replace(".a_to_q.", ".to_qkv.")] = param
    return state_dict_
```

## 如何确定映射规则

### 方法 1：对比 key 格式

1. **查看原始 checkpoint 的 key 格式**：
   ```python
   import torch
   ckpt = torch.load("model.safetensors", map_location="cpu")
   print(list(ckpt.keys())[:20])  # 打印前 20 个 key
   ```

2. **查看 DiffSynth 模型期望的 key 格式**：
   ```python
   from diffsynth.models.{series}_{component} import {ModelClass}
   model = {ModelClass}(...)
   print(list(model.state_dict().keys())[:20])
   ```

3. **对比两者差异**，编写映射规则

### 方法 2：运行时调试

1. 先尝试不转换直接加载，看报错信息中的 key 名
2. 根据报错逐步添加映射规则
3. 用实际 checkpoint 测试 converter 是否正确

## 参考实现

- **`ltx2_dit.py`** — 简洁函数形式，移除前缀并跳过特定层
- **`flux_text_encoder_t5.py`** — 最简单的转换，仅添加一个映射
- **`ltx2_text_encoder.py`** — 条件替换，处理多种前缀
- **`wan_video_dit.py`** — 复杂映射表，处理大量 key 重命名
- **`flux_dit.py`** — 多来源格式支持，带张量操作

## 注册时使用

在 `model_configs.py` 中：

```python
{
    "model_hash": "...",
    "model_name": "{series}_{component}",
    "model_class": "diffsynth.models.{series}_{component}.{ClassName}",
    "state_dict_converter": "diffsynth.utils.state_dict_converters.{series}_{component}.{ConverterName}",
}
```

如果不需要 converter，**省略 `state_dict_converter` 字段**即可。

## 注意事项

0. **必须是函数，不是类**：
   - ✅ `def XxxStateDictConverter(state_dict):`
   - ❌ `class XxxStateDictConverter:`
   - ❌ `class XxxConverter:`

1. **DiskMap 类型限制**：
   - ✅ 使用 `for name in state_dict:` 迭代
   - ✅ 使用 `state_dict[name]` 获取值（值是 key 名本身，不是张量）
   - ✅ 使用 `name in state_dict` 检查存在性
   - ❌ 不要使用 `state_dict.items()`、`state_dict.values()`
   - ❌ 不要访问 `.shape`、`.dtype`、`.item()` 等张量属性
   - ❌ 不要假设 `state_dict` 是普通 dict

2. **只修改 key 名，不修改值**：
   - converter 的职责是建立 key 的映射表
   - 返回的 dict 中，值可以是任意占位符（最终加载时会重新读取实际张量）

3. **跳过不需要的层**：
   - 使用 `continue` 不加入返回的 dict 即可

4. **处理特殊情况**（如 NexusGen 格式）：
   - 先检测 key 的特征（如 `key.startswith("pipe.dit.")`）
   - 然后分支处理

5. **为什么有这些限制？**
   - DiskMap 是延迟加载机制，用于高效处理大模型文件
   - converter 被调用时，只是构建 key 映射表，真正的张量数据还未加载
   - 访问张量属性会触发实际加载，但在 converter 阶段这是不允许的

## 相关参考

- 接入流程总览 → [SKILL.md](../SKILL.md) Step 4
- 模型注册 → [SKILL.md](../SKILL.md) Step 5
