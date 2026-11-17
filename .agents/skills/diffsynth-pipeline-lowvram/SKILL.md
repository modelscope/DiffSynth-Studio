---
name: diffsynth-pipeline-lowvram
description: >
  Generate low VRAM inference scripts and register fine-grained VRAM management configs for an
  integrated model in DiffSynth-Studio. Use this skill after `diffsynth-pipeline` has created the
  standard inference Pipeline and scripts, but before `diffsynth-pipeline-training`. Generates
  `model_inference_low_vram/` scripts by wrapping each `ModelConfig` with `vram_config`, and
  registers module maps in `vram_management_module_maps.py`. Use this skill when the user wants to
  add low VRAM support, enable VRAM management, or register fine-grained layer-level offload
  mappings for a model or feature being integrated.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — debugging or tuning existing VRAM management, bug fixes,
  refactoring, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 低显存推理接入

为标准 Pipeline 生成低显存推理脚本，并为每个模型组件注册细粒度显存管理。**每次调用处理一个 Pipeline 系列**，为该系列下所有已接入的功能生成对应的低显存版本。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `target_path`: `packages/{model-name}/{target-library}/`（从蓝图报告获取）
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**目标库路径**、**Conda 环境名称**和**已接入的功能信息**从蓝图报告中获取：
- 目标库路径：蓝图报告「基本信息」表格
- **Conda 环境名称**：蓝图报告「基本信息」表格中的 `Conda 环境名称` 字段。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。**
- 功能信息：蓝图报告「Pipeline 功能规划」章节

## 前置条件

本 skill 假设以下条件已满足：
1. `diffsynth-pipeline` 已完成，标准推理脚本存在于 `examples/{series}/model_inference/`
2. Pipeline 的 `from_pretrained` 已支持 `vram_limit` 参数
3. 模型组件的 `ModelConfig` 可接受 vram_config 的 8 个键值对

## VRAM 管理完整接入流程

### 接入涉及的内容

为模型接入低显存支持，需要完成以下三件事（**按顺序**）：

| 事项 | 文件 | 作用 |
|------|------|------|
| **1. 注册 module_map** | `diffsynth/configs/vram_management_module_maps.py` | 为每个模型组件类注册 Layer 级别的 VRAM wrapper 映射 |
| **2. 生成低显存推理脚本** | `examples/{series}/model_inference_low_vram/{feature}.py` | 仅对已注册 module_map 的组件注入 `**vram_config`，在 `from_pretrained` 注入 `vram_limit` |
| **3. 验证脚本正确性** | 静态检查 + 动态运行 | 确保注入后的脚本语法正确、参数完整、行为不变 |

### 显存管理工作原理

完整调用链：
```
Pipeline.from_pretrained(model_configs=[ModelConfig(..., **vram_config)], vram_limit=...)
  └─ download_and_load_models()
       └─ ModelPool.auto_load_model(vram_config, vram_limit)
            ├─ fetch_module_map(model_class, vram_config)  # 从 VRAM_MANAGEMENT_MODULE_MAPS 查找
            └─ load_model(..., module_map, vram_config, vram_limit)
                 └─ enable_vram_management(model, module_map, vram_config, vram_limit)
                      └─ enable_vram_management_recursively()  # 递归替换子模块
```

**关键判断**：`ModelPool.need_to_enable_vram_management(vram_config)` 仅在 `offload_dtype is not None AND offload_device is not None` 时返回 True。因此必须在 `ModelConfig` 中显式设置这两个值，VRAM 管理才会启用。

**module_map 查找**：`ModelPool.fetch_module_map()` 根据 `ModelConfig` 对应的模型类（如 `"diffsynth.models.wan_video_dit.WanModel"`）从 `VRAM_MANAGEMENT_MODULE_MAPS` 中查找映射。**所有模型组件的 ModelConfig 都必须注入 `**vram_config`，一个都不能少。** 如果某组件运行报错，唯一原因是其 module_map 未正确注册，解决方案是补充注册——绝不允许通过移除 `**vram_config` 来绕过问题。

**运行时**：Pipeline 的 `load_models_to_device()` 在推理过程中调用组件的 `offload()` / `onload()` 方法控制组件级切换；`AutoWrappedModule` / `AutoWrappedLinear` 的 `forward()` 中根据 `vram_limit` 动态检查显存，决定是否需要 `preparing()`（从 offload 位置加载到 computation 位置）。

### 默认 vram_config

**所有新模型默认使用以下 vram_config**：

```python
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
```

这个配置的含义：
- 模型参数平时存放在 **磁盘** 上，不占用内存和显存
- 计算时按层从磁盘读取并加载到 **CUDA（显存）** 计算
- 计算完成后释放，等待下次调用时重新读取
- 配合 `vram_limit` 控制显存占用上限

**低显存推理脚本的 ModelConfig 注入规则**：
- 只有**组件模型**的 ModelConfig 需要 `**vram_config`
- **tokenizer_config**、**processor_config** 等非组件配置**不需要**传入 `**vram_config`
- `vram_limit` 的值表示 GPU 显存上限（GB），`- 0.5` 留 0.5GB 余量防止 OOM

### 细粒度显存管理 — module_map 注册

**所有模型默认必须注册 module_map**，除非模型使用 LLM-like `generate()` 函数进行逐 token 生成（token-by-token generation）。

module_map 定义了如何将模型中的每个 Layer 包裹为 VRAM-aware 版本，格式如下：

```python
VRAM_MANAGEMENT_MODULE_MAPS = {
    "diffsynth.models.{module_path}.{ClassName}": {
        "diffsynth.models.{sub_module}.{SubClass}": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        ...
    },
}
```

**包裹器类型**：
- `AutoWrappedLinear`：专门用于 `torch.nn.Linear`，支持 FP8 量化和 LoRA 融合
- `AutoWrappedModule`：通用包裹器，用于其他所有有参数的 Layer

**引用必须用完整模块路径字符串**，不能用 class 引用。

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重读当前 Step 的描述内容，重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- 报告更新路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md`

**更新方式：先读取现有报告，再追加新内容。** 追加记录后使用 `sed -i` 更新步骤对照表中的状态。具体格式详见下方「渐进式步骤报告」章节。

**不要跳过报告更新** — 即使某个 Step 被跳过或失败，也必须记录到报告中。报告是执行过程的唯一可追溯记录。

### 0. 读取蓝图信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**每个 skill 执行的第一步，强制要求。** 从蓝图报告中读取必要信息。

```bash
MODEL_NAME="{model-name}"
BLUEPRINT_PATH="packages/${MODEL_NAME}/.sisyphus/integration-blueprints/${MODEL_NAME}-blueprint.md"
```

**本 skill 必须从蓝图报告中读取的信息：**

| 蓝图信息 | 用途 |
|---------|------|
| `Conda 环境名称` | 验证脚本运行环境 |
| `目标库路径` | 参考目标库的显存管理实现 |
| Pipeline 功能规划表 | 确定需要为哪些功能生成低显存脚本 |
| Pipeline 类名 | 定位 Pipeline 文件 |
| 模型组件列表 | 确定需要为哪些组件注册 module_map（DiT、VAE、TextEncoder、ControlNet 等） |

**判断模型是否为逐 token 模式**：
- 检查模型是否调用类似 LLM 的 `generate()` 函数进行逐 token 生成
- 如果是逐 token 模式，则**跳过 module_map 注册**，但仍需生成低显存推理脚本
- 非逐 token 模式的所有模型**必须执行 module_map 注册**

**验证前置条件**：
1. 确认 `examples/{series}/model_inference/` 下存在推理脚本
2. 确认 Pipeline 的 `from_pretrained` 支持 `vram_limit` 参数
3. 确认 `model_configs.py` 中已注册对应的模型 ID

如果前置条件不满足，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_lowvram"
mkdir -p ${EXEC_LOG_DIR}/{outputs,scripts,checkpoints}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始生成低显存脚本前，先制定完整的执行计划**，输出到 `packages/{model-name}/.sisyphus/plans/lowvram-plan.md`。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物」。模板如下：

```markdown
# Low VRAM 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pipeline-lowvram |
| 执行时间 | {timestamp} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取低显存接入所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、目标库路径、Pipeline 功能规划表、Pipeline 类名、模型组件列表
- 判断是否为逐 token 模式（决定是否注册 module_map）
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_lowvram/` 目录及子目录
- 创建 `manifest.json` 和 `latest` 软链接

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示低显存接入规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/lowvram-plan.md`
- 向用户展示以下内容并等待确认：
  - 功能列表：每个功能对应的标准脚本和低显存脚本路径
  - 组件列表：哪些组件注册 module_map、哪些跳过（如逐 token 模式）
  - module_map 注册方案：每个组件需要注册的 Layer 类型
  - **脚本规划表**：明确列出所有待操作的脚本文件

**脚本规划表**格式如下（必须包含）：

## 脚本规划

### 需要生成的低显存脚本

| 标准脚本 | 低显存脚本 | 状态 | 备注 |
|---------|-----------|------|------|
| `examples/{series}/model_inference/{feature1}.py` | `examples/{series}/model_inference_low_vram/{feature1}.py` | 新建/已有需修正 | DiT + TextEncoder 注入 vram_config |

### 需要修正的已有低显存脚本（如存在）

| 脚本路径 | 问题 | 修正内容 |
|---------|------|---------|
| `examples/{series}/model_inference_low_vram/{feature}.py` | vram_config 值不匹配/PE 误注入 | 修正为默认值/移除 `**vram_config` |

**产出物**：
- `packages/{model-name}/.sisyphus/plans/lowvram-plan.md`（含脚本规划表）

---

### Step 3: 分析标准推理脚本

**目标**：确认 Pipeline 支持 vram_limit，并提取 ModelConfig 列表、推理参数。

**执行内容**：
- 确认 Pipeline 的 `from_pretrained` 已支持 `vram_limit` 参数
- 如不支持，说明 `diffsynth-pipeline` step 未正确完成，应返回上一步修复，本 skill 不修改 Pipeline 代码
- 读取 `examples/{series}/model_inference/{feature}.py`
- 提取 ModelConfig 列表、from_pretrained 调用位置、pipe() 推理参数、输出保存方式

**产出物**：每个功能的标准脚本分析结果

---

### Step 4: 分析模型组件结构

**目标**：为每个模型组件分析 Layer 结构，生成 module_map 草案。

**执行内容**：
- 通过 `print(model)` 或代码分析，识别包含参数的 Layer
- 确定每个 Layer 的包裹器类型（AutoWrappedLinear / AutoWrappedModule）
- 输出每个组件的 module_map 草案

**产出物**：module_map 草案

---

### Step 5: 注册细粒度显存管理配置

**目标**：将 module_map 注册到 `vram_management_module_maps.py`。

**执行内容**：
- 读取现有 `vram_management_module_maps.py`，了解已有配置风格
- 为每个模型组件添加 module_map 条目
- 如有 transformers 版本依赖，添加 VERSION_CHECKER_MAPS 条目

**产出物**：
- 更新后的 `diffsynth/configs/vram_management_module_maps.py`

---

### Step 6: 生成低显存推理脚本

**目标**：从标准推理脚本生成低显存版本，仅对已注册 module_map 的组件注入 vram_config。

**执行内容**：
- 复制标准脚本，在 ModelConfig 列表前插入 vram_config 定义
- 仅对已注册 module_map 的 ModelConfig 添加 `**vram_config`
- 在 from_pretrained 中添加 `vram_limit` 参数
- 推理参数和输出保存方式保持不变

**产出物**：
- `examples/{series}/model_inference_low_vram/{feature}.py`（每个功能一个）

---

### Step 7: 验证低显存脚本

**目标**：验证生成的低显存脚本正确性，确认输出与标准版本一致，记录峰值显存。

**执行内容**：
- 静态验证：脚本数量、vram_config 注入、vram_limit 参数等
- 动态验证：运行低显存脚本，确认不报错
- 输出一致性验证：对比低显存版本与标准版本的输出
- 记录峰值显存占用

**产出物**：
- 验证结果
- 显存记录（最高占用）

---

### Step 8: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查 Plan 文件、skill_work_report 是否存在
- 检查低显存推理脚本是否全部生成
- 如有缺失，立即补充

**产出物**：验证通过确认
```

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始生成低显存脚本。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 3. 分析标准推理脚本

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**确认 Pipeline 的 `from_pretrained` 已支持 `vram_limit` 参数。**

DiffSynth-Studio 的 Pipeline 基类已内置 VRAM 管理支持，`from_pretrained` 应包含 `vram_limit: float = None` 参数，并在内部调用 `download_and_load_models(model_configs, vram_limit)`。

如果 Pipeline 是新生成的（通过 `diffsynth-pipeline`），该参数应已存在。**如果 Pipeline 不支持 `vram_limit`，说明 `diffsynth-pipeline` step 未正确完成，应返回上一步修复，本 skill 不修改 Pipeline 代码。**

**读取每个标准推理脚本**，提取以下信息：

| 提取项 | 从哪提取 | 用途 |
|--------|---------|------|
| `ModelConfig` 列表 | `model_configs=[...]` 参数 | 确定哪些组件需要添加 `**vram_config` |
| `from_pretrained` 调用 | `Pipeline.from_pretrained(...)` 行 | 确定 `vram_limit` 插入位置 |
| Pipeline 类名 | 文件头部 import | 确认 Pipeline 类型 |
| 推理参数 | `pipe(...)` 调用 | 确保低显存版本推理参数不变 |
| 输出保存方式 | 文件尾部 `.save()` | 确保低显存版本输出格式一致 |

读取文件：
- `examples/{series}/model_inference/{feature}.py` — 每个功能一个标准推理脚本

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 4. 分析模型组件结构

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**为每个需要注册 module_map 的模型组件，分析其 Layer 结构。**

**方法一：通过 `print(model)` 分析（推荐）**

在 Python 环境中加载模型并 `print(model)`，观察模型结构，找出所有包含参数的 Layer：

```python
from diffsynth.models.{component} import {ComponentClass}
import torch

model = {ComponentClass}(**config)  # 或通过 load_model 加载
print(model)
```

**方法二：通过代码分析**

读取模型文件 `diffsynth/models/{component}.py`，手动找出所有包含参数的 Layer 类。

**需要识别的有参数 Layer**（参考已有注册中的常见条目）：

| Layer | 包裹器 | 说明 |
|---|---|---|
| `torch.nn.Linear` | `AutoWrappedLinear` | **每个组件必有** |
| `torch.nn.Embedding` | `AutoWrappedModule` | TextEncoder、含 embedding 的组件 |
| `torch.nn.Conv1d` | `AutoWrappedModule` | Audio 编码、MotionController |
| `torch.nn.Conv2d` | `AutoWrappedModule` | VAE、ImageEncoder |
| `torch.nn.Conv3d` | `AutoWrappedModule` | Video VAE、3D DiT |
| `torch.nn.LayerNorm` | `AutoWrappedModule` | DiT、TextEncoder（仅 `elementwise_affine=True`） |
| `torch.nn.GroupNorm` | `AutoWrappedModule` | FLUX DiT、VAE |
| 自定义 `RMSNorm` | `AutoWrappedModule` | 各模型自定义的 RMSNorm 类，路径不同 |
| `Upsample` / `ConvTranspose1d/2d` | `AutoWrappedModule` | VAE 上采样 |
| DiTBlock 等复合模块 | `AutoWrappedNonRecurseModule` | 仅在组件内部管理递归时使用（Wan/Mova 有先例） |
| 其他自定义有参 Layer | `AutoWrappedModule` | 如 `Snake1d`、`FusedLeakyReLU` 等

**需要忽略的无参数 Layer**：
- `SiLU`、`GELU`、`ReLU` 等激活函数
- `Dropout`
- `elementwise_affine=False` 的 `LayerNorm`
- 纯容器类（`ModuleList`、`Sequential` 仅作为容器，不需要包裹，内部 Layer 单独处理）

**输出每个组件的 module_map 草案**，用于下一步注册。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 5. 注册细粒度显存管理配置

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**所有模型默认必须注册 module_map，除非模型使用 LLM-like `generate()` 函数进行逐 token 生成。**

**何时跳过 module_map 注册**：
- 仅当模型调用类似 LLM 的 `generate()` 函数（逐 token 模式 / token-by-token generation）时跳过
- 所有其他模型类型（DiT、VAE、UNet、CNN、Text Encoder 等）都必须执行

**注册步骤**：

1. **读取** `diffsynth/configs/vram_management_module_maps.py`，了解现有注册风格

2. **为每个模型组件添加 module_map 条目**：

```python
VRAM_MANAGEMENT_MODULE_MAPS = {
    ...
    "diffsynth.models.{component}.{ClassName}": {
        "diffsynth.models.{component}.{SubClass}": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        ...
    },
}
```

**关键规则**：
- `torch.nn.Linear` 必须映射到 `AutoWrappedLinear`（专用包裹器，支持 FP8 和 LoRA）
- 其他有参 Layer 映射到 `AutoWrappedModule`
- 所有引用必须用**完整模块路径字符串**
- transformers 库中的类也要写完整路径，如 `"transformers.models.siglip.modeling_siglip.SiglipVisionEmbeddings"`

**参考已有配置风格**：写入前必须先读取 `vram_management_module_maps.py`，参考已有模型的写法（如 FLUX、Wan、Qwen-Image 的写法）。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 6. 生成低显存推理脚本

> 📖 开始前：重读本步骤描述，确认流程与报告路径

对每个功能 `{feature}`，从标准推理脚本 `model_inference/{feature}.py` 生成低显存版本 `model_inference_low_vram/{feature}.py`。

**生成规则**：

1. **复制标准脚本的全部内容**，不做任何删改

2. **所有模型组件的 `ModelConfig` 都注入 vram_config**：
   - 所有组件模型（DiT、VAE、TextEncoder、ControlNet 等）的 `ModelConfig` 都添加 `**vram_config`
   - `tokenizer_config`、`processor_config` 等非组件配置**不需要**传入 `**vram_config`
   - 如果某组件运行报错，说明其 module_map 未正确注册，需补充注册（不要去掉 `**vram_config`）

3. **在已注册 module_map 的 `ModelConfig` 列表之前插入默认 vram_config 定义**：

```python
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
```

4. **在所有模型组件的 `ModelConfig(...)` 中添加 `**vram_config`**：

```python
# 修改前
ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),

# 修改后
ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
```

5. **在 `from_pretrained(...)` 中添加 `vram_limit`**：

```python
# 修改前
pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[...],
    tokenizer_config=...,
)

# 修改后
pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[...],
    tokenizer_config=...,
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
```

6. **推理参数和输出保存方式保持不变**

**文件位置**：`{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py`

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 7. 验证低显存脚本

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**静态验证**（必须执行）：
- [ ] `model_inference_low_vram/` 下脚本数量 = `model_inference/` 下脚本数量
- [ ] 每个低显存脚本包含 `vram_config` 定义（8 个键）
- [ ] 已注册 module_map 的组件对应的 `ModelConfig` 调用包含 `**vram_config`
- [ ] 所有组件对应的 `ModelConfig` 调用均包含 `**vram_config`（含 DiT、VAE、TextEncoder 等所有模型组件）
- [ ] `from_pretrained` 调用包含 `vram_limit` 参数
- [ ] 推理参数与标准版本完全一致
- [ ] 输出保存方式与标准版本完全一致
- [ ] `vram_management_module_maps.py` 中存在所有使用 `**vram_config` 组件的 module_map 条目

**动态验证：成功运行低显存脚本**（如环境允许）：

```bash
cd {diffsynth_root}
export DIFFSYNTH_SKIP_DOWNLOAD=true
python examples/{series}/model_inference_low_vram/{first_feature}.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/low_vram_run.log
```

**如果运行报错**：
- **唯一修复方式**：补充 module_map 注册（见上方 Debug 提示）
- 查看报错定位漏注册的 Layer → 添加到 `vram_management_module_maps.py` → 重新运行
- 重复直到无报错
- **禁止**：修改 Pipeline/Model 代码、修改推理脚本逻辑、修改 vram_config 值

**输出一致性验证**（可选，如已有标准版本输出）：

对比低显存版本与标准版本的输出：
- 输出类型一致（图像/视频/音频）
- 输出形状一致（分辨率、帧数、通道数）
- 视觉上无明显差异（图像/视频）

**记录显存占用**（如环境允许）：

编写测试脚本，传入 `vram_limit=0`（不限制显存），记录峰值显存：

```python
# 测试脚本：packages/{model-name}/.sisyphus/execution-logs/{timestamp}_lowvram/scripts/test_vram.py
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        # 所有组件都加 **vram_config
        ModelConfig(model_id="...", origin_file_pattern="...", **vram_config),
    ],
    vram_limit=0,  # 不限制显存
)
# 使用与低显存脚本相同的推理参数
output = pipe(...)
output.save("output_vram_test.jpg")

peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"[VRAM] Peak VRAM usage: {peak_vram:.2f} GB")
```

```bash
cd {diffsynth_root}
export DIFFSYNTH_SKIP_DOWNLOAD=true
python packages/{model-name}/.sisyphus/execution-logs/{timestamp}_lowvram/scripts/test_vram.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/vram_test.log
```

记录结果到报告：

| 字段 | 值 |
|------|-----|
| 低显存脚本运行 | PASSED / FAILED |
| 输出一致性 | PASSED / FAILED |
| 测试脚本 vram_limit | 0 (不限制) |
| 最高显存占用 | {peak_vram:.2f} GB |

> 注：显存占用仅供参考，超出 24GB 只需记录，不影响验证通过判定。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

### 8. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

所有步骤完成后，执行最终验证：

```bash
# 1. 检查报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/lowvram-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 2. 检查低显存脚本是否存在
LOW_VRAM_DIR="packages/{model-name}/DiffSynth-Studio/examples/{series}/model_inference_low_vram"
for script in "$LOW_VRAM_DIR"/*.py; do
  if [ -f "$script" ]; then
    echo "OK: $script 已创建"
  else
    echo "WARNING: 低显存脚本缺失: $script"
  fi
done

# 3. 检查显存测试日志
VRAM_TEST_LOG="packages/{model-name}/.sisyphus/execution-logs/{timestamp}_lowvram/outputs/vram_test.log"
if [ -f "$VRAM_TEST_LOG" ]; then
  echo "OK: 显存测试日志已生成"
  grep "Peak VRAM" "$VRAM_TEST_LOG"
else
  echo "WARNING: 显存测试日志缺失（可能无 GPU 环境）"
fi
```

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/lowvram-report.md`

## 蓝图需求

本 skill 需要从蓝图报告中获取：
- **Conda 环境名称**：验证脚本运行环境
- **目标库路径**：参考目标库的显存管理实现
- **Pipeline 功能列表**：确定需要生成哪些低显存推理脚本
- **Pipeline 类名**：确认 Pipeline 结构
- **模型组件列表**：确定需要为哪些组件注册 module_map

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的验证脚本（含 `monitor_vram.py` 显存监控脚本）
- **输出目录**: `outputs/` - 命令执行日志 + `vram-monitor.log`（显存监控结果）

### 生成的文件

- **低显存推理脚本**: `{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py`
  - 每个 `model_inference/{feature}.py` 对应一个低显存版本
- **细粒度显存管理配置**: `{diffsynth_root}/diffsynth/configs/vram_management_module_maps.py`（新增条目）

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md`

**首次创建（skill 开始时）**：
```bash
mkdir -p packages/{model-name}/.sisyphus/skill_work_report

cat > packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md << EOF
# diffsynth-pipeline-lowvram 执行报告

> 自动生成，记录本 skill 执行过程中的每个步骤。

## 基本信息

| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pipeline-lowvram |
| 开始时间 | \$(date -Iseconds) |
| 状态 | 进行中 |

## 步骤对照表

| Step | 名称 | 对应 Workflow | 状态 |
|------|------|---------------|------|
| 1 | 读取蓝图信息 | Step 0 | ⬜ |
| 2 | 初始化执行日志目录 | Step 1 | ⬜ |
| 3 | 制定执行计划 | Step 2 | ⬜ |
| 4 | 分析标准推理脚本 | Step 3 | ⬜ |
| 5 | 分析模型组件结构 | Step 4 | ⬜ |
| 6 | 注册细粒度显存管理配置 | Step 5 | ⬜ |
| 7 | 生成低显存推理脚本 | Step 6 | ⬜ |
| 8 | 验证低显存脚本 | Step 7 | ⬜ |
| 9 | 最终验证 | Step 8 | ⬜ |

---

## 执行时间线
EOF
```

**每完成一个步骤后追加**：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败 / ⬜ 跳过
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

同时更新步骤对照表中的状态：
```bash
sed -i 's/| {N} | {步骤名称} | ... | ⬜ |/| {N} | {步骤名称} | ... | ✅ |/' \
  packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md
```

**全部完成后追加摘要**：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md << EOF

---

## 执行摘要

- **总步骤数**: 9
- **成功**: {N}
- **失败**: {N}
- **跳过**: {N}
- **总耗时**: {duration}
EOF

# 更新状态为已完成
sed -i 's/| 状态 | 进行中 |/| 状态 | 已完成 |/' \
  packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md
```

### 向用户报告

低显存脚本生成完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/lowvram-report.md << 'OUTER_EOF'
## ✅ 低显存推理脚本生成完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 生成的文件

- 低显存推理脚本: `{diffsynth_root}/examples/{series}/model_inference_low_vram/`
- vram_config: 默认 Disk Offload（offload/onload 均为 `"disk"`）
- 细粒度显存管理: {已注册 module_map / 逐 token 模式跳过}

### module_map 注册

| 组件类 | module_map 条目数 |
|--------|-------------------|
| {component1} | {N} |
| {component2} | {N} |

### 验证结果

| 字段 | 值 |
|------|-----|
| 低显存脚本运行 | PASSED / FAILED |
| 输出一致性 | PASSED / FAILED |
| 最高显存占用 | {peak_vram:.2f} GB |

请确认脚本是否符合预期。
OUTER_EOF
```

## 核心原则：Bug 修复唯一路径

**本 skill 执行期间，如果出现任何报错或异常，唯一的修复方式是修改 module_map 注册。**

具体规则：
1. **不修改 Pipeline 代码** — 禁止改动 `diffsynth/pipelines/` 下的任何文件
2. **不修改 Model 代码** — 禁止改动 `diffsynth/models/` 下的任何文件
3. **不修改低显存推理脚本的逻辑** — 生成的脚本只注入 `vram_config` 和 `vram_limit`，不为了修 bug 而改脚本逻辑
4. **所有组件都必须注入 vram_config** — DiT、VAE、TextEncoder、ControlNet 等每一个模型组件的 ModelConfig 都必须添加 `**vram_config`，一个都不能少
5. **唯一的修 bug 方式：注册 module_map** — 所有运行时报错都归结为 module_map 中遗漏了某个 Layer 的注册

**为什么所有 bug 都只能通过注册 module_map 修复？**

VRAM 管理的工作原理是：框架通过 `module_map` 查找表，递归地将模型中的每个 Layer 替换为 VRAM-aware wrapper。如果某个 Layer **没有被注册到 module_map 中**，它会保持原始状态：
- 参数停留在 `offload_device` 上不会被加载
- 或者不会被正确的 wrapper 包裹，导致显存管理失效
- 从而导致 device mismatch 等运行时错误

因此，**所有低显存推理的 bug 本质上都是 "漏注册" 问题**，解决方案只有一个：找到漏掉的 Layer，把它加到 module_map 中。

**排查步骤**：
1. 读取报错信息，定位是哪个 Layer 的 `forward` 出了问题
2. 检查该 Layer 的类是否已在对应组件的 `module_map` 中注册
3. 如果未注册 → 添加到 `vram_management_module_maps.py` 中对应组件的条目下
4. 重新运行，如果还有报错 → 重复步骤 1-3，直到所有 Layer 都已注册

**不要做的事情**：
- 不要修改低显存脚本的推理逻辑
- 不要修改 Pipeline 的 `from_pretrained` 或 `download_and_load_models`
- 不要修改模型文件的 `forward` 方法
- 不要试图通过修改 vram_config 的值来绕过问题
- 不要添加条件判断或 try/except 来吞掉错误

## 低显存接入原则

- **不修改 Pipeline 源代码。** 禁止修改 `diffsynth/pipelines/` 下的任何文件，Pipeline 的 `vram_limit` 参数支持应由 `diffsynth-pipeline` 在创建时完成
- **不修改 Model 源代码。** 禁止修改 `diffsynth/models/` 下的任何文件，VRAM 管理通过外部 module_map 配置实现，不侵入模型代码
- **仅修改 module_map 配置。** 只能编辑 `diffsynth/configs/vram_management_module_maps.py` 注册映射
- **仅生成新的低显存推理脚本。** 在 `model_inference_low_vram/` 下创建新文件，不修改 `model_inference/` 下的标准脚本
- **以标准脚本为模板。** 低显存脚本在标准推理脚本基础上注入 vram_config 和 vram_limit，不重写
- **所有模型默认注册 module_map。** 除 LLM 逐 token 模式外，所有模型组件都必须在 `vram_management_module_maps.py` 中注册，这是默认行为
- **所有组件都必须注入 vram_config。** DiT、VAE、TextEncoder、ControlNet 等每一个模型组件的 ModelConfig 都必须添加 `**vram_config`，一个都不能少。如果某组件运行报错，唯一原因是 module_map 未正确注册，需补充注册（绝不允许通过移除 `**vram_config` 来绕过问题）
- **vram_config 三层含义独立。** offload/onload/preparing/computation 的 device 和 dtype 可独立配置，默认 offload/onload 为 `"disk"`，模型参数平时驻留磁盘
- **推理行为不变。** 低显存脚本的推理参数、输出格式必须与标准版本完全一致
- **module_map 有参考。** 注册时必须参考 `vram_management_module_maps.py` 中已有模型的写法，保持风格一致
- **优先使用 AutoWrappedModule。** 不要使用 `AutoWrappedNonRecurseModule`，除非 `AutoWrappedModule` 运行报错。`AutoWrappedNonRecurseModule` 用于阻止递归包裹的特殊场景（如 Wan/Mova 的 DiTBlock 内部管理递归），但大多数模型不需要。先试 `AutoWrappedModule`，报错再换。
- **module_map 注册最小单元原则。** 不要注册复合模块（如 `FeedForward`、`TimestepEmbedding`、`PixArtAlphaTextProjection` 等），只注册最底层的有参 Layer：`torch.nn.Linear` → `AutoWrappedLinear`，`torch.nn.Conv2d`、`RMSNorm`、`LayerNorm` 等 → `AutoWrappedModule`。**例外情况**：如果某个大类/复合模块内部包含无法单独注册的纯参数子类（比如自定义的非标准有参 Layer，不属于 `torch.nn` 或 `transformers` 中的已知类），才需要把该大类模块整体注册，以确保这些参数被 VRAM 管理覆盖。

## Debug 提示

**遇到任何 device 报错或运行时错误，唯一修复方式：补充 module_map 注册。**

> 所有低显存推理的 bug 都只有一个原因：**module_map 中漏掉了某个包含参数的 Layer**。
>
> 所有低显存推理的 bug 都只有一个修复方式：**将漏掉的 Layer 添加到 module_map 注册表中**。
>
> **不要修改任何其他代码**（不修改 Pipeline、不修改 Model、不修改推理脚本逻辑、不修改 vram_config 值），只修 module_map 注册。
>
> **排查方法**：
> 1. 查看报错信息，定位是哪个 Layer 的 forward 出了问题
> 2. 检查该 Layer 的类是否已注册到 module_map 中
> 3. 如果未注册，将其添加到对应组件的 module_map 条目中
> 4. 重新运行，如果还有报错 → 重复步骤 1-3，直到所有 Layer 都已注册
>
> **常见遗漏场景**（基于已有注册总结）：
>
> | 遗漏类型 | 示例 | 说明 |
> |---------|------|------|
> | **自定义 RMSNorm** | `diffsynth.models.wan_video_dit.RMSNorm` | 每个 DiT 模型都有自己的 RMSNorm，路径各不相同，必须单独注册 |
> | **transformers 库的 Norm** | `transformers.models.t5.modeling_t5.T5LayerNorm`<br>`transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm`<br>`transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm` | TextEncoder 依赖 transformers 的 LayerNorm/RMSNorm，路径随版本变化 |
> | **Upsample** | `diffsynth.models.wan_video_vae.Upsample` | VAE 上采样模块，包含 Conv 参数 |
> | **CausalConv3d** | `diffsynth.models.wan_video_vae.CausalConv3d` | 视频 VAE 特有的 3D 因果卷积 |
> | **特殊激活/归一化** | `torch.nn.SiLU`（有参时）<br>`diffsynth.models.mova_audio_vae.Snake1d`<br>`diffsynth.models.wan_video_animate_adapter.FusedLeakyReLU` | 一般 SiLU 无参数可忽略，但有自定义变体时需注册 |
> | **复合模块（特殊情况）** | `diffsynth.models.wan_video_dit.DiTBlock` → `AutoWrappedModule` | 仅当复合模块内部包含无法单独注册的自定义参数子类时才注册整体模块，否则优先注册内部基础 Layer（Linear、Norm 等） |
> | **Embedding 子类** | `transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTEmbeddings` | 自定义 Embedding 或包含多个 Embedding 的复合模块 |
> | **其他自定义 Layer** | `diffsynth.models.wan_video_animate_adapter.EqualLinear`<br>`diffsynth.models.wan_video_animate_adapter.ConvLayer` | 目标库特有的自定义 Layer |
>
> **检查清单** — 对照 `print(model)` 输出，逐一确认以下类型的 Layer 都已注册：
> 1. [ ] `torch.nn.Linear` → `AutoWrappedLinear`（**每个组件必有**）
> 2. [ ] `torch.nn.Embedding` → `AutoWrappedModule`（如有）
> 3. [ ] `torch.nn.Conv1d` → `AutoWrappedModule`（如有）
> 4. [ ] `torch.nn.Conv2d` → `AutoWrappedModule`（如有）
> 5. [ ] `torch.nn.Conv3d` / `CausalConv3d` → `AutoWrappedModule`（如有）
> 6. [ ] `torch.nn.LayerNorm`（`elementwise_affine=True`）→ `AutoWrappedModule`（如有）
> 7. [ ] `torch.nn.GroupNorm` → `AutoWrappedModule`（如有）
> 8. [ ] 所有自定义的 RMSNorm / LayerNorm 变体 → `AutoWrappedModule`
> 9. [ ] 所有来自 transformers 的 Norm 类 → `AutoWrappedModule`
> 10. [ ] `Upsample` / `ConvTranspose1d` / `ConvTranspose2d` → `AutoWrappedModule`（如有）
> 11. [ ] 其他自定义有参 Layer → `AutoWrappedModule`
