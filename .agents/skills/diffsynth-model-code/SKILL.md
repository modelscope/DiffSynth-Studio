---
name: diffsynth-model-code
description: >
  Integrate model code (DiT, VAE, TextEncoder, etc.) into DiffSynth-Studio's models/ directory
  and register them in configs. Use this skill after the environment step is done, or when, during a
  model integration, the user wants to add new model components, register a new model hash, write a
  state_dict_converter, or run consistency tests between the original model and the integrated
  version. This skill handles the full range from "just add a new model_hash for a version upgrade"
  to "create entirely new model files for a new series."
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — bug fixes or refactoring of existing model code, performance
  tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 模型代码接入

将目标模型的组件接入 DiffSynth-Studio，按需创建或修改文件。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `target_path`: `packages/{model-name}/{target-library}/`（从蓝图报告获取）
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**目标库路径**和**Conda 环境名称**从蓝图报告中获取：
- 目标库路径：蓝图报告「基本信息」表格
- **Conda 环境名称**：蓝图报告「基本信息」表格中的 `Conda 环境名称` 字段。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。**
  - **单一数据源**：蓝图报告是环境名称的唯一存储位置，详见 [blueprint-contract.md](../diffsynth-integrator/references/blueprint-contract.md)

**执行日志初始化**遵循 [exec-log-init.md](../diffsynth-integrator/references/exec-log-init.md) 规范。

## 核心原则

### 原则 1：只做必要的改动

版本升级时，如果模型结构不变，只需在 `model_configs.py` 注册新 hash + extra_kwargs。只有 `__init__` 或 `forward` 签名变化时才需要修改模型文件。

### 原则 2：最小重构原则（关键）

**优先直接复制代码，除非要解决依赖问题，否则不建议修改代码。**

拿到目标库的模型代码后，第一步是直接复制完整文件，不改动任何逻辑。之后只做以下两类修改：

| 允许的修改 | 说明 | 示例 |
|-----------|------|------|
| **解决依赖问题** | 替换无法 import 的模块为 DiffSynth 核心模块或内联实现 | `from diffusers.models.activations import ACT2FN` → 内联定义或替换 |
| **替换 Attention / Gradient Checkpointing** | 使用 DiffSynth 的 `attention_forward` 和 `gradient_checkpoint_forward` | 目标库自研 attention → `from ..core.attention import attention_forward` |
| **保持 key 一致性的必要调整** | 仅当与 DiffSynth 核心模块冲突时改名 | 如必须使用 `attention_forward` 时，q/k/v 投影名可能需要调整 |

除上述两类外，**不要修改代码**：不重构结构、不统一风格、不提取公共函数、不重命名变量。

**为什么这样做？**
- 目标库的代码经过了完整测试，修改越多，引入 bug 的概率越大
- 复制后只改依赖，改动面最小化，定位问题更容易
- 当上游版本更新时，diff 对比清晰，合并成本低
- 保持原始逻辑不动，forward 输出一致性自然更高

**核心原则：以不需要 converter 为目标进行模型接入。**

这是 DiffSynth 模型接入的核心诉求——**DiffSynth 模型的 `state_dict().keys()` 必须与原始 checkpoint 的 keys 尽可能完全一致，尤其是 DiT 模型。**

**为什么？**
- 模型结构与开源社区一致 → 社区训练的 LoRA、checkpoint 可以直接使用，无需额外适配
- 没有 converter → 减少维护成本，降低出风险
- 模型架构透明 → 用户和开发者能直接理解模型的参数组织方式

**决策树**：
```
DiffSynth 模型的 state_dict().keys() vs 原始 checkpoint 的 keys：
├── 基本一致 → ✅ 不需要 converter（首选目标）
├── 少量 key 需要映射 → 写简单的 converter（可接受）
└── 大量 key 都不匹配 → ❌ 复杂的 converter 容易出错，必须向用户询问方案
```

**具体做法**：

| ❌ 避免 | ✅ 正确 |
|--------|--------|
| 重命名 `self.proj` 为 `self.proj_qkv` | 保持原名 `self.proj`，checkpoint 中有 "proj.weight" |
| 重构 `layers` 为 `blocks` | 保持目标库的命名，如 `self.layers = nn.ModuleList(...)` |
| 改变层的嵌套结构 | 保持与 checkpoint key 匹配的结构 |
| 在 `__init__` 中随意重命名变量 | `self.{name}` 直接对应 checkpoint 中的 `{name}.weight` |

**强制要求（仅针对 DiT 模型）**：
- DiT 模型 **必须** 以不需要 converter 为目标进行接入
- 如果由于 DiffSynth 框架限制（如必须使用 `attention_forward` 等核心模块）导致无法保持 key 一致，**必须向用户说明情况并询问解决方案**，不要自行编写复杂 converter
- 在编写 DiT 模型代码之前，先对比目标库的 `state_dict().keys()` 与 DiffSynth 现有的 key 模式，确认能保持一致再动手
- 其他组件（VAE、TextEncoder 等）以尽量保持一致为目标，但不做同等强制要求

**注意**：不要求 100% 一致，尽量保持一致即可。个别需要调整的 key 通过 converter 处理，但应以"不需要 converter"为第一目标。

### 原则 3：向后兼容是前提

version upgrade 时，任何对现有模型文件的修改都必须保证：
- 不改变已有模型的 `forward` 逻辑和输出
- 所有新增参数必须有默认值（False 或 None），确保已有注册条目行为不变
- 不删除或修改已有注册条目
- 不干扰已有的推理和训练功能

### 原则 4：初始化参数必须以目标库打印结果为准

**模型 `__init__` 参数的默认值必须先在目标库中打印确认，再填入 DiffSynth 模型。** 不得猜测或凭直觉填写。

执行流程：
1. 在目标库中初始化模型，打印所有参数值 → 保存为权威参考
2. 编写 DiffSynth 模型 `__init__` 时，逐行对比打印结果
3. 确认一致后再继续后续步骤

**高频错误模式**：
- 参数名猜错：`num_heads` vs `n_heads` vs `num_attention_heads`
- 数值猜错：`hidden_size=2048` 实际是 `1536`
- list 参数编造：`downsampling_ratios=[2,4,4,6,10]` 实际顺序不同
- 布尔参数设反：`bias=True` 实际是 `False`
- 漏掉必填参数：目标库有 `rope_scaling` 但 DiffSynth 模型没定义

**验证方法**：见 Step 5「初始化参数验证」。

**执行留痕，过程可追溯。** 所有测试脚本先保存再执行，所有命令输出保存到日志，关键检查点明确标记。遵循 [execution-traceability.md](../diffsynth-integrator/references/execution-traceability.md) 规范。

**在 DiffSynth-Studio 根目录下运行。** 所有涉及 diffsynth 的验证代码（测试脚本、import、ModelConfig 解析等）都必须在 `{diffsynth_root}` 下执行，确保能正确 import `diffsynth` 模块。

## DiffSynth 模型接入架构

DiffSynth 的模型系统由三个部分组成：

| 部分 | 位置 | 作用 |
|------|------|------|
| 模型文件 | `diffsynth/models/{series}_{component}.py` | 定义模型类（`__init__` + `forward`） |
| 权重转换器 | `diffsynth/utils/state_dict_converters/{series}_{component}.py` | 将原始 checkpoint key 映射为 DiffSynth 格式 |
| 模型注册 | `diffsynth/configs/model_configs.py` | 注册 model_hash → model_class 的映射 |

**加载机制**：`ModelLoader` 通过 `hash_model_file` 计算 checkpoint 文件的 state dict key+shape hash，在 `model_configs.py` 中查找对应条目，自动加载模型类和转换器。同一个 checkpoint 可以有多个注册条目（不同 extra_kwargs）。

**model_hash 计算**：使用 `diffsynth.core.loader.file.hash_model_file` 函数。这个 hash 不是文件 MD5，而是对 state dict 的 key 名和 shape 排序后计算的 MD5。相同架构的文件会得到相同的 hash。

## 模型拆分设计原则

**核心问题**：目标库的模型代码往往耦合了训练逻辑、推理逻辑、条件准备和模型前向传播。DiffSynth 要求清晰的模型边界。

**完整设计原则**见 [references/model-separation-design.md](references/model-separation-design.md)。以下是关键要点：

### 四大拆分原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **DiT 只负责去噪预测** | 输入噪声 latent + 条件 embeds，输出 predicted noise/velocity | 不处理文本、不编码音频、不解码 VAE |
| **条件准备是独立模型** | TextEncoder、AudioEncoder、TimbreEncoder 各自独立 | 每个都是独立的 `nn.Module` |
| **VAE 独立编解码** | VAE 是完全独立的组件，不知道 DiT 的存在 | `encode()` 和 `decode()` 方法 |
| **Pipeline 负责完整流程** | Pipeline 是唯一的编排者，决定调用顺序 | 条件准备 → 去噪循环 → VAE 解码 |

### 拆分步骤

1. **识别耦合点**：找出训练+推理耦合、条件+DiT 耦合、VAE+DiT 耦合、采样+模型耦合
2. **定义模型边界**：为每个组件定义清晰的接口（输入/输出张量）
3. **实现拆分**：
   - 提取条件编码器为独立模型
   - 将训练逻辑移到训练脚本（不在模型文件中）
   - 确保 DiT forward 只接收张量，不接收原始文本/音频

### ACE-Step 拆分示例

| 目标库组件 | DiffSynth 模型 | 说明 |
|-----------|---------------|------|
| AceStepLyricEncoder | `AceStepTextEncoder` | 独立文本编码器 |
| AceStepTimbreEncoder | `AceStepTimbreEncoder` | 独立音色编码器 |
| AceStepDiTLayer + 包装 | `AceStepDiT` | 纯去噪预测 |
| AudioTokenDetokenizer | `AceStepAudioDecoder` | 独立音频解码 |
| diffusers.AutoencoderOobleck | `AceStepVAE` | 独立 VAE |

**检查清单**：
- [ ] DiT forward 只接收张量输入，不接收原始文本/音频
- [ ] 每个条件编码器独立可测试
- [ ] VAE 可以独立编解码，不依赖其他组件
- [ ] 模型文件中没有训练损失计算逻辑
- [ ] 模型文件中没有采样循环逻辑

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/model-code-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/model-code-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

**不要跳过报告更新** — 即使某个 Step 被跳过或失败，也必须记录到报告中。报告是执行过程的唯一可追溯记录。

### 0. 读取蓝图信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**每个 skill 执行的第一步，强制要求。** 从蓝图报告中读取 Python 运行环境信息和本 skill 必要的信息。

```bash
MODEL_NAME="{model-name}"
BLUEPRINT_PATH="packages/${MODEL_NAME}/.sisyphus/integration-blueprints/${MODEL_NAME}-blueprint.md"
```

**本 skill 必须从蓝图报告中读取的信息：**

| 蓝图信息 | 用途 |
|---------|------|
| 基本信息表中的 `Conda 环境名称` | 环境验证 |
| 基本信息表中的 `目标库路径` | 代码参考 |
| 模型组件清单表 | DiT/VAE/TextEncoder 等每个组件的类名、`__init__` 签名、`forward` 签名、代码来源 |
| 外部依赖分析 | 每个组件是否依赖 diffusers/transformers |
| 接入类型 | new_series / version_upgrade |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
# 从蓝图或上下文获取模型名称
MODEL_NAME="{model-name}"

# 检查并确保 DiffSynth-Studio 存在
if [ ! -d "packages/${MODEL_NAME}/DiffSynth-Studio" ]; then
    git clone https://github.com/modelscope/DiffSynth-Studio.git packages/${MODEL_NAME}/DiffSynth-Studio/
fi

# 设置路径变量
diffsynth_root="packages/${MODEL_NAME}/DiffSynth-Studio"
target_path="packages/${MODEL_NAME}/{target-library}"

export EXEC_LOG_DIR="packages/${MODEL_NAME}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_model_code"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "diffsynth-model-code",
  "model_name": "${MODEL_NAME}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "model_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "user_checks": []
}
EOF

ln -sfn ${EXEC_LOG_DIR} packages/${MODEL_NAME}/.sisyphus/execution-logs/latest
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始接入模型代码前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/model-code-plan.md`。** 基于蓝图报告中的模型组件清单，明确接入类型、组件接入方案（每个组件的来源、拷贝路径、重构点）、注册方案和一致性测试方案。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始编写代码。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 3. 确定接入方式（为每个组件制定接入方案）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**⚠️ 重要：执行此步骤前，必须阅读本节引用的参考文件** —— 本节包含简化的决策规则，详细操作步骤和代码示例在各参考文件中。

#### 3.1 从整合蓝图读取输入

- **接入类型**：`new_series`（新系列）或 `version_upgrade`（版本升级）
- **模型组件清单**：每个组件的类名、`__init__` 签名、`forward` 签名、代码来源
- **外部依赖分析**：确定组件代码来源（目标库自有、diffusers、transformers）

#### 3.2 统一决策树（按优先级顺序判断）

```
Step 1: 接入类型是什么？
├── new_series → 进入 Step 2
└── version_upgrade → 检查新组件的 __init__ 和 forward 签名
    ├── 与已有组件完全相同 → 走【完全复用】方案
    └── 有变化 → 进入 Step 2

Step 2: 组件的代码来源是什么？
├── 目标库自有源码 → 【自有源码】方案
├── diffusers 依赖 → 【diffusers 重构】方案
├── transformers 依赖 → 【transformers 包装】方案
└── version_upgrade 且有结构变化 → 【结构变化】方案
```

| 决策路径 | 判定条件 | 必须阅读的参考文件 |
|---------|---------|------------------|
| **完全复用** | `version_upgrade` + `__init__/forward` 与已有组件完全相同 | [model-file-reuse.md](references/model-file-reuse.md) |
| **自有源码** | `new_series` + 目标库自有实现 | [model-file-own-source.md](references/model-file-own-source.md) |
| **diffusers 重构** | `new_series` + 依赖 diffusers（VAE/UNet） | [model-file-diffusers-refactor.md](references/model-file-diffusers-refactor.md) |
| **transformers 包装** | `new_series` + 依赖 transformers（TextEncoder） | [model-file-transformers-wrapper.md](references/model-file-transformers-wrapper.md) |
| **结构变化** | `version_upgrade` + 新增模块/参数变化 | [model-file-version-upgrade.md](references/model-file-version-upgrade.md) |

#### 3.3 制定组件接入方案（每个组件必须输出）

**必须为每个组件制定接入方案**，格式如下：

```markdown
## 组件接入方案: {组件名}

### 决策依据
- **接入类型**: {new_series / version_upgrade}
- **代码来源**: {目标库自有 / diffusers / transformers}
- **签名对比结果**: {与已有组件相同 / 有变化}
- **结构变化**: {无 / 有（说明具体变化）}

### 选定的接入方式
**接入方式**: {完全复用 / 自有源码 / diffusers 重构 / transformers 包装 / 结构变化}

### 参考文件（已阅读）
- [ ] [model-file-{xxx}.md](references/model-file-{xxx}.md) — 已阅读并理解操作步骤

### 执行动作
| 步骤 | 动作 | 对应 Workflow Step | 状态 |
|-----|------|------|------|
| 1 | 阅读参考文件 | Step 2 | ⬜ |
| 2 | 创建/修改模型文件 | Step 2 | ⬜ |
| 3 | **初始化参数验证（打印目标库真实参数 → 对比 DiffSynth __init__）** | Step 3 | ⬜ 必须 |
| 4 | 编写 state_dict_converter（如需要） | Step 4 | ⬜ |
| 5 | 注册到 model_configs.py | Step 5 | ⬜ |
| 6 | ModelConfig + ModelPool 加载测试 | Step 6 | ⬜ 必须 |
| 7 | Forward 一致性测试 | Step 6 | ⬜ 必须 |

### 文件路径规划
- **模型文件**: `diffsynth/models/{series}_{component}.py`
- **Converter 文件**: `diffsynth/utils/state_dict_converters/{series}_{component}.py`（如需要）

### 测试要求
- **ModelConfig + ModelPool 加载测试**: {✅ 必须 / 如适用}
- **Forward 一致性测试**: {✅ 必须 / 如适用}
```

#### 3.4 ⚠️ 强制性要求

1. **必须阅读参考文件**：在为组件执行具体操作前，必须先阅读对应的参考文件，理解详细步骤
2. **必须制定接入方案**：每个组件都必须制定接入方案，作为执行依据
3. **必须执行测试**：无论哪种接入方式，ModelConfig + ModelPool 加载测试和 Forward 一致性测试都必须执行（永不跳过），详见 Step 8

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 4. 编写模型源代码

> 📖 开始前：重读本步骤描述，确认流程与报告路径

根据 Step 1 制定的接入方案，阅读对应的参考文件并按其指引接入：

| 接入方式 | 参考文件 |
|---------|---------|
| 完全复用 | [model-file-reuse.md](references/model-file-reuse.md) |
| 自有源码 | [model-file-own-source.md](references/model-file-own-source.md) + [attention-replacement.md](references/attention-replacement.md) + [gradient-checkpointing.md](references/gradient-checkpointing.md) |
| diffusers 重构 | [model-file-diffusers-refactor.md](references/model-file-diffusers-refactor.md) |
| transformers 包装 | [model-file-transformers-wrapper.md](references/model-file-transformers-wrapper.md) |
| 结构变化 | [model-file-version-upgrade.md](references/model-file-version-upgrade.md) |

文件命名统一为：`diffsynth/models/{series}_{component}.py`

**核心原则**：
- 优先直接复制代码，除非要解决依赖问题，否则不建议修改代码逻辑
- 只做必要的改动，版本升级时如果模型结构不变，只需注册新 hash + extra_kwargs
- 向后兼容是前提，任何修改都不能改变已有模型的 `forward` 逻辑和输出
- 不要添加代码来源相关的注释（如 "copied from xxx"），只保留必要的代码解释注释

**DiT 必须声明 `_repeated_blocks`**：在 DiT 类体第一行声明 `_repeated_blocks = ["{RepeatedBlockClass}"]`，值是重复 transformer block 的**类名字符串**（不是属性名，也不是模块路径），有多种重复 block 时全部列出。`torch.compile` 用它做 regional 编译；缺失则退化为整模编译。

⚠️ 框架用 `submod.__class__.__name__` 做字符串相等匹配，类名写错不会报错、不编译任何模块，但仍打印编译成功日志。写完必须与实际 block 类名逐字核对。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 5. 初始化参数验证（每个组件必须执行）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**⚠️ 强制步骤。编写完 DiffSynth 模型文件后，必须先在目标库中打印模型初始化参数，验证 `__init__` 默认值是否正确。**

**为什么必须？**：模型 `__init__` 参数填错是最高频的接入错误——参数名拼写错误、数值猜错、漏传必填参数。通过打印目标库真实参数，与 DiffSynth 模型 `__init__` 默认值对比，可以在 forward 测试前发现问题。

#### 5.1 在目标库中打印真实参数

在 `{target_path}` 下执行初始化脚本，打印每个模型组件的 `__init__` 参数：

```python
# 在 {target_path} 下创建并运行 print_model_init.py
"""打印目标库模型的初始化参数，作为 DiffSynth 模型接入的权威参考。"""
import torch

# ===== 根据目标库实际加载方式初始化模型 =====
# 示例 1：从 config 创建模型
from transformers import AutoConfig
config = AutoConfig.from_pretrained("{config_path}", trust_remote_code=True)
model = YourModelClass(config)

# 示例 2：从 checkpoint 直接创建
model = YourModelClass.from_pretrained("{checkpoint_path}")

# 示例 3：手动创建
model = YourModelClass(param1=value1, param2=value2)

# ===== 打印初始化参数 =====
print(f"\n=== {model.__class__.__name__} 初始化参数 ===")
for name, value in model.config.to_dict().items():
    print(f"  {name} = {value!r}")
# 如果模型没有 config.to_dict()，直接打印 __init__ 参数：
print(f"\n=== {model.__class__.__name__} 属性值 ===")
for name in ["hidden_size", "num_layers", "num_heads", "ffn_embed_dim", ...]:
    if hasattr(model, name):
        val = getattr(model, name)
        print(f"  {name} = {val!r}")
```

#### 5.2 对比 DiffSynth 模型 __init__ 签名

将 Step 2 编写的 DiffSynth 模型 `__init__` 参数与 3.1 的输出逐行对比：

```python
class YourSeriesDiT(nn.Module):
    def __init__(
        self,
        # 每个参数的默认值必须与目标库打印结果一致
        hidden_size=2048,        # 目标库: hidden_size = 2048
        num_layers=28,           # 目标库: num_layers = 28
        num_heads=16,            # 目标库: num_heads = 16
        ...
    ):
```

**对比检查项**：
- [ ] 参数名拼写与目标库完全一致
- [ ] 数值型参数（int/float/list）的默认值与目标库打印结果完全一致
- [ ] 布尔型参数的默认值与目标库一致
- [ ] 没有遗漏目标库中的必填参数
- [ ] 没有凭空添加目标库中不存在的参数
- [ ] `extra_kwargs` 中传入的参数在 `__init__` 中有对应

**常见错误模式**：
- `num_attention_heads` vs `num_heads` vs `n_heads` — 必须与目标库一致
- `hidden_size` vs `d_model` vs `dim` — 必须与目标库一致
- list 参数顺序或值猜错 — 必须从目标库打印结果获取
- 布尔参数默认值设反 — 如 `bias=True` 但目标库是 `False`

#### 5.3 将打印结果保存到报告

将 3.1 的输出保存到执行日志目录，作为接入依据：

```bash
# 保存目标库模型初始化参数
python {target_path}/print_model_init.py > ${EXEC_LOG_DIR}/outputs/target_model_init.txt
cat ${EXEC_LOG_DIR}/outputs/target_model_init.txt
```

如果对比发现不一致，修正 DiffSynth 模型文件的 `__init__` 默认值后重新验证。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 6. 编写 state_dict_converter（如需要）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

文件命名：`diffsynth/utils/state_dict_converters/{series}_{component}.py`

**何时需要**：原始 checkpoint 的 key 格式与 DiffSynth 模型的 `state_dict().keys()` 不一致。
**何时不需要**：key 格式已一致，或新版本只是参数变化、key 格式不变 → 复用已有 converter。

详细模板见 [references/converter-template.md](references/converter-template.md)，必须阅读。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 7. 注册到 model_configs.py

> 📖 开始前：重读本步骤描述，确认流程与报告路径

编辑 `{diffsynth_root}/diffsynth/configs/model_configs.py`。

每个注册条目的固定格式：

```python
{
    "model_hash": "由 diffsynth.core.loader.file.hash_model_file 计算",
    "model_name": "{component_name}",          # fetch_model 的检索键，惯例上与文件名一致但不强制
    "model_class": "diffsynth.models.{series}_{component}.{ClassName}",
    "extra_kwargs": {...},                      # 可选，用于区分同架构不同参数
    "state_dict_converter": "diffsynth.utils.state_dict_converters.{series}_{component}.{ConverterName}",  # 可选
}
```

`model_name` 惯例上取 `models/` 下的文件名（不含 `.py`），但它本质是 `fetch_model(model_name)` 的检索键，与文件名无强绑定——上游就有 `qwen_image_blockwise_controlnet` 定义在 `qwen_image_controlnet.py`、`wan_video_dit` 对应 `wan_video_dit.WanModel` 这类情况。

**⚠️ 注意：每个注册条目必须带有 ModelConfig 示例注释**

**每个注册条目的第一行必须包含 `# Example: ModelConfig(...)` 注释。** 这是仓库约定，全部注册条目都遵守。

```python
# ✅ 正确：有完整的 ModelConfig 注释
{
    # Example: ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/model.safetensors")
    "model_hash": "ba29d8bddbb6ace65675f6a757a13c00",
    "model_name": "ace_step_dit",
    "model_class": "diffsynth.models.ace_step_dit.AceStepDiTModel",
    "state_dict_converter": "diffsynth.utils.state_dict_converters.ace_step_dit.AceStepDiTModelStateDictConverter",
}

# ❌ 错误：缺少 ModelConfig 注释
{
    "model_hash": "ba29d8bddbb6ace65675f6a757a13c00",
    "model_name": "ace_step_dit",
    "model_class": "diffsynth.models.ace_step_dit.AceStepDiTModel",
    "state_dict_converter": "diffsynth.utils.state_dict_converters.ace_step_dit.AceStepDiTModelStateDictConverter",
}

# ❌ 错误：注释为空
{
    # Example:
    "model_hash": "a166c33455cdbd89c0888a3645ca5c0f",
    "model_name": "qwen_image_image2lora_coarse",
    "model_class": "diffsynth.models.qwen_image_image2lora.QwenImageImage2LoRAModel",
}
```

**填写原则**：

| 字段 | 来源 | 说明 |
|------|------|------|
| `model_id` | 目标 checkpoint 所在的 ModelScope/HF 仓库 ID | 如 `"ACE-Step/Ace-Step1.5"`、`"Wan-AI/Wan2.1-T2V-14B"` |
| `origin_file_pattern` | 目标 checkpoint 在仓库内的相对路径（支持 glob） | 如 `"transformer/diffusion_pytorch_model*.safetensors"`（分片）、`"vae/diffusion_pytorch_model.safetensors"`（单文件）、`"models_t5_umt5-xxl-enc-bf16.pth"` |

**如何确定 `origin_file_pattern`**：
1. 查看目标仓库的文件结构（ModelScope/HF 页面上可见）
2. 如果 checkpoint 是**分片文件**（如 `model-00001-of-00003.safetensors`），使用 `*` 通配符：`"text_encoder/model*.safetensors"`
3. 如果 checkpoint 是**单个文件**，写完整相对路径：`"vae/diffusion_pytorch_model.safetensors"`
4. 如果多个注册条目共享**同一个 checkpoint 文件**（如 LTX-2 单个 `.safetensors` 同时包含 DiT、VAE、text encoder），它们的 `origin_file_pattern` **相同**

**为什么必须填写**：
- 这是仓库约定，20 个系列的注册条目都带该注释，缺失会在 review 中被指出
- 注释是给使用者的文档，告知"用什么 `ModelConfig` 加载这个模型"，也是 `origin_file_pattern` 唯一的可读记录
- 注意注释**不参与加载逻辑**：`ModelPool.auto_load_model()` 只用 `hash_model_file` 算出的 `model_hash` 去匹配 `MODEL_CONFIGS`。所以加载失败时不要往注释上找原因

**extra_kwargs 填写原则**：

**核心原则：首次接入时 extra_kwargs 为空，让模型使用默认值。后续 variant 只传入与首次接入不同的参数。**

模型类的 `__init__` 默认值应该设为**首次接入模型**的实际参数。这样做的好处：
- 首次接入时，`extra_kwargs` 为空 `{}`，模型完全使用自身默认值
- 后续接入同系列不同规格的模型时，只需传入变化的参数，注册条目保持简洁
- 默认值就是该系列的基准配置，后续 variant 通过 extra_kwargs 覆盖差异部分

**示例**：

```python
# 首次接入：模型类默认值 = 实际参数
class AceStepVAE(nn.Module):
    def __init__(
        self,
        encoder_hidden_size=128,
        downsampling_ratios=[2, 4, 4, 6, 10],   # 首次接入的实际值
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128,
        decoder_input_channels=64,
        audio_channels=2,
        sampling_rate=48000,                      # 首次接入的实际值
    ):
        ...

# ✅ 首次接入：extra_kwargs 为空
{
    "model_name": "ace_step_vae",
    "model_class": "diffsynth.models.ace_step_vae.AceStepVAE",
    "extra_kwargs": {},  # 首次接入，使用模型默认值
}

# 后续接入 variant 时，只传入与首次接入不同的参数
# 假设有个 VAE v2，只有 downsampling_ratios 和 sampling_rate 不同
{
    "model_name": "ace_step_vae_v2",
    "model_class": "diffsynth.models.ace_step_vae.AceStepVAE",
    "extra_kwargs": {
        "downsampling_ratios": [2, 4, 4, 8, 8],  # 与首次接入的 [2,4,4,6,10] 不同
        "sampling_rate": 44100,                    # 与首次接入的 48000 不同
    },
}
```

**如何计算 model_hash**：

使用 DiffSynth 内置的 `hash_model_file` 函数（位于 `diffsynth/core/loader/file.py`），对 `ModelConfig.path` 计算：

```python
from diffsynth.core.loader.config import ModelConfig
from diffsynth.core.loader.file import hash_model_file

model_config = ModelConfig(
    model_id="{model_id}",
    origin_file_pattern="{pattern}",  # 如 "transformer/diffusion_pytorch_model*.safetensors"
)
model_config.skip_download = True
model_config.download_if_necessary()

# model_config.path 可能是单个路径字符串，也可能是文件列表（分片 checkpoint）
# hash_model_file 内部已处理两种情况，直接传入即可
model_hash = hash_model_file(model_config.path)
```

这个 hash 不是文件的 MD5，而是**对 state dict 的 key 名和 shape 排序后计算的 MD5**。相同架构的文件（key 名和 shape 相同）会得到相同的 hash。

**new_series**：
1. 在 `model_configs.py` 末尾新建 `{series}_series = [...]` 列表，紧跟当前最后一个 series 列表之后
2. **必须**把它追加到文件末尾的 `MODEL_CONFIGS` 元组中（漏掉不会有静态报错，但加载时 hash 永远匹配不到）

**version_upgrade**：在对应 series 列表中添加新条目，`MODEL_CONFIGS` 无需改动。

**注意**：同一个 checkpoint 文件可以有多个注册条目（不同 model_name 对应不同组件），如 LTX-2 的单个 `.safetensors` 文件同时包含 DiT、VAE、text encoder 的权重。

#### 完全复用的注册方式

对于判定为**完全复用**的组件：无需创建新的模型文件或 converter，只需计算 hash 并在 `model_configs.py` 中添加新条目（复用已有 `model_class` 和 `state_dict_converter`）。详细流程和验证清单见 [model-file-reuse.md](references/model-file-reuse.md)。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 8. 一致性验证（核心：验证与原始代码输出一致）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**核心目标**：Forward 一致性测试不是验证"模型能跑"，而是验证**DiffSynth 代码与目标库原始代码的输出完全一致**。

**为什么必须完全一致？**
- 扩散模型推理是累积过程，微小的 forward 误差会在多步采样中被放大
- 即使第 1 步误差很小，第 50 步后输出可能完全不同
- 只有与原始代码完全一致，才能保证与官方权重、官方示例结果一致

**测试类型说明**：

| 测试类型 | 核心目标 | 验证内容 |
|---------|---------|---------|
| **ModelConfig + ModelPool 加载测试** | 验证注册链正确 | ModelConfig 路径解析 → ModelPool.auto_load_model → MODEL_CONFIGS 匹配 → converter 导入 → 模型实例化 → 权重加载 |
| **Forward 一致性测试** | **验证与原始代码输出一致** | 使用相同输入，对比原始代码和 DiffSynth 代码的 forward 输出 |

**⚠️ 关键区分**：
- ❌ **不是**：验证 forward "能运行"、不报错
- ✅ **是**：验证 forward 输出 **==** 原始代码输出（在数值精度范围内）

**所有情况都必须执行两种测试，永不跳过**：

| 接入方式 | ModelConfig + ModelPool 加载测试 | Forward 一致性测试 | 原因 |
|---------|------------------|-------------------|------|
| `自有源码` | ✅ 必须 | ✅ 必须 | 新代码，需验证与原始代码输出一致 |
| `diffusers 重构` | ✅ 必须 | ✅ 必须 | 重构代码，需验证与原始 diffusers 输出一致 |
| `transformers 包装` | ✅ 必须 | ✅ 必须 | 包装类，需验证与原始 transformers 输出一致 |
| `完全复用` | ✅ 必须 | ✅ 必须 | 验证不同 checkpoint 的加载和输出 |
| `结构变化` | ✅ 必须 | ✅ 必须 | 修改后的代码，需验证与原始代码输出一致 |

**⚠️ 永不跳过的原因**：
- **不同 checkpoint**：即使是相同架构，不同 checkpoint 也可能有细微差异
- **参数变化**：extra_kwargs 变化后，需验证参数传递正确性
- **权重加载**：converter 可能有 bug，必须通过 forward 验证

**Forward 一致性测试的严格标准**：

| 指标 | 阈值 | 说明 |
|------|------|------|
| max_diff | < 1e-5 | 最大元素差异 |
| mean_diff | < 1e-6 | 平均元素差异 |
| rel_diff | < 1e-5 | 相对差异（考虑数值范围）|
| 形状 | 必须完全一致 | shape、dtype、device |
| 非 NaN/Inf | 必须无 NaN/Inf | 数值合法性 |

**重要**：如果对比不通过，必须排查原因，不能降低阈值蒙混过关。常见原因：
- 模型代码有 bug（如 attention 实现错误）
- converter 映射错误（如 key 对应错）
- 输入准备不一致（如 random seed 不同）
- 权重加载问题（如部分参数未加载）

**通过 `ModelConfig` + `ModelPool` 验证的完整链路**：
1. `ModelConfig(model_id=..., origin_file_pattern=...)` → 解析模型文件路径（支持多分片）
2. `model_config.skip_download = True` → 跳过下载（模型已在环境准备阶段下载）
3. `model_config.download_if_necessary()` → 仅解析路径，不下载
4. `model_pool = ModelPool()` → 创建模型池
5. `model_pool.auto_load_model(model_config.path)` → 验证完整注册链：path → hash → MODEL_CONFIGS → model_class → 实例化 → 权重加载
6. `model_pool.fetch_model("{model_name}")` → 获取加载的模型

#### 8a. ModelPool 加载测试（前置验证）

**目的**：在运行一致性测试之前，先通过 ModelConfig + ModelPool 验证注册链正确性，提前解决模型读取问题。

**测试脚本**：

```bash
cat > ${EXEC_LOG_DIR}/scripts/test_modelpool_load.py << 'SCRIPT'
#!/usr/bin/env python3
"""
ModelPool 加载测试
目的：在一致性测试之前，先验证注册链正确性，提前解决模型读取问题。

验证链路：ModelConfig 路径解析 → ModelPool.auto_load_model → MODEL_CONFIGS 匹配
         → model_class 实例化 → state_dict_converter 导入 → 权重加载 → 参数量
"""

import sys
import os
import torch
import json
import gc
from datetime import datetime

LOG_FILE = "${EXEC_LOG_DIR}/outputs/modelpool_load.log"
RESULT_FILE = "${EXEC_LOG_DIR}/checkpoints/modelpool_load_result.json"


def log(msg):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    log("=" * 60)
    log("ModelPool 加载测试")
    log("=" * 60)

    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "FAILED",
    }

    try:
        from diffsynth.core.loader.config import ModelConfig
        from diffsynth.core.loader.file import hash_model_file
        from diffsynth.models.model_loader import ModelPool

        # ===== 1. ModelConfig 解析路径 =====
        log("\nStep 1: 解析 ModelConfig 路径...")
        model_config = ModelConfig(
            model_id="{model_id}",           # 如 "Qwen/Qwen-Image-2512"
            origin_file_pattern="{pattern}",  # 如 "transformer/diffusion_pytorch_model*.safetensors"
        )
        model_config.skip_download = True
        model_config.download_if_necessary()
        log(f"  文件路径: {model_config.path}")

        # ===== 2. 计算 model_hash =====
        log("\nStep 2: 计算 model_hash...")
        model_hash = hash_model_file(model_config.path)
        log(f"  model_hash: {model_hash}")

        # ===== 3. 通过 ModelPool 加载模型 =====
        log("\nStep 3: ModelPool.auto_load_model...")
        model_pool = ModelPool()
        model_pool.auto_load_model(model_config.path)
        log("  ✅ ModelPool.auto_load_model 成功")

        # ===== 4. 获取模型 =====
        log("\nStep 4: 获取模型...")
        model = model_pool.fetch_model("{model_name}")
        log(f"  模型类型: {model.__class__.__name__}")
        log(f"  模型模块: {model.__class__.__module__}")

        # ===== 5. 参数量统计 =====
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nStep 5: 参数量统计...")
        log(f"  总参数量: {total_params:,}")
        log(f"  可训练参数: {trainable_params:,}")

        # ===== 6. 权重加载验证 =====
        model_keys = set(model.state_dict().keys())
        log(f"\nStep 6: 权重 key 数量: {len(model_keys)}")

        # 检查是否有 key 未加载（值为 NaN 或全部为 0 的异常参数）
        abnormal_count = 0
        for name, param in model.named_parameters():
            if param.isnan().any() or param.isinf().any():
                abnormal_count += 1
                log(f"  ⚠️ 异常参数: {name}")
        if abnormal_count == 0:
            log("  ✅ 所有权重无 NaN/Inf 异常")

        result["status"] = "PASSED"
        result["model_name"] = "{model_name}"
        result["model_hash"] = model_hash
        result["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
        result["total_params"] = total_params
        result["trainable_params"] = trainable_params
        result["weight_keys_count"] = len(model_keys)
        result["abnormal_params"] = abnormal_count

        log("\n" + "=" * 60)
        log("✅ PASSED: ModelPool 加载成功")
        log("=" * 60)

    except Exception as e:
        log(f"❌ ModelPool 加载失败: {e}")
        import traceback
        log(traceback.format_exc())
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    # 保存结果
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"\n结果已保存: {RESULT_FILE}")

    # 清理
    clear_gpu_memory()

    return 0 if result.get("status") == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/test_modelpool_load.py
```

**执行测试**：

```bash
cd {diffsynth_root}
python ${EXEC_LOG_DIR}/scripts/test_modelpool_load.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/modelpool_load.log
```

**查看结果**：

```bash
cat ${EXEC_LOG_DIR}/checkpoints/modelpool_load_result.json
```

**通过标准**：
- ModelConfig 路径解析成功
- model_hash 计算成功
- ModelPool.auto_load_model 不报错
- fetch_model 获取到正确类型的模型实例
- 参数量符合预期
- 无 NaN/Inf 异常参数

**如果失败**：排查 model_configs.py 注册条目、model_class 路径、state_dict_converter 导入、extra_kwargs 参数名。确认修复后再运行。

#### 8b. 运行统一测试脚本

使用 `test_forward_consistency.py` 脚本（hook-based 逐层对比设计）。

**测试脚本**：

```bash
cat > ${EXEC_LOG_DIR}/scripts/test_forward_consistency.py << 'SCRIPT'
#!/usr/bin/env python3
"""
Forward 一致性测试（逐层对比定位）
核心目标：验证 DiffSynth 代码输出 == 原始代码输出
通过 forward hook 捕获每一层的中间输出，一次性完成整体输出对比 + 逐层差异定位

执行流程：
1. 原始模型：注册 hook → forward → 捕获所有中间层输出 + 最终输出
2. 卸载原始模型 → 清理显存
3. DiffSynth 模型：注册 hook → forward → 捕获所有中间层输出 + 最终输出
4. 卸载 DiffSynth 模型 → 清理显存
5. 对比最终输出（整体一致性）
6. 逐层对比所有中间输出（定位差异来源）
"""

import sys
import os
import torch
import json
import gc
import numpy as np
from datetime import datetime

# 配置
LOG_FILE = "${EXEC_LOG_DIR}/outputs/forward_consistency.log"
RESULT_FILE = "${EXEC_LOG_DIR}/checkpoints/forward_consistency_result.json"
LAYER_RESULTS_FILE = "${EXEC_LOG_DIR}/checkpoints/layer_comparison.json"

# 严格阈值（必须满足）
FINAL_THRESHOLDS = {
    "max_diff": 1e-5,      # 最大元素差异
    "mean_diff": 1e-6,     # 平均差异
    "rel_diff": 1e-5,      # 相对差异
}

# 逐层对比阈值（放宽一些，中间层可能有数值放大）
LAYER_THRESHOLDS = {
    "max_diff": 1e-3,      # 中间层允许更大差异
    "rel_diff": 1e-3,
}


def log(msg):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def clear_gpu_memory():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LayerOutputCapture:
    """Forward hook 捕获器，记录每个叶子模块的输入和输出"""

    def __init__(self):
        self.handles = []
        self.layer_outputs = {}

    def _make_hook(self, name):
        def hook(module, input, output):
            # 只捕获叶子模块（无子模块的模块）
            if len(list(module.children())) == 0:
                self.layer_outputs[name] = {
                    "output": output.detach().cpu(),
                    "output_shape": list(output.shape),
                    "module_type": module.__class__.__name__,
                }

        return hook

    def register(self, model):
        """为模型所有叶子模块注册 hook"""
        for name, module in model.named_modules():
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def remove(self):
        """移除所有 hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_model_with_hooks(model, input_data):
    """
    运行模型 forward，同时捕获所有中间层输出
    返回: (final_output_cpu, {layer_name: {output, output_shape, module_type}})
    """
    capturer = LayerOutputCapture()
    capturer.register(model)

    with torch.no_grad():
        input_cuda = input_data.cuda()
        final_output = model(input_cuda)

    final_cpu = final_output.cpu()
    layer_outputs = capturer.layer_outputs
    capturer.remove()

    return final_cpu, layer_outputs


def compare_tensor(a, b, thresholds):
    """对比两个 tensor，返回指标和是否通过"""
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}

    diff = (a.float() - b.float()).abs()

    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    original_range = float(a.float().max()) - float(a.float().min())
    rel_diff = max_diff / original_range if original_range > 0 else 0.0

    passed = (
        max_diff < thresholds["max_diff"] and
        (rel_diff < thresholds["rel_diff"] if original_range > 0 else True)
    )

    return {
        "status": "passed" if passed else "failed",
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
        "original_range": original_range,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
    }


def summarize_layer_comparison(layer_names, original_layers, diffsynth_layers, thresholds):
    """
    对比指定层列表的中间输出
    返回: {layer_name: comparison_result}
    """
    results = {}
    for name in layer_names:
        if name not in original_layers or name not in diffsynth_layers:
            continue

        orig = original_layers[name]["output"]
        ds = diffsynth_layers[name]["output"]

        results[name] = {
            "module_type": original_layers[name]["module_type"],
            **compare_tensor(orig, ds, thresholds),
        }

    return results


def find_first_divergent_layer(layer_results, max_diff_threshold=1e-5):
    """
    找出第一个出现显著差异的层（按名称顺序扫描）
    返回: (layer_name, result_info) 或 (None, None)
    """
    for name, info in layer_results.items():
        if info.get("status") == "shape_mismatch":
            return name, info
        if info.get("max_diff", 0) > max_diff_threshold:
            return name, info
    return None, None


def main():
    log("=" * 60)
    log("Forward 一致性测试（逐层对比定位）")
    log("=" * 60)
    log(f"最终输出阈值: {FINAL_THRESHOLDS}")
    log(f"逐层对比阈值: {LAYER_THRESHOLDS}")
    log(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

    result = {
        "timestamp": datetime.now().isoformat(),
        "final_thresholds": FINAL_THRESHOLDS,
        "layer_thresholds": LAYER_THRESHOLDS,
        "status": "FAILED",
    }

    try:
        # ===== 1. 准备输入 =====
        log("\n准备输入数据...")
        torch.manual_seed(42)
        np.random.seed(42)

        # TODO: 根据模型输入要求准备输入
        # input_data = torch.randn(1, 3, 224, 224)  # 示例
        input_data = None  # 占位符，运行时替换
        log(f"输入形状: {input_data.shape}")

        # ===== 2. 原始模型：hook 捕获所有中间层 =====
        log("\n" + "=" * 60)
        log("Step 1: 加载原始模型（注册 hook）")
        log("=" * 60)

        # TODO: 根据目标库实际的模型加载方式填写
        # from {target_library} import {ModelClass}
        # model_orig = {ModelClass}(...)
        # checkpoint = torch.load("{checkpoint_path}", map_location="cuda")
        # model_orig.load_state_dict(checkpoint)
        # model_orig = model_orig.cuda().eval()

        model_orig = None  # 占位符

        log(f"注册 hook 到 {sum(1 for _ in model_orig.named_modules())} 个模块...")
        orig_final, orig_layers = run_model_with_hooks(model_orig, input_data)
        log(f"✅ 原始模型完成，捕获 {len(orig_layers)} 个中间层输出")
        log(f"   最终输出形状: {orig_final.shape}, 范围: [{orig_final.min():.6f}, {orig_final.max():.6f}]")

        # 卸载原始模型
        del model_orig
        clear_gpu_memory()
        log("   已卸载原始模型，清理显存")

        # ===== 3. DiffSynth 模型：hook 捕获所有中间层 =====
        log("\n" + "=" * 60)
        log("Step 2: 加载 DiffSynth 模型（注册 hook）")
        log("=" * 60)

        # TODO: 使用 ModelConfig 解析路径 + ModelPool 加载模型（验证完整注册链）
        # ModelConfig 通过 origin_file_pattern 匹配所有分片文件（如 "transformer/diffusion_pytorch_model*.safetensors"）
        # ModelPool.auto_load_model(model_config.path) 验证注册链：path → hash → MODEL_CONFIGS → model_class → 实例化
        from diffsynth.core.loader.config import ModelConfig
        from diffsynth.models.model_loader import ModelPool

        model_config = ModelConfig(
            model_id="{model_id}",           # 如 "Qwen/Qwen-Image-2512"
            origin_file_pattern="{pattern}",  # 如 "transformer/diffusion_pytorch_model*.safetensors"
        )
        model_config.skip_download = True
        model_config.download_if_necessary()

        # model_config.path 会被解析为文件路径列表（多个分片文件）或单文件路径
        log(f"模型文件路径: {model_config.path}")

        # 通过 ModelPool 加载（验证注册链正确性）
        model_pool = ModelPool()
        model_pool.auto_load_model(model_config.path)
        model_ds = model_pool.fetch_model("{model_name}")
        model_ds = model_ds.cuda().eval()

        log(f"注册 hook 到 {sum(1 for _ in model_ds.named_modules())} 个模块...")
        ds_final, ds_layers = run_model_with_hooks(model_ds, input_data)
        log(f"✅ DiffSynth 模型完成，捕获 {len(ds_layers)} 个中间层输出")
        log(f"   最终输出形状: {ds_final.shape}, 范围: [{ds_final.min():.6f}, {ds_final.max():.6f}]")

        # 卸载 DiffSynth 模型
        del model_ds
        clear_gpu_memory()
        log("   已卸载 DiffSynth 模型，清理显存")

        # ===== 4. 最终输出对比 =====
        log("\n" + "=" * 60)
        log("Step 3: 最终输出对比")
        log("=" * 60)

        final_result = compare_tensor(orig_final, ds_final, FINAL_THRESHOLDS)
        final_passed = final_result.get("status") == "passed"

        log(f"  最大差异:     {final_result.get('max_diff', 'N/A')}")
        log(f"  平均差异:     {final_result.get('mean_diff', 'N/A')}")
        log(f"  相对差异:     {final_result.get('rel_diff', 'N/A')}")
        log(f"  状态:         {'✅ PASSED' if final_passed else '❌ FAILED'}")

        # ===== 5. 逐层对比 =====
        log("\n" + "=" * 60)
        log("Step 4: 逐层中间输出对比")
        log("=" * 60)

        # 找出两个模型都有的层
        common_layers = sorted(set(orig_layers.keys()) & set(ds_layers.keys()))
        log(f"共同层数: {len(common_layers)} / {len(orig_layers)} (原始) / {len(ds_layers)} (DiffSynth)")

        layer_results = summarize_layer_comparison(common_layers, orig_layers, ds_layers, LAYER_THRESHOLDS)

        # 按差异从大到小排序
        layer_ranking = sorted(
            layer_results.items(),
            key=lambda x: x[1].get("max_diff", 0) if x[1].get("status") != "shape_mismatch" else float("inf"),
            reverse=True,
        )

        # 输出 Top 20 差异最大的层
        log("\nTop 20 差异最大的层:")
        log(f"  {'层名':<60} {'类型':<25} {'最大差异':>12} {'状态':>10}")
        log("  " + "-" * 110)

        for name, info in layer_ranking[:20]:
            if info.get("status") == "shape_mismatch":
                log(f"  {name:<60} {info.get('module_type', ''):<25} {'SHAPE_MISMATCH':>12} {'❌':>10}")
            else:
                log(f"  {name:<60} {info.get('module_type', ''):<25} {info.get('max_diff', 0):>12.2e} {'✅' if info.get('status') == 'passed' else '❌':>10}")

        # 找出第一个出现差异的层
        first_divergent, divergence_info = find_first_divergent_layer(layer_results)
        if first_divergent:
            log(f"\n⚠️ 第一个出现显著差异的层: {first_divergent}")
            log(f"   模块类型: {divergence_info.get('module_type', 'N/A')}")
            if divergence_info.get("status") == "shape_mismatch":
                log(f"   原始形状: {divergence_info.get('a_shape')}")
                log(f"   Diff形状: {divergence_info.get('b_shape')}")
            else:
                log(f"   最大差异: {divergence_info.get('max_diff'):.2e}")
                log(f"   输出形状: {divergence_info.get('a_shape')}")

        # 统计通过率
        total_layers = len(layer_results)
        passed_layers = sum(1 for v in layer_results.values() if v.get("status") == "passed")
        shape_mismatch = sum(1 for v in layer_results.values() if v.get("status") == "shape_mismatch")
        log(f"\n层对比通过率: {passed_layers}/{total_layers} ({passed_layers/total_layers*100:.1f}%)")
        if shape_mismatch > 0:
            log(f"  形状不匹配: {shape_mismatch} 层")

        # ===== 6. 保存结果 =====
        result["final_output"] = final_result
        result["layer_stats"] = {
            "total": total_layers,
            "passed": passed_layers,
            "failed": total_layers - passed_layers,
            "shape_mismatch": shape_mismatch,
        }
        result["first_divergent_layer"] = first_divergent

        # 判断整体通过：最终输出必须通过
        if final_passed:
            result["status"] = "PASSED"
            log("\n" + "=" * 60)
            log("✅ PASSED: DiffSynth 输出与原始代码一致")
            log("=" * 60)
        else:
            result["status"] = "FAILED"
            log("\n" + "=" * 60)
            log("❌ FAILED: 最终输出差异超出阈值")
            if first_divergent:
                log(f"🔍 差异源头可能在: {first_divergent}")
            log("\n排查建议:")
            log("1. 查看 Top 差异层的模块类型，定位到具体代码")
            log("2. 检查 attention、normalization 等容易引入数值差异的实现")
            log("3. 检查 state_dict_converter 权重映射是否正确")
            log("4. 检查输入预处理是否完全一致")
            log("5. 保存原始和 DiffSynth 的中间层 tensor 到磁盘，深入分析差异层")
            log("=" * 60)

    except Exception as e:
        log(f"❌ 测试失败: {e}")
        import traceback

        log(traceback.format_exc())
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["status"] = "ERROR"

    # 保存结果
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"\n结果已保存: {RESULT_FILE}")

    # 保存详细层对比结果（用于调试）
    # 注意：完整 tensor 可能很大，只保存对比结果摘要，避免文件过大
    layer_summary = {}
    for name, info in (layer_results if "layer_results" in dir() else {}).items():
        layer_summary[name] = info
    with open(LAYER_RESULTS_FILE, "w") as f:
        json.dump({"layers": layer_summary}, f, indent=2, default=str)
    log(f"层对比详情已保存: {LAYER_RESULTS_FILE}")

    return 0 if result.get("status") == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/test_forward_consistency.py
```

**执行测试脚本**：

```bash
cd {diffsynth_root}
python ${EXEC_LOG_DIR}/scripts/test_forward_consistency.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/forward_consistency.log
```

测试脚本会自动：
1. 加载原始模型，注册 hook，forward，捕获所有中间层 + 最终输出
2. 卸载原始模型，清理显存
3. 通过 ModelConfig + ModelPool 加载 DiffSynth 模型，注册 hook，forward，捕获所有中间层 + 最终输出
4. 对比最终输出（整体一致性）
5. 逐层对比所有中间输出（定位差异来源）
6. 输出 Top 20 差异层和第一个差异层

**查看结果**：

```bash
# 整体结果
cat ${EXEC_LOG_DIR}/checkpoints/forward_consistency_result.json

# 逐层对比详情
cat ${EXEC_LOG_DIR}/checkpoints/layer_comparison.json

# 完整日志
cat ${EXEC_LOG_DIR}/outputs/forward_consistency.log
```

**通过标准**：

| 指标 | 阈值 | 说明 |
|------|------|------|
| max_diff | < 1e-5 | 最终输出最大元素差异 |
| mean_diff | < 1e-6 | 最终输出平均元素差异 |
| rel_diff | < 1e-5 | 最终输出相对差异 |
| 形状 | 必须完全一致 | shape、dtype、device |
| 非 NaN/Inf | 必须无 NaN/Inf | 数值合法性 |
| 中间层 max_diff | < 1e-3 | 中间层允许更大差异 |

**如果测试失败**：
1. 查看日志中 "Top 20 差异最大的层" 和 "第一个出现显著差异的层"
2. 查看 `layer_comparison.json` 中具体差异层的模块类型

**测试设计说明**（供了解）：

- **逐层对比定位**：使用 forward hook 捕获每一层（叶子模块）的中间输出，一次性完成整体输出对比 + 逐层差异定位
- **单卡顺序执行**：先运行原始模型（hook 捕获全部中间层）→ 卸载 → 清理显存 → 运行 DiffSynth 模型（hook 捕获全部中间层）→ 对比
- **双层阈值**：最终输出用严格阈值（`max_diff < 1e-5`），中间层用放宽阈值（`max_diff < 1e-3`）
- **Top 20 差异排名 + 第一个差异层检测**：快速定位问题源头

**🔍 USER_CHECK: 一致性验证**

生成用户检查清单（使用 `>>` 追加确保不覆盖已有检查点）：

```bash
# 首次生成：创建文件头
cat > ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - diffsynth-model-code

执行时间: $(date -Iseconds)
模型: {model-name}

---
EOF

# 追加检查点
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - diffsynth-model-code

## [CHECK-001] ModelPool 加载验证（前置）
- **严重程度**: critical
- **描述**: 通过 ModelConfig 解析路径 + ModelPool.auto_load_model 验证注册链
- **验证内容**:
  - [ ] ModelConfig 路径解析成功
  - [ ] model_hash 计算正确
  - [ ] ModelPool.auto_load_model 成功
  - [ ] fetch_model 获取到正确类型的模型实例
  - [ ] 参数量符合预期
  - [ ] 无 NaN/Inf 异常参数
- **查看结果**: \`cat ${EXEC_LOG_DIR}/checkpoints/modelpool_load_result.json\`
- **状态**: ⬜ 待确认

## [CHECK-002] ModelConfig + ModelPool 注册链验证（关键）
- **严重程度**: critical
- **描述**: 通过 `ModelConfig` 解析路径 + `ModelPool.auto_load_model()` 验证完整注册链（支持多文件分片 checkpoint）
- **验证内容**:
  - [ ] `ModelConfig(model_id=..., origin_file_pattern=...)` 路径解析正确
  - [ ] `model_config.skip_download = True` 已设置
  - [ ] `model_config.download_if_necessary()` 成功
  - [ ] `model_pool = ModelPool()` 创建成功
  - [ ] `model_pool.auto_load_model(model_config.path)` 成功加载
  - [ ] hash_model_file 计算正确
  - [ ] MODEL_CONFIGS 中存在匹配条目
  - [ ] state_dict_converter 导入成功（如有）
  - [ ] 模型实例化成功（extra_kwargs 正确）
  - [ ] 参数量符合预期
- **状态**: ⬜ 待确认

## [CHECK-003] Forward 一致性（核心）
- **严重程度**: critical
- **描述**: 验证 DiffSynth 代码输出与原始代码输出完全一致
- **核心目标**: 不只是"能运行"，而是"输出数值完全相同"
- **测试脚本**: 统一脚本 `test_forward_consistency.py`（hook-based 逐层对比）
  - 通过 forward hook 捕获所有中间层输出，一次性完成整体输出对比 + 逐层差异定位
  - 输出：`forward_consistency_result.json`（整体结果）、`layer_comparison.json`（逐层对比详情）
- **查看结果**: `cat ${EXEC_LOG_DIR}/checkpoints/forward_consistency_result.json`（整体结果）
- **查看逐层对比**: `cat ${EXEC_LOG_DIR}/checkpoints/layer_comparison.json`（逐层差异详情）
- **查看日志**: `${EXEC_LOG_DIR}/outputs/forward_consistency.log`
- **通过标准**（必须全部满足）:
  - max_diff < 1e-5
  - mean_diff < 1e-6
  - rel_diff < 1e-5
  - 输出形状、类型完全一致
  - 无 NaN/Inf
- **失败排查**:
  1. 查看日志中 "Top 20 差异最大的层" 和 "第一个出现显著差异的层"
  2. 查看 `layer_comparison.json` 中具体差异层的模块类型
  3. 检查对应层的代码实现（attention、normalization 等）
  4. 检查 converter 权重映射是否匹配
  5. 检查输入准备（随机种子、预处理）是否完全一致
- **状态**: ⬜ 待确认

## [CHECK-004] 模型文件结构
- **严重程度**: info
- **描述**: 检查生成的模型文件结构是否正确
- **查看文件**: \`${diffsynth_root}/diffsynth/models/{series}_{component}.py\`
- **检查点**:
  - [ ] __init__ 签名与目标库一致
  - [ ] forward 签名与目标库一致
  - [ ] 已替换 attention 和 gradient checkpointing
- **状态**: ⬜ 待确认

## [CHECK-005] Converter 正确性
- **严重程度**: critical
- **描述**: state_dict_converter 是否正确映射权重 key
- **查看文件**: \`${diffsynth_root}/diffsynth/utils/state_dict_converters/{series}_{component}.py\`
- **验证**: ModelConfig + ModelPool.auto_load_model 加载成功即证明 converter 正确
- **状态**: ⬜ 待确认
EOF
```

### 9. 参数对比验证（一致性测试后的二次验证）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**⚠️ 此步骤在一致性验证通过后执行，用于二次确认接入代码的所有参数配置与目标库完全一致。** 一致性测试验证 forward 输出的数值一致性，本步骤验证参数配置的一致性——两者互补，前者检测运行时差异，后者检测静态配置差异。

**为什么需要此步骤？**：Forward 一致性通过不代表参数配置完全正确。可能存在：目标库通过配置文件动态计算的参数、extra_kwargs 中遗漏的参数、或某些参数虽然不影响 forward 输出但会导致训练/推理时行为不一致。

#### 9.1 提取目标库参数配置

在目标库中提取完整的模型参数配置，并标注每个参数的来源（config 文件、argparse、代码默认值、运行时推断）：

```python
# 在 {target_path} 下创建并运行 extract_all_params.py
"""提取目标库模型的完整参数配置，标注来源，作为对比的权威参考。"""
import torch
import json
import inspect
from pathlib import Path

# ===== 1. 初始化模型 =====
# TODO: 根据目标库实际加载方式初始化
# from {target_library} import {ModelClass}
# model = {ModelClass}(...)
model = None  # 占位符

# ===== 2. 提取 config 参数（来源: config 文件/JSON） =====
config_params = {}
if hasattr(model, "config"):
    if hasattr(model.config, "to_dict"):
        config_params = model.config.to_dict()
    elif hasattr(model.config, "__dict__"):
        config_params = model.config.__dict__
    elif isinstance(model.config, dict):
        config_params = model.config

# ===== 3. 提取模型属性（来源: __init__ 赋值） =====
model_attrs = {}
attr_names = [
    "hidden_size", "num_layers", "num_heads", "num_attention_heads",
    "ffn_embed_dim", "intermediate_size", "d_model", "dim",
    "num_kv_heads", "head_dim", "num_key_value_heads",
    "max_position_embeddings", "max_seq_len",
    "vocab_size", "num_embeddings",
    "dropout", "attention_dropout", "hidden_dropout",
    "bias", "use_bias",
    "eps", "layer_norm_eps",
    "activation", "hidden_act",
]
for name in attr_names:
    if hasattr(model, name):
        val = getattr(model, name)
        if not callable(val):
            model_attrs[name] = val

# ===== 4. 提取 __init__ 签名默认值（来源: 代码默认值） =====
init_defaults = {}
sig = inspect.signature(model.__class__.__init__)
for param_name, param in sig.parameters.items():
    if param_name == "self":
        continue
    if param.default is not inspect.Parameter.empty:
        init_defaults[param_name] = param.default

# ===== 5. 提取运行时推断的参数（来源: 运行时动态计算） =====
runtime_params = {}
if hasattr(model, "num_parameters"):
    runtime_params["total_parameters"] = model.num_parameters()
elif hasattr(model, "parameters"):
    runtime_params["total_parameters"] = sum(p.numel() for p in model.parameters())
# 添加其他运行时可推断的参数

# ===== 6. 按来源分类输出 =====
result = {
    "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}" if model else "N/A",
    "params_by_source": {
        "config_file": config_params,           # 来自 config.json / yaml
        "init_defaults": init_defaults,          # 来自 __init__ 默认值
        "model_attributes": model_attrs,          # 来自 model.xxx 属性
        "runtime_computed": runtime_params,       # 运行时推断
    },
    "all_params_merged": {**config_params, **init_defaults, **model_attrs, **runtime_params},
}

print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
```

保存输出到执行日志：
```bash
cd {target_path}
python extract_all_params.py > ${EXEC_LOG_DIR}/outputs/target_all_params.json 2>&1
```

#### 9.2 提取 DiffSynth 接入代码参数配置

从 DiffSynth 接入代码中提取相同维度的参数配置：

```python
# 在 {diffsynth_root} 下运行 extract_diffsynth_params.py
"""提取 DiffSynth 接入代码的完整参数配置。"""
import sys
import json
import inspect
import torch

sys.path.insert(0, ".")

# ===== 1. 导入 DiffSynth 模型 =====
# TODO: 根据实际模型路径导入
# from diffsynth.models.{series}_{component} import {ClassName} as DSModelClass
DSModelClass = None  # 占位符

# ===== 2. 提取 __init__ 签名默认值 =====
init_defaults = {}
sig = inspect.signature(DSModelClass.__init__)
for param_name, param in sig.parameters.items():
    if param_name == "self":
        continue
    if param.default is not inspect.Parameter.empty:
        init_defaults[param_name] = param.default

# ===== 3. 实例化模型，提取属性 =====
# TODO: 根据模型实际创建方式初始化
# model = DSModelClass()
model = None  # 占位符

model_attrs = {}
attr_names = list(init_defaults.keys())  # 使用相同名称列表提取
for name in attr_names:
    if hasattr(model, name):
        val = getattr(model, name)
        if not callable(val):
            model_attrs[name] = val

# ===== 4. 提取 extra_kwargs 中传入的参数 =====
extra_kwargs_params = {}
if hasattr(model, "extra_kwargs"):
    extra_kwargs_params = model.extra_kwargs

# ===== 5. 提取 model_configs.py 注册条目 =====
# TODO: 从 model_configs.py 中提取对应注册条目的 extra_kwargs
model_config_extra = {}  # 占位符，运行时从 model_configs.py 提取

result = {
    "model_class": f"{DSModelClass.__module__}.{DSModelClass.__name__}" if DSModelClass else "N/A",
    "init_defaults": init_defaults,
    "model_attributes": model_attrs,
    "extra_kwargs_params": extra_kwargs_params,
    "model_config_extra": model_config_extra,
}

print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
```

保存输出到执行日志：
```bash
cd {diffsynth_root}
python ${EXEC_LOG_DIR}/scripts/extract_diffsynth_params.py > ${EXEC_LOG_DIR}/outputs/diffsynth_all_params.json 2>&1
```

#### 9.3 生成参数对比报告

对比两组参数配置，生成详细的 JSON 报告：

```bash
cat > ${EXEC_LOG_DIR}/scripts/compare_params.py << 'SCRIPT'
#!/usr/bin/env python3
"""对比目标库和 DiffSynth 接入代码的参数配置，生成详细报告。"""
import json
import sys

def deep_compare(target, diffsynth):
    """对比两个参数字典，返回差异报告。"""
    all_keys = set(list(target.keys()) + list(diffsynth.keys()))
    results = []

    for key in sorted(all_keys):
        in_target = key in target
        in_diffsynth = key in diffsynth

        if in_target and in_diffsynth:
            t_val = target[key]
            d_val = diffsynth[key]
            match = t_val == d_val
            results.append({
                "param": key,
                "status": "match" if match else "mismatch",
                "target_value": repr(t_val),
                "diffsynth_value": repr(d_val),
                "target_source": _infer_source(target, key),
                "diffsynth_source": "init_default / model_attribute",
            })
        elif in_target and not in_diffsynth:
            results.append({
                "param": key,
                "status": "missing_in_diffsynth",
                "target_value": repr(target[key]),
                "diffsynth_value": None,
                "target_source": _infer_source(target, key),
                "diffsynth_source": "N/A",
            })
        else:
            results.append({
                "param": key,
                "status": "extra_in_diffsynth",
                "target_value": None,
                "diffsynth_value": repr(diffsynth[key]),
                "target_source": "N/A",
                "diffsynth_source": "init_default / model_attribute",
            })

    return results

def _infer_source(params, key):
    """推断参数来源。"""
    return "config / init / attribute"

def main():
    with open("${EXEC_LOG_DIR}/outputs/target_all_params.json") as f:
        target = json.load(f)
    with open("${EXEC_LOG_DIR}/outputs/diffsynth_all_params.json") as f:
        diffsynth = json.load(f)

    # 对比 merged 参数
    target_merged = target.get("all_params_merged", {})
    diffsynth_merged = {
        **diffsynth.get("init_defaults", {}),
        **diffsynth.get("model_attributes", {}),
        **diffsynth.get("extra_kwargs_params", {}),
        **diffsynth.get("model_config_extra", {}),
    }

    comparison = deep_compare(target_merged, diffsynth_merged)

    # 统计
    match_count = sum(1 for r in comparison if r["status"] == "match")
    mismatch_count = sum(1 for r in comparison if r["status"] == "mismatch")
    missing_count = sum(1 for r in comparison if r["status"] == "missing_in_diffsynth")
    extra_count = sum(1 for r in comparison if r["status"] == "extra_in_diffsynth")

    report = {
        "target_model_class": target.get("model_class", "N/A"),
        "diffsynth_model_class": diffsynth.get("model_class", "N/A"),
        "summary": {
            "total_params": len(comparison),
            "match": match_count,
            "mismatch": mismatch_count,
            "missing_in_diffsynth": missing_count,
            "extra_in_diffsynth": extra_count,
            "pass_rate": f"{match_count}/{len(comparison)} ({match_count/len(comparison)*100:.1f}%)",
        },
        "comparison": comparison,
    }

    with open("${EXEC_LOG_DIR}/outputs/params_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"参数对比验证报告")
    print(f"{'='*60}")
    print(f"目标库模型: {report['target_model_class']}")
    print(f"DiffSynth 模型: {report['diffsynth_model_class']}")
    print(f"\n总参数数: {len(comparison)}")
    print(f"✅ 匹配: {match_count}")
    print(f"❌ 不匹配: {mismatch_count}")
    print(f"⚠️ DiffSynth 缺失: {missing_count}")
    print(f"➕ DiffSynth 多余: {extra_count}")
    print(f"通过率: {report['summary']['pass_rate']}")

    # 打印不匹配项
    mismatches = [r for r in comparison if r["status"] != "match"]
    if mismatches:
        print(f"\n{'='*60}")
        print("不匹配/差异详情:")
        print(f"{'='*60}")
        for item in mismatches:
            print(f"\n  参数: {item['param']}")
            print(f"  状态: {item['status']}")
            print(f"  目标库: {item['target_value']}")
            print(f"  DiffSynth: {item['diffsynth_value']}")
            print(f"  来源: {item['target_source']} vs {item['diffsynth_source']}")

    # 通过/失败判断
    passed = mismatch_count == 0 and missing_count == 0
    print(f"\n{'='*60}")
    if passed:
        print("✅ PASSED: 所有参数配置一致")
    else:
        print("❌ FAILED: 存在参数不一致，需要修正")
        print("\n处理策略: 回到 Step 5（初始化参数验证），")
        print("修正 DiffSynth 模型 __init__ 默认值后，")
        print("重新从 Step 5 执行后续所有步骤。")
    print(f"{'='*60}")

    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/compare_params.py
```

**执行对比**：
```bash
cd {diffsynth_root}
python ${EXEC_LOG_DIR}/scripts/compare_params.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/params_comparison.log
```

**通过标准**：
- 所有目标库存在的参数在 DiffSynth 代码中都有对应
- 同名参数值完全一致
- 允许 DiffSynth 代码中存在目标库没有的额外参数（如 `use_gradient_checkpointing` 等框架特有参数）

**如果验证失败**：

参数对比不通过时，**回到 Step 5（初始化参数验证），修正后重新从 Step 5 开始执行后续所有步骤**：

```
参数对比失败
  ↓
修正 DiffSynth 模型 __init__ 默认值
  ↓
重新执行 Step 5: 初始化参数验证
  ↓
重新执行 Step 6: state_dict_converter（如需要）
  ↓
重新执行 Step 7: model_configs.py 注册
  ↓
重新执行 Step 8: 一致性验证
  ↓
重新执行 Step 9: 参数对比验证
  ↓
通过后才能继续
```

**重要**：不允降低阈值或忽略不匹配项蒙混过关。每个不匹配的参数都必须逐一修正，直到全部通过。

**对比报告输出**：
- JSON 报告: `${EXEC_LOG_DIR}/outputs/params_comparison_report.json`
- 文本日志: `${EXEC_LOG_DIR}/outputs/params_comparison.log`

报告内容包含：
- 目标库参数配置（按来源分类：config 文件、__init__ 默认值、模型属性、运行时推断）
- DiffSynth 接入代码参数配置（按来源分类：__init__ 默认值、模型属性、extra_kwargs、model_configs.py 注册参数）
- 逐参数对比结果（匹配/不匹配/缺失/多余）
- 统计摘要（总数、匹配数、不匹配数、通过率）

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

### 10. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有模型代码接入步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/model-code-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/model-code-report.md" \
  "packages/{model-name}/.sisyphus/user_report/model-code-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 2. 检查所有执行的测试脚本是否存在于执行日志目录
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/latest"
for script in "$EXEC_LOG_DIR/scripts/"*; do
  if [ -f "$script" ]; then
    echo "OK: $script 已保存"
  else
    echo "WARNING: 测试脚本缺失: $script"
  fi
done
```

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`


## 目录结构

本 skill 读取和写入以下路径：
- **蓝图报告**: `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- **中间结果**: `packages/{model-name}/.sisyphus/intermediate/model-pool/` - ModelPool 加载结果
- **中间结果**: `packages/{model-name}/.sisyphus/intermediate/forward/` - Forward 测试输出
- **执行日志**: `packages/{model-name}/.sisyphus/execution-logs/$(date)_model_code/`

## 蓝图需求

本 skill 需要从蓝图报告中获取：
- **接入类型**：new_series / version_upgrade
- **模型组件清单**：每个组件的类名、`__init__` 签名、`forward` 签名、特殊依赖
- **权重 key 模式**：用于判断是否需要 state_dict_converter
- **是否需要 state_dict_converter**：每个组件的权重格式是否与 DiffSynth 现有 converter 兼容
- **外部依赖组件分析**（关键）：
  - 哪些组件使用 diffusers？→ 需要重构，列出 diffusers 源文件路径
  - 哪些组件使用 transformers？→ 需要包装，列出 transformers 类名和版本要求
  - 每个组件的代码来源：目标库自有 / diffusers 需重构 / transformers 需包装
- **一致性测试规划**（关键）：
  - **所有组件都必须执行两种测试**（详见 Step 8）

**本 skill 必须输出**（每个组件）：
- **组件接入方案**：按「确定接入方式」章节 3.3 节的格式，为每个组件制定接入方案
  - 决策依据（接入类型、代码来源、签名对比结果、结构变化）
  - 选定的接入方式
  - 参考文件阅读确认
  - 执行动作清单
  - 文件路径规划
  - 测试要求

> 📝 完成后：更新渐进式报告 → `skill_work_report/model-code-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的所有测试脚本（forward 测试、对比脚本）
- **输出目录**: `outputs/` - 命令执行日志
- **检查点目录**: `checkpoints/` - ModelConfig 加载结果、forward 输出、对比结果、model_hash
- **用户检查清单**: `user-checks.md` - 需要人工确认的项目

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/model-code-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Model Code 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-model-code |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取模型代码接入所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：模型组件清单、接入类型、接入方式（自有源码/diffusers 重构/transformers 包装/完全复用/结构变化）
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_model-code/` 目录及子目录
- 创建 `manifest.json` 和 `latest` 软链接

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示模型代码接入规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/model-code-plan.md`
- 向用户展示组件接入方案、注册方案、一致性测试方案
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/model-code-plan.md`

---

### Step 3: 确定接入方式

**目标**：为每个组件选择接入方式。

**执行内容**：
- 对照蓝图中的组件接入方式判定
- 确认每个组件的代码来源

**产出物**：组件接入方式确认

---

### Step 4: 编写模型源代码

**目标**：按接入方案创建/修改模型组件代码。

**执行内容**：
- 自有源码：从目标库复制，替换 attention 和 gradient checkpointing
- diffusers 重构：从 diffusers 源码重构为独立实现
- transformers 包装：编写包装类接入 transformers 模型
- 完全复用：无需修改代码

**产出物**：
- `diffsynth/models/{series}_{component}.py`

---

### Step 5: 初始化参数验证

**目标**：验证模型可以初始化并统计参数量。

**执行内容**：
- 导入模型类，使用默认参数初始化
- 统计参数量

**产出物**：参数统计结果

---

### Step 6: 编写 state_dict_converter

**目标**：编写权重映射逻辑。

**执行内容**：
- 分析目标库权重格式
- 编写 state_dict_converter 进行权重映射

**产出物**：
- `diffsynth/utils/state_dict_converters/{series}_{component}.py`（如需要）

---

### Step 7: 注册到 model_configs.py

**目标**：添加组件注册信息。

**执行内容**：
- 在 `model_configs.py` 中添加 ModelConfig 注册条目
- 每个注册条目必须包含 `# Example: ModelConfig(...)` 注释

**产出物**：更新后的 `model_configs.py`

---

### Step 8: ModelPool 加载测试

**目标**：验证注册链正确。

**执行内容**：
- 使用 `ModelPool.auto_load_model` 测试加载
- 验证 ModelConfig 路径解析、converter 导入、模型实例化

**产出物**：加载测试通过确认

---

### Step 9: 一致性验证

**目标**：验证 DiffSynth 代码与目标库原始代码的 forward 输出完全一致。

**执行内容**：
- 使用相同输入分别运行目标库和 DiffSynth 代码
- 对比 forward 输出（阈值 max_diff < 1e-5）
- 中间层对比（阈值 max_diff < 1e-3）

**产出物**：一致性验证报告

---

### Step 10: 参数对比验证

**目标**：提取目标库与接入代码参数，生成对比报告。

**执行内容**：
- 提取目标库模型 `__init__` 参数
- 提取 DiffSynth 模型 `__init__` 参数
- 生成对比报告

**产出物**：参数对比报告

---

### Step 11: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 组件接入方案

### 组件 1: {ComponentName}
- **接入方式**: {复用/自有源码/diffusers 重构/transformers 包装/结构变化}
- **文件路径**: `diffsynth/models/{series}_{component}.py`
- **Converter 路径**: `diffsynth/utils/state_dict_converters/{series}_{component}.py`（如需要）
- **ModelConfig 注册**: `{model_id}` + `{origin_file_pattern}`
- **测试方案**: forward 逐层对比，阈值 max_diff < 1e-5

### 组件 2: {ComponentName}
- （同上格式）

## 验证方案
- **参数验证**: ModelPool.auto_load_model 加载 + 参数统计
- **一致性测试**: 原始 vs DiffSynth forward 逐层对比
  - 最终输出阈值: max_diff < 1e-5
  - 中间层阈值: max_diff < 1e-3
- **排查预案**: 查看 Top 20 差异层 + 第一个差异层定位问题
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/model-code-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-10 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 确定接入方式 | Step 3 |
| 4 | 编写模型源代码 | Step 4 |
| 5 | 初始化参数验证 | Step 5 |
| 6 | 编写 state_dict_converter | Step 6 |
| 7 | 注册到 model_configs.py | Step 7 |
| 8 | ModelPool 加载测试 | Step 8（前置验证） |
| 9 | 一致性验证 | Step 8 |
| 10 | 参数对比验证 | Step 9 |
| 11 | 最终验证 | Step 10 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/model-code-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败 / ⬜ 跳过
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 生成的代码文件

- **模型文件**: `{diffsynth_root}/diffsynth/models/{series}_{component}.py`
- **Converter**: `{diffsynth_root}/diffsynth/utils/state_dict_converters/{series}_{component}.py`
- **组件接入方案**: `packages/{model-name}/.sisyphus/plans/component-integration-plans.md`（每个组件的接入决策和执行清单）
- **注册配置**: `{diffsynth_root}/diffsynth/configs/model_configs.py`（修改）

### 向用户报告

模型代码接入完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/model-code-report.md << 'OUTER_EOF'
## ✅ 模型代码接入完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 📋 组件接入方案

已为每个组件制定接入方案：
`cat packages/{model-name}/.sisyphus/plans/component-integration-plans.md`

### 📋 人工检查清单

请查看并确认以下检查项：
\`cat packages/{model-name}/.sisyphus/execution-logs/latest/user-checks.md\`

关键检查项：
1. [CHECK-001] ModelPool 加载验证 - **前置验证，所有情况都必须执行**
2. [CHECK-003] Forward 一致性 - **所有情况都必须确认**
3. [CHECK-005] Converter 正确性 - **必须确认**

**接入类型**: {new_series / version_upgrade}
**组件接入方式**:
- {组件1}: {接入方式}
- {组件2}: {接入方式}
- ...

**Hash 检查结果**: {hash 已存在: 直接使用已有条目 / hash 不存在: 已添加新条目}
**测试执行情况**:
- **ModelPool 加载测试: 所有组件已执行（前置验证）**
- **ModelConfig + ModelPool 注册链验证: 所有组件已执行（永不跳过）**
- **Forward 一致性测试: 所有组件已执行（永不跳过）**
- **参数对比验证: 已执行，通过率 {pass_rate}**

### 📁 生成的文件

- 组件接入方案: `.sisyphus/plans/component-integration-plans.md`
- 模型文件: `diffsynth/models/{series}_{component}.py`
- Converter: `diffsynth/utils/state_dict_converters/{series}_{component}.py`
- ModelPool 加载测试脚本: `scripts/test_modelpool_load.py`
- ModelPool 加载结果: `checkpoints/modelpool_load_result.json`
- 一致性测试脚本: `scripts/test_forward_consistency.py`
- 一致性测试结果: `checkpoints/forward_consistency_result.json`
- 逐层对比详情: `checkpoints/layer_comparison.json`
- 参数对比报告: `outputs/params_comparison_report.json`

### 🚀 下一步

确认所有检查项后，可以继续：
- `diffsynth-pipeline` - 创建推理 Pipeline
OUTER_EOF
```
