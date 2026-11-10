---
name: diffsynth-analyze-target
description: >
  Analyze an external model codebase and generate an integration blueprint for DiffSynth-Studio.
  Use this skill whenever a user wants to integrate a new diffusion model into DiffSynth-Studio —
  whether they provide a local path (like "packages/ACE-Step-1.5"), a GitHub URL, or just a model
  name. This includes audio models (ACE-Step), video models (LTX, Wan), image models (Flux,
  Qwen-Image), or any diffusion-based model. Even if the user says "add support for X" or "look
  into integrating Y", consult this skill first to map out what needs to be done before any code
  changes are made. Do NOT start writing code until the blueprint is reviewed and confirmed.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — bug fixes, refactoring, performance tuning, general code
  changes, or using/training already-supported models.
---

# DiffSynth-Studio: 分析目标代码库

分析外部模型仓库，生成接入蓝图报告和代码阅读报告。这是所有后续接入步骤的基础。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**目标库路径确定流程：**

当用户提到接入某个模型时（如"接入 joyai-image"）：
1. **提取模型名称**：从用户输入中提取（如 "joyai-image"），作为 `{model-name}`
2. **检查项目目录**：查看 `{workspace_root}/packages/{model-name}/` 是否存在
   - 如果不存在 → 创建目录 `mkdir -p packages/{model-name}/`
3. **检查目标库**：在 `packages/{model-name}/` 下查找目标库目录
   - 通常目标库目录名与模型名称相同或相似（如 `joyai-image/`）
   - 如果找到多个子目录，向用户确认哪个是目标库
   - 如果没有找到目标库 → **向用户索要 git 路径**，然后 clone 到 `packages/{model-name}/`
4. **检查 DiffSynth-Studio**：检查 `packages/{model-name}/DiffSynth-Studio/` 是否存在
   - 如果不存在 → **自动 clone**：`git clone https://github.com/modelscope/DiffSynth-Studio.git packages/{model-name}/DiffSynth-Studio/`
5. **确定路径**：
   - `target_path`: `packages/{model-name}/{target-library}/`
   - `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
6. 将这些路径记录到对话上下文，后续写入蓝图报告

**目录结构示例**：
```
packages/
└── joyai-image/                    # {model-name}
    ├── joyai-image/                # 目标库（从git clone）
    │   ├── README.md
    │   ├── requirements.txt
    │   └── ...
    ├── DiffSynth-Studio/           # DiffSynth框架（自动clone）
    │   ├── diffsynth/
    │   ├── requirements.txt
    │   └── ...
    └── .sisyphus/                  # 自动生成
        ├── integration-blueprints/  # 蓝图报告
        ├── execution-logs/
        └── ...
```

## 核心原则

**先理解下游，再分析目标库。** 在开始分析之前，必须完整阅读所有下游 skill 的 SKILL.md 全文（见 Step 1）。不理解下游规则，分析结果将无法被使用。

**蓝图是下游 skill 的数据源。** 蓝图中的每个字段都必须对应下游 skill 的实际需求，不写无用信息。分析时始终问自己："这个信息哪个下游 skill 会用？怎么用？"

**分析必须贴合 DiffSynth 规则，不是目标库规则。** 目标库的代码结构、依赖管理、模型组织方式可能与 DiffSynth 完全不同。分析的目标是找出"如何将目标库映射到 DiffSynth 架构"，不是"目标库是怎么组织的"。

**执行留痕，过程可追溯。** 所有分析过程保存到日志，关键检查点明确标记。遵循 [execution-traceability.md](../diffsynth-integrator/references/execution-traceability.md) 规范。

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述：**
开始执行任何 Step 时，必须先重新阅读本 SKILL.md 全文。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/analyze-target-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/analyze-target-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

**不要跳过报告更新** — 即使某个 Step 被跳过或失败，也必须记录到报告中。报告是执行过程的唯一可追溯记录。

### 0. 初始化配置和执行日志

> 📖 开始前：重读本步骤描述，确认流程与报告路径

当用户提到接入某个模型时（如"接入 joyai-image"）：

**步骤 0a: 确定模型名称并创建项目目录**
```bash
# 从用户输入提取模型名称
MODEL_NAME="{model-name}"  # 如 joyai-image

# 创建项目目录
mkdir -p packages/${MODEL_NAME}
```

**步骤 0b: 获取目标库**
```bash
# 检查目标库是否已存在
target_dirs=$(find packages/${MODEL_NAME}/ -maxdepth 1 -type d ! -name "DiffSynth-Studio" ! -name ".sisyphus" ! -path "*/\.*" 2>/dev/null)

if [ -z "$target_dirs" ]; then
    # 目标库不存在，向用户索要 git 路径
    echo "在 packages/${MODEL_NAME}/ 下未找到目标库"
    # 向用户询问: "请提供目标库的 git URL"
    # git clone {用户提供的URL} packages/${MODEL_NAME}/{target-name}/
fi
```

**步骤 0c: 获取 DiffSynth-Studio**
```bash
# 检查 DiffSynth-Studio 是否已存在
if [ ! -d "packages/${MODEL_NAME}/DiffSynth-Studio" ]; then
    # 自动 clone DiffSynth-Studio
    git clone https://github.com/modelscope/DiffSynth-Studio.git packages/${MODEL_NAME}/DiffSynth-Studio/
    echo "已自动 clone DiffSynth-Studio 到 packages/${MODEL_NAME}/DiffSynth-Studio/"
fi
```

**步骤 0d: 确定最终路径**
- `target_path`: `packages/${MODEL_NAME}/{target-library}/`
- `diffsynth_root`: `packages/${MODEL_NAME}/DiffSynth-Studio/`

**规划 conda 环境名称**：
- 环境名称：`${MODEL_NAME}-diffsynth`（蛇形命名）
- 记录到蓝图报告的「基本信息」表格中

**创建执行日志目录**（在项目目录下的 .sisyphus 中）：

```bash
export EXEC_LOG_DIR="packages/${MODEL_NAME}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_analyze"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "diffsynth-analyze-target",
  "model_name": "${MODEL_NAME}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "analyze_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "user_checks": []
}
EOF

ln -sfn ${EXEC_LOG_DIR} packages/${MODEL_NAME}/.sisyphus/execution-logs/latest
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/analyze-target-report.md`

### 1. 强制读取所有diffsynth模型接入 Skill 全文（关键步骤）

**⚠️ 在开始分析目标库之前，必须完整阅读以下所有diffsynth模型接入 skill 的 SKILL.md 全文，以及它们 references/ 目录下的所有参考文件。** 这不是可选步骤——如果不理解下游 skill 的规则和约束，分析结果将无法被下游使用。

**为什么必须读全文（包括参考文件）**：
- 每个 skill 有严格的格式要求、文件命名规范、目录结构约定
- 每个 skill 都有「蓝图需求」章节，明确说明需要从蓝图获取什么
- 每个 skill 都有核心原则（如环境创建的"双向验证"、模型代码的"只做必要改动"、Pipeline 的"每次一个功能"）
- **参考文件包含详细的操作指南、模板、示例，是理解具体执行步骤的关键**
- 如果不理解这些规则，分析时可能遗漏关键字段、提供错误路径、或给出无法执行的方案

**必须读取的文件清单**（按执行顺序）：

| 顺序 | Skill 文件 | 参考文件（必须全部阅读） | 核心关注点 |
|------|-----------|------------------------|-----------|
| 1 | `diffsynth-environment/SKILL.md` | `references/*.md`（如 dependency-examples.md 等） | 环境创建流程、双向验证原则、diffusers 环境可安装但代码需重构策略、模型软链接结构、依赖分批安装策略 |
| 2 | `diffsynth-model-code/SKILL.md` | `references/*.md`（如 model-file-*.md、attention-replacement.md 等） | 模型接入架构、拆分设计原则、文件命名规范、state_dict_converter 格式、model_configs.py 注册格式、一致性验证流程 |
| 3 | `diffsynth-pipeline/SKILL.md` | `references/pipeline-template.md`、`inference-scripts.md`、`scheduler-integration.md`、`e2e-test.md` | Pipeline 文件结构、PipelineUnit 链设计、每次只接入一个功能、推理示例脚本格式、训练模块结构 |
| 4 | `diffsynth-pipeline-lowvram/SKILL.md` | 无 references | 低显存脚本生成规则、module_map 注册、vram_config 注入 |
| 5 | `diffsynth-pipeline-training/SKILL.md` | `references/*.md`（如 dataset-guidelines.md 等） | 训练模块结构、InputXXXEmbedder 设计、数据集规范、训练验证流程 |
| 6 | `diffsynth-style/SKILL.md` | 无 references | 代码风格规范、推理脚本 ModelConfig 单行规则、extra_kwargs 规则 |
| 7 | `diffsynth-docs/SKILL.md` | `references/doc-templates.md` | 文档固定模板结构、模型总览表格列定义、快速开始代码要求 |
| 8 | `diffsynth-testing/SKILL.md` | `references/test-report-format.md` | 测试范围、自动化判断规则、测试报告格式 |
| 9 | `diffsynth-pr/SKILL.md` | `references/pr-template.md` | PR 描述结构、收集范围 |
| 10 | `diffsynth-pr-review/SKILL.md` | 无 references | PR 反馈分类、最小测试策略 |

**阅读顺序**：
1. 首先阅读 SKILL.md 全文，理解整体流程
2. **然后必须阅读该 skill 的 references/ 目录下所有 .md 文件**，这些文件包含详细的操作指南、模板和示例
3. 对于 `diffsynth-model-code`，特别重要的是阅读 `references/model-file-*.md` 系列文件，因为分析时需要判断每个组件使用哪种接入方式

**读取后必须理解的关键规则**：

#### 来自 diffsynth-environment
- ⚠️ **diffusers**：环境验证阶段**可以安装**（用于验证目标库运行），但模型代码阶段**必须重构**为独立实现（DiffSynth 模型代码不可 import diffusers）
- ✅ **transformers 可以安装**：DiffSynth 允许依赖 transformers，但需编写接入包装类
- 环境创建遵循"DiffSynth OK → Target OK → DiffSynth still OK"双向验证
- 模型软链接结构：`{target_path}/models/` 和 `{diffsynth_root}/models/` 各自创建真实目录，内部每个模型 repo 直接软链接到 `~/.cache/modelscope/hub/models/{org}/{model}/`

#### 来自 diffsynth-model-code
- **模型拆分原则**：DiT 只负责去噪预测，条件准备是独立模型，VAE 独立编解码，Pipeline 负责编排
- 文件命名：`diffsynth/models/{series}_{component}.py`
- 只做必要的改动，版本升级时如果结构不变只需注册新 hash
- 必须通过 `ModelConfig` + `ModelPool.auto_load_model` 验证完整注册链

#### 来自 diffsynth-pipeline
- **每次只接入一个功能**，按从简单到难的顺序
- Pipeline 文件按系列组织，不按功能拆分（一个系列一个文件）
- Pipeline 是编排者，不是模型实现者
- 训练模块整个系列共用一个脚本

#### 来自 diffsynth-docs
- 文档有固定模板结构，不能随意更改章节顺序
- 模型总览表格固定 8 列，不增不减
- 快速开始代码必须完整复制推理脚本，不能简化

#### 来自 diffsynth-testing
- 测试范围：推理脚本、低显存推理、训练验证脚本
- 失败不阻塞，一个脚本失败不影响其他
- 输出质量必须人工确认

#### 来自 diffsynth-pr
- 分阶段提交：deps → models → pipelines → training → examples → docs
- 每个 commit 包含：做了什么、为什么、验证结果

**带着这些规则去分析目标库**，确保分析结果：
- 提供的路径符合下游 skill 的命名规范
- 提供的组件清单包含下游需要的字段
- 提供的依赖分析考虑了 diffusers 代码不可依赖的限制
- 提供的功能规划考虑了"每次只接入一个功能"的原则
- 提供的拆分建议符合 DiffSynth 的模型边界设计

### 2. 按需分析目标库

**2a. 读取项目概览**
读取根目录结构、`pyproject.toml`/`requirements.txt`、`README.md`、`AGENTS.md`（如果存在）。

**2b. 识别模型组件**
找到所有模型定义文件（通常在 `*/models/` 或 `*/core/` 下）。对每个组件提取：
- 类名
- `__init__` 参数
- `forward` 签名
- 特殊依赖（如 flash-attn、xformers）
- 权重 key 模式（打印 `model.state_dict().keys()` 的前几条）

**同时判断**：与 DiffSynth 已有组件的差异（version_upgrade 时关键）。

**2b.1. 模型耦合分析（关键步骤）**

**目标库的模型代码往往耦合了多种职责**。必须分析耦合情况，为后续拆分提供依据。

分析以下耦合模式：

| 耦合类型 | 特征 | 拆分建议 |
|---------|------|---------|
| **训练+推理耦合** | forward 中包含损失计算、labels 参数 | 分离 forward（推理）和 compute_loss（训练） |
| **条件+DiT 耦合** | DiT 内部直接调用 tokenizer、text_encoder | 提取 TextEncoder 为独立模型 |
| **VAE+DiT 耦合** | DiT 或 Pipeline 内部调用 VAE encode/decode | 将 VAE 移到 Pipeline 中 |
| **采样+模型耦合** | 模型类包含 sample/generate 方法 | 将采样逻辑移到 Pipeline |
| **多条件混合** | 文本、音频、图像条件在同一 forward 中处理 | 拆分为独立的条件编码器 |

**分析方法**：
1. 检查模型 forward 方法的参数：是否接收原始文本/音频（应该是 embeds）
2. 检查模型内部是否调用 tokenizer、processor（应该是独立模型）
3. 检查模型是否包含训练逻辑（如 loss 计算、labels 处理）
4. 检查模型是否包含采样逻辑（如 scheduler step、循环）

**在蓝图报告中记录**：
- 耦合点清单（具体文件、类名、方法名）
- 拆分建议（每个耦合点如何拆分）
- 拆分后的模型组件清单

**2b.2. 外部依赖组件识别（关键步骤）**

对每个识别出的组件，**必须判断其代码来源**：

| 代码来源 | 特征 | DiffSynth 处理方式 |
|---------|------|------------------|
| **目标库自有源码** | 目标库 `*/models/` 下有完整 `.py` 文件 | 直接接入，按正常流程处理 |
| **Diffusers 组件** | 使用 `from diffusers import AutoencoderKL` 或类似导入 | ⚠️ **DiffSynth 模型代码不可 import diffusers**。必须找到 diffusers 对应源码，重构为 DiffSynth 风格的独立实现（参考 `flux2_vae.py`） |
| **Transformers 组件** | 使用 `from transformers import AutoModel` 或类似导入 | ✅ **可以依赖 transformers**，但仍需编写接入包装类（参考 `qwen_image_text_encoder.py`） |
| **其他外部库** | 如 dacite、custom audio codec 等 | 逐个评估，一般需重构或找到替代实现 |

**Diffusers 组件处理流程**：
1. 识别目标库使用的是 diffusers 的哪个组件（如 `AutoencoderKL`、`UNet2DConditionModel`）
2. 在 diffusers 源码中找到对应实现（`src/diffusers/models/autoencoders/autoencoder_kl.py` 等）
3. 评估重构工作量：该组件有多少层、是否有特殊操作（如 spatial normalization、custom attention）
4. 在蓝图报告中明确标注：「需要重构 diffusers 组件」，并列出需要复制/重构的源文件
5. **向用户确认**：是否接受重构 diffusers 组件的工作量（通常必须接受，因为 DiffSynth 模型代码不可 import diffusers）

**Transformers 组件处理流程**：
1. 识别目标库使用的是 transformers 的哪个模型（如 `Qwen2_5_VLModel`、`CLIPTextModel`）
2. 确认 transformers 版本要求
3. 编写接入包装类（参考 `qwen_image_text_encoder.py` 模式）：
   - 在 `__init__` 中 lazy import transformers 类
   - 手动构建 config dict（或从预训练配置加载）
   - 包装 transformers 模型为 `torch.nn.Module`
   - 实现符合 DiffSynth 需求的 `forward` 方法
4. 在蓝图报告中明确标注：「需要编写 transformers 接入包装类」

**参考实现**：
- **Transformers 接入**：`diffsynth/models/qwen_image_text_encoder.py` — 在 `__init__` 中 import transformers，手动构建 config，包装为 torch.nn.Module
- **Diffusers 重构**：`diffsynth/models/flux2_vae.py` — 从 diffusers 源码重构为独立实现，保留 Apache 2.0 许可证头（但不添加代码来源注释如 "copied from"）

**2c. 分析推理入口 / 功能规划**

找到推理流程入口（含 `pipeline`、`inference`、`generate` 的文件），理解输入→处理→输出的完整流程。

**关键**：识别出目标库的所有功能，按从简单到难的顺序规划：
- **简单**：基础生成（如 Text-to-X），无额外控制输入
- **中等**：带条件输入（如 Image-to-X、Audio-to-X），需要额外编码步骤
- **复杂**：带控制信号（如 In-Context Control、Camera Control、Retake），需要理解控制逻辑

每个功能记录：
- 原库 example 脚本路径
  - 脚本需要的模型及下载方式（ModelScope model_id + origin_file_pattern）
  - 脚本的输入参数（prompt、seed、步数、控制信号等）
  - 功能之间的依赖关系

**模型识别与下载规划**：
对每个功能，明确识别需要的模型：
1. **模型名称**：如 `ACE-Step-1.5-Text2Music`
2. **ModelScope ID**：如 `ACE-Step/ACE-Step-1.5-Text2Music`
3. **模型用途**：如 DiT 权重、VAE 权重、TextEncoder 等
4. **文件大小**（如果可从 API 获取）：用于预估下载时间
5. **是否必需**：核心功能必需 / 可选功能可选

**模型下载统一流程**：
1. 通过 `modelscope download --model {model_id} --local_dir ~/.cache/modelscope/hub/models/{model_id}` 下载到 ModelScope cache `~/.cache/modelscope/hub/models/`
2. 在 `{target_path}/models/` 和 `{diffsynth_root}/models/` 下分别创建软链接，直接指向 `~/.cache/modelscope/hub/models/{model_id}`

**生成下载命令清单**：
在分析完成后，为每个识别出的模型生成具体的下载命令：
```bash
# 下载命令（在环境准备前手动执行）
modelscope download --model {model_id} --local_dir ~/.cache/modelscope/hub/models/{model_id}

# 软链接命令（环境准备阶段自动执行）
mkdir -p {target_path}/models/$(dirname {model_id})
ln -sf ~/.cache/modelscope/hub/models/{model_id} {target_path}/models/{model_id}
mkdir -p {diffsynth_root}/models/$(dirname {model_id})
ln -sf ~/.cache/modelscope/hub/models/{model_id} {diffsynth_root}/models/{model_id}
```

**2d. 代码运行链路分析**
绘制完整的代码运行链路：
- **数据流**：输入数据（text/image/audio）如何在各组件间流转
- **控制流**：生成过程中的关键决策点（如 cfg_scale 应用、timestep 调度）
- **组件调用链**：TextEncoder → DiT → VAE 的调用顺序和数据传递
- **特殊处理**：与标准扩散流程不同的步骤

**2e. 对比依赖**
对比目标库与 DiffSynth 的 `pyproject.toml`，产出三分类：
- **已有包**：DiffSynth 已有，版本满足目标库要求 → 跳过
- **额外包**：目标库独有 → 需要安装
- **冲突包**：版本不兼容 → 谨慎处理

注意平台特定依赖（`sys_platform`、`platform_machine` 标记）。

**2e.1. LLM/推理后端分析（如果目标库使用外部推理引擎）**

如果目标库使用了外部推理引擎（如 vLLM、nano-vllm、TGI 等），必须分析：
1. **后端选项**：是否有多种后端可选（如 vLLM / PyTorch / MLX）
2. **Fallback 机制**：当首选后端不可用时，是否自动降级
3. **DiffSynth 接入策略**：
   - 默认使用哪种后端（优先选择依赖最少的）
   - 可选后端如何标记（在蓝图中注明"可选依赖"）
   - 是否需要额外安装本地包（如 `third_parts/` 下的包）

**示例**（ACE-Step-1.5）：
- 支持 vLLM（nano-vllm）、PyTorch、MLX 三种后端
- vLLM 不可用时自动 fallback 到 PyTorch
- DiffSynth 默认使用 PyTorch 后端（transformers 直接加载）
- nano-vllm 标记为可选依赖，用户可自行安装

**2e.2. 功能可行性评估**

对目标库的每个功能，评估接入可行性：
- **核心依赖**：DiffSynth 是否已有或可安装
- **特殊依赖**：是否有难以接入的依赖（如特定推理引擎）
- **输入输出**：是否匹配 DiffSynth 的 Pipeline 接口
- **复杂度**：实现难度（简单/中等/复杂）
- **外部组件评估**：
  - 哪些组件使用 diffusers？→ 需要重构，评估工作量
  - 哪些组件使用 transformers？→ 需要包装，评估工作量
  - 是否有其他不可接入的外部依赖？

输出到 `packages/{model-name}/.sisyphus/integration-blueprints/feasibility-analysis.md`。

**2f. 判断整体接入类型（new_series vs version_upgrade）**

**整体接入类型**是针对整个模型系列的判断，决定是否需要创建全新的模型系列。

| 类型 | 特征 | 判定标准 |
|------|------|---------|
| **`new_series`** | 全新模型家族 | DiffSynth 中没有类似架构的模型系列。需要为每个组件新建模型文件 |
| **`version_upgrade`** | 已有模型升级 | DiffSynth 中已有相同架构的模型系列。需要进一步判断每个组件的接入类型 |

**判断流程**：

```
Step 1: DiffSynth 中是否已有类似架构的模型系列？
├── 无 → new_series
└── 有 → version_upgrade（继续判断每个组件的接入类型）
```

**示例**：
- ACE-Step（全新音频生成模型）→ `new_series`
- Wan2.1-T2V-14B（已有 Wan2.1 1.3B 的基础上）→ `version_upgrade`

**2f.1. 判断组件接入方式（每个组件独立判断）**

**⚠️ 关键：整体接入类型和组件接入方式是两个不同维度的判断**

- **整体接入类型**：针对整个模型系列（new_series / version_upgrade）
- **组件接入方式**：针对每个组件的独立判断（自有源码 / diffusers 重构 / transformers 包装 / 完全复用 / 结构变化）

**组件接入方式与 model-code skill 保持一致**：

| 整体接入类型 | 组件情况 | 组件接入方式 | 说明 |
|-------------|---------|-------------|------|
| `new_series` | 目标库自有代码 | `自有源码` | 复制目标库代码，仅替换 attention 和 gradient checkpointing |
| `new_series` | 依赖 diffusers | `diffusers 重构` | 从 diffusers 源码重构为独立实现 |
| `new_series` | 依赖 transformers | `transformers 包装` | 编写包装类接入 transformers 模型 |
| `version_upgrade` | 签名完全相同 | `完全复用` | 复用已有组件，仅注册新 hash |
| `version_upgrade` | 签名有变化 | `结构变化` | 修改现有模型文件 |

**判断流程**（每个组件独立执行）：

```
Step 1: 整体接入类型是什么？
├── new_series → 进入 Step 2（判断代码来源）
└── version_upgrade → 检查签名
    ├── 完全相同 → 完全复用
    └── 有变化 → 结构变化

Step 2: 代码来源是什么？（new_series 时）
├── 目标库自有源码 → 自有源码
├── diffusers 依赖 → diffusers 重构
└── transformers 依赖 → transformers 包装
```

**示例**（Wan2.1-T2V-14B 相对于 Wan2.1-T2V-1.3B）：

| 组件 | 整体接入类型 | 代码来源 | 组件接入方式 | 说明 |
|------|-------------|---------|-------------|------|
| DiT | version_upgrade | 目标库自有 | `结构变化` | 参数量变化（30层→40层），`__init__` 参数变化 |
| VAE | version_upgrade | diffusers | `完全复用` | 完全复用 WanVAE，签名相同 |
| TextEncoder | version_upgrade | transformers | `transformers 包装` | Wan2.1-T2V-1.3B 没有 TextEncoder，14B 版本新增 |

**在蓝图报告中记录**：
- 整体接入类型（new_series / version_upgrade）
- 每个组件的组件接入方式（自有源码 / diffusers 重构 / transformers 包装 / 完全复用 / 结构变化）
- 每个组件的代码来源（目标库自有 / diffusers / transformers）

**2g. 指定验证脚本与模型需求**
- **DiffSynth 验证脚本**：选择一个已有的、能正常运行的推理脚本，用于环境创建前后验证 DiffSynth 是否正常
- **目标库验证脚本**：选择要接入的第一个 pipeline 功能对应的 example 脚本，用于验证目标库在共享环境中能否运行
- **分析目标库验证脚本的模型需求**：
  - 需要的模型列表（model_id 列表）
  - 每个模型的用途（DiT/VAE/TextEncoder 等）
  - 预估总下载大小（帮助用户规划时间）
  - 生成完整的下载命令脚本（见第 3 节「模型下载命令清单」）

**2h. 功能-脚本映射规划**

为每个识别出的功能创建**独立的可运行验证脚本**，用于环境验证和推理一致性测试。

对每个功能，明确以下映射关系：
- **目标库 Example**：原库的 example 脚本或配置文件路径（如 `examples/text2music/example_01.json`）
- **目标库验证脚本**：为当前功能创建的可独立运行的 Python 脚本（如 `test_env_text2music.py`）
  - 脚本必须能直接运行（`python test_env_xxx.py`），自动初始化模型、执行推理、保存输出
  - 脚本需要自动生成测试所需的输入数据（如测试音频），或明确说明需要的输入文件
  - 脚本输出测试结果汇总（通过/失败/跳过）
- **DiffSynth 推理脚本**：规划 DiffSynth 接入后对应的推理脚本路径（如 `examples/ace_step/model_inference/acestep-v15-base.py`）

输出功能-脚本映射表到 `packages/{model-name}/.sisyphus/integration-blueprints/function-script-mapping.md`，包含：
- 每个功能的完整映射关系（目标库Example → 验证脚本 → DiffSynth推理脚本）
- 每个验证脚本的运行方式、模型要求、输入输出说明
- 功能依赖关系图（哪些功能依赖其他功能的基础能力）
- 推理一致性测试流程（如何对比目标库和 DiffSynth 的输出）

**关键**：验证脚本必须是**可直接运行的**，不是伪代码或示例片段。下游 skill（diffsynth-environment、diffsynth-testing）将直接使用这些脚本进行环境验证和一致性测试。

### 3. 生成蓝图报告

将 Step 2 分析得到的信息，按固定格式组织成蓝图报告。

**固定格式**见 [references/blueprint-format.md](references/blueprint-format.md)。

输出到 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`。

**关键**：蓝图中的每个字段都必须有具体值，不写"待分析"、"待定"等占位符。如果某个字段无法确定，明确说明原因和建议的调查方向。

**生成前自检**：在生成蓝图报告之前，对照以下清单检查是否收集了所有下游 skill 需要的信息：

| 下游 Skill | 必须提供的信息 | 蓝图中的对应章节 |
|-----------|--------------|----------------|
| diffsynth-environment | 依赖差异分析、DiffSynth 验证脚本路径、目标库验证脚本路径、验证脚本所需模型清单、模型下载命令 | 第 1 节依赖分析、第 1.4 节验证脚本、第 6 节模型下载 |
| diffsynth-model-code | 整体接入类型、各组件接入方式、模型组件清单（类名、__init__ 签名、forward 签名）、权重 key 模式、是否需要 converter、组件代码来源、耦合分析结果、拆分建议、一致性测试规划 | 基本信息、第 2 节（2.0/2.0.1/2.1/2.2/2.2.1） |
| diffsynth-pipeline | 功能名称（每次一个）、原库推理脚本路径、输入输出、特殊处理、依赖的模型组件、是否首次调用、Scheduler 信息 | 第 3 节 Pipeline 功能规划、第 3.1 节 Scheduler |
| diffsynth-pipeline-lowvram | Pipeline 类名、功能列表、模型组件列表、组件显存分析、优化建议 | 基本信息、第 3 节、第 2 节、第 4.2 节 |
| diffsynth-pipeline-training | 训练支持、训练类型、数据集格式、Input Embedder 需求、Scheduler 训练策略、训练参数 | 第 3 节（功能表）、第 4.3 节、第 3.1 节 |
| diffsynth-style | Conda 环境名称、模型组件清单、接入类型 | 基本信息、第 2 节 |
| diffsynth-docs | 接入类型、功能列表、Pipeline 类名、与已有版本的差异 | 基本信息、第 3 节、第 4 节 |
| diffsynth-testing | 推理脚本路径、低显存推理脚本路径、训练验证脚本路径（以实际文件系统为准） | 第 4.1 节功能-脚本映射 |
| diffsynth-pr | 接入类型、Pipeline 功能规划（仅作参考，以实际代码为准） | 基本信息、第 3 节 |
| diffsynth-pr-review | Conda 环境名称（执行最小测试用），其余不依赖蓝图 | 基本信息 |

**如果任何一项缺失，不要生成蓝图报告**，先回到 Step 2 补充分析。

### 4. 生成代码阅读报告

**目标**：让没有看过目标库代码的人，能在 5 分钟内理解整个推理流程。

报告内容：
- **项目结构速览**：关键目录和文件说明（用目标库的实际目录名，不预设 DiffSynth 式的名称）
- **入口点指引**：从哪个文件开始阅读
- **核心类说明**：DiT、VAE、TextEncoder 等核心类的职责和关系
- **运行链路图解**：使用 Mermaid 流程图展示数据流、控制流、组件调用链
- **关键代码段**：标注值得重点阅读的代码位置
- **与 DiffSynth 的映射**：目标库的哪些部分对应 DiffSynth 的哪些模式
- **快速开始**：如何运行目标库的 example

Mermaid 流程图规范：
- 使用 `flowchart TD` 或 `flowchart LR` 定义方向
- 使用 `-.->` 标注文件调用关系
- 使用 `subgraph` 分组相关组件
- 在节点中简要说明职责，代码位置用 `<br/>` 换行

完整模板见 [references/code-report-template.md](references/code-report-template.md)。

输出到 `packages/{model-name}/.sisyphus/code-reports/code-report.md`。

### 5. 最终验证

在所有分析步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/analyze-target-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/analyze-target-report.md" \
  "packages/{model-name}/.sisyphus/user_report/analyze-target-report.md"; do
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

## 输出

### 生成的报告

1. **接入蓝图报告**（`packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`）
   - 供后续 skill 使用，指导接入工作
   - **必须包含「模型下载命令清单」章节**，列出所有需要下载的模型和具体命令

2. **代码阅读报告**（`packages/{model-name}/.sisyphus/code-reports/code-report.md`）
   - 供人快速理解和熟悉目标库代码

3. **功能可行性分析**（`packages/{model-name}/.sisyphus/integration-blueprints/feasibility-analysis.md`）
   - 评估各功能的接入可行性
   - 特别关注复杂依赖（如 vLLM）的处理方案
   - 给出接入优先级建议

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的分析脚本
- **输出目录**: `outputs/` - 命令执行日志
- **检查点目录**: `checkpoints/` - 中间分析结果

### Plan

在开始分析前，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/analyze-target-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Analyze Target 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-analyze-target |
| 执行时间 | {timestamp} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 初始化配置和执行日志

**目标**：确定模型名称并创建项目目录结构。

**执行内容**：
- 从用户输入提取模型名称
- 创建项目目录 `packages/{model-name}/`
- 检查/获取目标库和 DiffSynth-Studio
- 创建执行日志目录、manifest.json、latest 软链接
- 规划 conda 环境名称

**产出物**：
- 项目目录结构
- 执行日志目录
- Conda 环境名称规划

---

### Step 1: 强制读取所有下游 Skill 全文

**目标**：理解所有下游 skill 的需求，确保分析结果可被使用。

**执行内容**：
- 阅读 10 个下游 skill 的 SKILL.md 全文
- 阅读每个 skill references/ 目录下所有参考文件
- 理解每个 skill 的核心原则和蓝图需求

**产出物**：下游需求理解确认

**注意事项**：
- 这是关键步骤，不理解下游规则分析结果将无法被使用

---

### Step 2: 按需分析目标库

**目标**：分析目标库的模型组件、依赖、功能和耦合情况。

**执行内容**：
- 读取项目概览（README、pyproject.toml、requirements.txt）
- 识别模型组件（类名、__init__ 参数、forward 签名）
- 分析模型耦合情况（训练+推理、条件+DiT、VAE+DiT 等）
- 识别外部依赖组件（diffusers/transformers/其他）
- 分析推理入口和功能规划
- 对比依赖差异
- 判断接入类型（new_series / version_upgrade）

**产出物**：
- 模型组件清单
- 耦合分析报告
- 依赖差异分析
- 接入类型判断

---

### Step 3: 生成蓝图报告

**目标**：按固定格式输出集成蓝图报告。

**执行内容**：
- 按 `references/blueprint-format.md` 固定格式组织
- 确保所有下游 skill 需要的信息都已包含
- 生成前自检：对照下游需求清单检查完整性
- 输出到 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`

**产出物**：
- 蓝图报告（供下游 skill 使用）
- 模型下载命令清单（第 6 节）

---

### Step 4: 生成代码阅读报告

**目标**：让没有看过目标库代码的人，能在 5 分钟内理解整个推理流程。

**执行内容**：
- 项目结构速览
- 入口点指引
- 核心类说明
- 运行链路图解（Mermaid 流程图）
- 与 DiffSynth 的映射

**产出物**：
- 代码阅读报告（供人阅读）

**注意事项**：
- 使用目标库的实际目录名，不预设 DiffSynth 式的名称

---

### Step 5: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 分析规划

### 分析目标
- **目标库路径**: {target_path}
- **关注点**: 模型组件清单、依赖差异、推理流程
- **输出**: 蓝图报告（供下游 skill 使用）+ 代码阅读报告（供人阅读）

### 下游 Skill 需求清单
- **environment**: 依赖差异、验证脚本路径
- **model-code**: 组件清单、接入方式、权重 key 模式
- **pipeline**: 推理流程、Unit 设计、scheduler 信息
- **pipeline-lowvram**: 组件列表、module_map 注册
- **pipeline-training**: 训练方式、InputXXXEmbedder 设计
- **style**: 无特殊需求
- **testing**: 推理脚本路径、一致性测试方案
- **docs**: Pipeline 类名、接入类型
- **pr**: 无特殊需求
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/analyze-target-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-4 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 1 | 读取下游 Skill 全文 | Step 1 |
| 2 | 分析目标库概览 | Step 2 |
| 3 | 识别模型组件 | Step 2 |
| 4 | 模型耦合分析 | Step 2 |
| 5 | 外部依赖组件分析 | Step 2 |
| 6 | 功能规划 | Step 2 |
| 7 | 依赖对比 | Step 2 |
| 8 | 生成蓝图报告 | Step 3 |
| 9 | 生成代码阅读报告 | Step 4 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/analyze-target-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

分析完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/analyze-target-report.md << 'OUTER_EOF'
## 📋 目标库分析完成

### 接入类型判断
- **整体接入类型**: {new_series / version_upgrade}
- **判断依据**: {说明}

### 各组件接入方式
| 组件 | 接入方式 | 代码来源 |
|------|---------|---------|
| {组件1} | {方式} | {来源} |

### 功能列表和优先级
...

### 📦 模型下载清单

在继续环境准备之前，请先下载以下模型：

### 必需模型

| 模型 | ModelScope ID | 用途 | 预估大小 |
|------|--------------|------|---------|
| ... | ... | ... | ... |

### 下载命令

```bash
# 模型1
modelscope download --model {model_id_1} --local_dir ~/.cache/modelscope/hub/models/{model_id_1}
```
OUTER_EOF
```

## 后续步骤

用户确认蓝图后，根据需要依次调用：`diffsynth-environment` → `diffsynth-model-code` → `diffsynth-pipeline`（每个功能调用一次） → `diffsynth-docs` → `diffsynth-testing` → `diffsynth-pr`。
