# DiffSynth-Studio 模型接入：使用教程

## 这是什么

一套完整的自动化流程，将外部扩散模型接入 DiffSynth-Studio。由 11 个独立 skill 组成，支持逐步执行和全流程编排。

| Skill | 能力 |
|-------|------|
| `diffsynth-analyze-target` | 阅读目标库代码，识别模型组件（DiT/VAE/TextEncoder 等）和所有可用功能，分析依赖差异和模型耦合关系，输出接入蓝图和代码阅读报告 |
| `diffsynth-environment` | 从源环境 clone 创建新 conda 环境，安装 DiffSynth 和目标库依赖，创建模型软链接，双向验证两个库都能正常运行 |
| `diffsynth-model-code` | 自动编写模型的 `nn.Module` 源代码（DiT/VAE/TextEncoder 等），编写 `state_dict_converter` 对齐权重 key，注册到 `model_configs.py`，通过原库 vs DiffSynth 的 Forward 输出一致性验证 |
| `diffsynth-pipeline` | 创建或更新 Pipeline 文件，设计 Unit 链和数据流，生成推理示例脚本（`model_inference/`），通过原库 vs DiffSynth 的端到端输出对比验证 |
| `diffsynth-pipeline-lowvram` | 在 `vram_management_module_maps.py` 注册 module_map，为每个已接入功能生成低显存推理脚本（`model_inference_low_vram/`），注入 `vram_config` 实现 CPU offload |
| `diffsynth-pipeline-training` | 创建训练模块（`train.py` + `.sh` 配置 + 验证脚本），准备数据集，通过 LoRA 降参调试和完整训练两轮验证 |
| `diffsynth-style` | 统一代码风格（ModelConfig 一行写完、`extra_kwargs` 一行写完、Pipeline dict 按参数组换行等），纯格式修改不改变逻辑 |
| `diffsynth-testing` | 收集所有推理脚本和低显存脚本的运行输出，使用 mtime 快照 + 移动策略不删除文件，生成测试报告和输出审查 |
| `diffsynth-docs` | 从已接入的代码中提取信息（Pipeline 参数签名、推理脚本列表、训练脚本路径），生成中英文文档，更新 `index.rst` 和 `README.md`/`README_zh.md` |
| `diffsynth-pr` | 扫描实际的 Pipeline 文件、模型组件和推理脚本，生成 PR 描述 Markdown 文件（`pr/description.md`），不执行 git 操作 |
| `diffsynth-pr-review` | 从 GitHub 抓取 PR Review 评论，生成修改蓝图，针对性修复并运行验证（推理 smoke test / 训练验证） |

> 以上 11 个 skill 虽然组成完整流程，但每个都是独立的——只要提供合理的上下文（目标库路径、蓝图报告等），任何一个都可以单独运行，不必走完全流程。

## 注意事项

### Human Must Check

**以下六项必须在流程中由人工逐一确认，不可跳过：**

1. **Pipeline 代码** — 所有生成的 Pipeline 代码必须人工 review 一遍，确认数据流、Unit 链和参数传递逻辑正确。Pipeline 是整个接入流程中最容易出错的环节，且 agent 生成的代码可能隐藏细微问题。
2. **Scheduler 严格对齐目标** — Scheduler 的实现必须与目标库严格一致，包括调度算法、参数默认值、每一步的计算顺序。任何偏差都会直接导致生成结果错误。
3. **train.py 文件** — 训练模块的代码逻辑必须人工确认，包括损失函数、数据加载、训练步骤是否正确，确保训练模块可以正常运行。
4. **VRAM 正确性** — 低显存推理的 module_map 注册和 vram_config 注入必须确认每个模块的 offload/加载时机正确，不出现模块未加载或重复加载的问题。
5. **输出结果确认** — 所有成功运行的推理脚本，其输出文件（图片/视频等）必须人工检查质量，不能仅依赖脚本无报错就认为接入完成。
6. **一致性测试结果** — `diffsynth-model-code` skill 输出的 Forward 一致性测试结果必须人工检查，确认 `forward_consistency_result.json` 中 DiffSynth 与原始代码的 forward 输出在数值精度范围内完全一致（`allclose(atol=1e-5)`），不能仅依赖测试脚本无报错就认为接入完成。

### 效果预期

- **除 Pipeline 外，其余 skill 已比较稳定。** Pipeline skill 生成的代码一般能正常运行，但如果效果不对，仍需人工 debug 一遍 Pipeline 流程。这是整个流程中唯一可能需要手动写代码和调试的环节。
- **建议先完整阅读所有 skill 的工作流程再使用。** 这本身也是一套完整的 DiffSynth 模型接入说明。

### 使用建议

- **蓝图是后续所有 skill 的数据源，务必在 Step 0 投入足够精力。** 接入人对目标库的接入方案需要有明确的标准和判断，建议先确定好方案，再与 agent 一起修改确认好蓝图后再往后走。蓝图有误，会给后续步骤带来不确定性。
- **每个 skill 都会先生成 plan 或蓝图，建议先确认 plan 再开始执行。**
- **每个 skill 都有对应的报告和完整的执行留痕。** 每个 skill 对应一个接入流程的 workstep，推荐确认当前 step 准确无误后再往下运行，确认无误后建议 commit 备份一次。
- **所有生成的代码建议 review 一遍。** 特别是 Pipeline 的流程逻辑，避免引入潜在风险。未来再定位问题、再 debug 的成本可能更高。
- **Agent 容易遗漏上下文。** 每次执行 skill 时，建议在 prompt 中明确要求 agent 参考 skill 自身的工作流程执行，防止跳过步骤或偏离标准流程。
- **简单、确定性的工作推荐手动完成。** 如下载模型、准备目录结构、创建软链接等，手动执行可显著减少 agent 引入的不确定性。
- **每个 skill 都必须输出 plan 和 report，以确保执行的稳定性和可追溯性。** 但如果是一个简单的需求（比如修改某个文件、调整某段代码），可以在 prompt 中直接指定具体该做什么，此时 skill 可以直接执行，不用输出 plan。
- **不一定所有步骤都要交给 agent 执行。** 如果一项工作人工可以很快完成，没必要让 AI 做。例如 testing 只包含一两个脚本时，手动跑一遍比启动完整的 testing skill 更快、更可控。

### 安全限制

- **Skill 本身不被允许执行 commit 或任何 git 相关操作。** 即使是 pr-review，也只能通过 curl 来拉取 pr 信息。代码库管理、分支创建、PR 提交由人工手动执行。

## 全流程

```
  准备：目标库就位 + 源 Conda 环境 + 模型下载
                            │
                            ▼
              ┌──────────────────────────────────────┐
              │ Step 0  diffsynth-analyze-target     │
              │ 分析目标库，生成蓝图                    │
              └──────────────────┬───────────────────┘
                                 ▼ ✅ 确认蓝图
              ┌──────────────────────────────────────┐
              │ Step 1  diffsynth-environment        │
              │ 创建环境，双向验证                      │
              └──────────────────┬───────────────────┘
                                 ▼
              ┌──────────────────────────────────────┐
              │ Step 2  diffsynth-model-code         │
              │ 编写模型代码，一致性验证             │
              └──────────────────┬───────────────────┘
                                 ▼
              ┌───────────────────────────────────────┐
              │ Step 3  Pipeline 接入                  │
              │                                       │
              │ diffsynth-pipeline ✅ 确认效果 ◄──┐    │
              │ diffsynth-pipeline-lowvram       │    │
              │ diffsynth-pipeline-training──────┘    │
              └───────────────────┬───────────────────┘
                                  ▼
              ┌──────────────────────────────────────┐
              │ Step 4  diffsynth-style              │
              │ 统一代码风格                           │
              └──────────────────┬───────────────────┘
                                 ▼
            ┌────────────────────┴───────────────────┐
            ▼                                        ▼
┌────────────────────────────┐  ┌────────────────────────────┐
│ Step 5  diffsynth-         │  │ Step 6  diffsynth-docs     │
│           testing          │  │ 文档生成                    │
│ 全量测试（可选）              │  │ 中英文文档+README           │
└────────────────┬───────────┘  └───────────────┬────────────┘
                 │                              │
                 ▼ ✅ 确认输出                   │
                 └───────────────┬──────────────┘
                                 ▼
              ┌──────────────────────────────────────┐
              │ Step 7  diffsynth-pr                 │
              │ 扫描代码，生成 PR 描述               │
              └──────────────────┬───────────────────┘
                                 ▼ ✅ Review PR
              ┌──────────────────────────────────────┐
              │ Step 8  diffsynth-pr-review          │
              │ 处理 Review 评论，修复验证           │
              └──────────────────────────────────────┘
```

## 如何与 agent 沟通

**重要**：本 workflow 包含 11 个 skill、大量文件生成和跨目录引用，上下文非常庞大。整个流程通常无法在同一个 session 内完成——每次 skill 调用可能在一个全新的 session 中执行，agent 没有之前对话的记忆。因此，每次发起任务时请提供完整的上下文信息。

**一句话模板**：`"{模型名} 的 {当前状态}，现在 {想做什么}，执行 {skill 名称}"`

#### 好的沟通方式

```
✅ "我要接入 ernie-image 模型，目标库在 packages/ernie-image/ernie-image/，
   DiffSynth-Studio 在 packages/ernie-image/DiffSynth-Studio/，
   请先分析目标库，执行 diffsynth-analyze-target"

✅ "ernie-image 的蓝图已确认（见 .sisyphus/integration-blueprints/ernie-image-blueprint.md），
   现在执行环境创建，用 diffsynth 作为源环境 clone，执行 diffsynth-environment"

✅ "ernie-image 环境已就绪，模型代码已接入完成（DiT + VAE + TextEncoder），
   现在接入 Text-to-Image 功能，执行 diffsynth-pipeline"

✅ "ernie-image 的 Text-to-Image Pipeline 已接入，推理脚本在 model_inference/，
   现在生成文档，执行 diffsynth-docs"
```

#### 不好的沟通方式

```
❌ "分析目标库"              → 哪个模型？
❌ "执行环境创建"            → 哪个模型？源环境是什么？
❌ "接入 Pipeline"           → 哪个模型？哪个功能？
❌ "继续下一步"              → 当前在哪一步？哪个模型？
❌ "生成文档"                → 哪个模型？Pipeline 完成了吗？
```

#### 每次沟通建议包含

| 要素 | 示例 |
|------|------|
| 模型名称 | "ernie-image" |
| 当前状态 | "蓝图已确认" / "环境已就绪" / "模型代码已完成" / "Pipeline 已接入" |
| 想做什么 | "接入 Text-to-Image 功能" |
| 调用的 skill | "执行 diffsynth-pipeline" |

> **进阶技巧**：在 prompt 中附带关键路径信息（如蓝图路径、已完成的功能列表、模型软链接位置等），可以让新 session 的 agent 快速定位上下文，减少不必要的文件搜索和猜测。

## 快速开始

### 开始前的准备

**1. 目标库**

将目标模型仓库 clone 或复制到 `{workspace_root}/packages/{model-name}/` 下。`{model-name}` 是你自己定义的模型标识（如 `ernie-image`）。

```
packages/{model-name}/
├── {target-library}/          # 目标模型库
│   └── models/                # 真实目录
│       └── model-owner/
│           └── Model-Name/ → ~/.cache/modelscope/hub/models/model-owner/Model-Name
├── DiffSynth-Studio/         # DiffSynth 框架（如不存在，agent 会自动 clone）
│   └── models/                # 真实目录
│       └── model-owner/
│           └── Model-Name/ → ~/.cache/modelscope/hub/models/model-owner/Model-Name
└── .sisyphus/                # 中间报告和日志（由 skill 自动生成）
    ├── plans/                # Plan 文件
    ├── skill_work_report/    # 渐进式步骤报告
    ├── user_report/          # 向用户报告
    ├── execution-logs/       # 执行日志
    ├── tests/                # 测试结果
    ├── integration-blueprints/  # 蓝图报告
    ├── code-reports/         # 代码阅读报告
    ├── reports/              # Pipeline 设计蓝图
    ├── pr/                   # PR 相关文件
    └── pr-review/            # PR Review 反馈
```

> **命名建议**：外层 `{model-name}` 尽量与原仓库名一致（如 `ernie-image`）。

**2. 源 Conda 环境**

准备一个已经安装好 DiffSynth 的 conda 环境，用于 clone 出新的工作环境。

```bash
conda activate your-diffsynth-env
python -c "import diffsynth; print('DiffSynth OK')"
```

> 这个环境名称会记录在 config.yaml 的 `env_clone_source` 中。agent 会在首次接入时引导你配置。

**3. 模型文件（可选）**

提前下载好模型文件，系统会创建软链接指向 `~/.cache/modelscope/hub/models/`。

```bash
modelscope download --model model-owner/model-name --local_dir ~/.cache/modelscope/hub/models/model-owner/model-name
```

**推荐的模型目录结构**（手动创建软链接可以让后续步骤更稳定）：

```
~/.cache/modelscope/hub/models/model-owner/Model-Name/          # ModelScope 缓存（真实文件）

packages/{model-name}/{target-library}/models/
└── model-owner/Model-Name  → ~/.cache/modelscope/hub/models/model-owner/Model-Name

packages/{model-name}/DiffSynth-Studio/models/
└── model-owner/Model-Name  → ~/.cache/modelscope/hub/models/model-owner/Model-Name
```

**创建方式**：

```bash
# 目标库
mkdir -p {target_path}/models/model-owner
ln -sfn ~/.cache/modelscope/hub/models/model-owner/Model-Name {target_path}/models/model-owner/Model-Name

# DiffSynth
mkdir -p {diffsynth_root}/models/model-owner
ln -sfn ~/.cache/modelscope/hub/models/model-owner/Model-Name {diffsynth_root}/models/model-owner/Model-Name
```

> **说明**：`models/` 是真实目录，内部每个模型 repo 单独软链接到 ModelScope 缓存。目标库和 DiffSynth 各自拥有独立的 `models/`，但软链接指向同一个缓存路径。
>
> 如果不手动创建，后续的 environment skill 也会自动帮你创建软链接。提前准备可以避免 agent 询问时中断流程。

> 也可以先跳过，等环境准备步骤中 agent 向你索要时再下载。提前准备可以节省等待时间。

### 完整 Workflow 流程

三项都准备好后，按照以下步骤逐步执行。每一步都由你显式控制，不自动推进。

---

#### Step 0: 分析目标库

```
用户: "我要接入 ernie-image 模型，目标库在 packages/ernie-image/ernie-image/，
     DiffSynth-Studio 在 packages/ernie-image/DiffSynth-Studio/，
     请先分析目标库，执行 diffsynth-analyze-target"
```

agent 会：
1. 在 `packages/` 下找到目标库
2. 自动 clone DiffSynth-Studio（如不存在）
3. 读取所有下游 skill，理解它们需要什么信息
4. 分析目标库：识别模型组件（DiT/VAE/TextEncoder 等）、Pipeline 类型、依赖差异
5. 生成蓝图报告 + 代码阅读报告

**你需要做什么**：阅读蓝图报告，确认分析结果是否正确。

**产出物**：
- `packages/ernie-image/.sisyphus/integration-blueprints/ernie-image-blueprint.md`
- `packages/ernie-image/.sisyphus/code-reports/code-report.md`

> 蓝图是后续所有 skill 的数据源。蓝图有误，后续全错。务必仔细确认。

---

#### Step 1: 环境创建

```
用户: "ernie-image 的蓝图已确认（见 .sisyphus/integration-blueprints/ernie-image-blueprint.md），
     现在执行环境创建，用 diffsynth 作为源环境 clone，执行 diffsynth-environment"
```

agent 会在同一个 conda 环境中，确保 DiffSynth-Studio 和目标库都能正常运行。验证序列：`DiffSynth OK → Target OK → DiffSynth still OK`

**流程**：
1. 克隆源环境 → 创建 conda 环境
2. 安装 DiffSynth → import 验证通过
3. 推荐模型下载 → 等待你下载完成
4. 创建模型软链接
5. 分析依赖差异 → 分批安装缺失包
6. 验证目标库能运行
7. 再次验证 DiffSynth 未被破坏

**你需要做什么**：下载模型时暂停，等全部下载完成后回复"已下载完成"。

---

#### Step 2: 模型代码接入

```
用户: "ernie-image 环境已就绪（环境名见蓝图报告），
     现在接入模型代码，执行 diffsynth-model-code"
```

agent 会将目标模型的组件（DiT、VAE、TextEncoder 等）接入 DiffSynth-Studio。

**核心原则**：
- 自动编写模型的 `nn.Module` 源代码，只调整依赖问题
- 编写 `state_dict_converter` 对齐权重 key，保持与原始 checkpoint 一致
- 注册到 `model_configs.py`
- 以实际打印的参数值为准，不猜测

**验证**：原库 forward vs DiffSynth forward → `allclose(atol=1e-5)`

---

#### Step 3: Pipeline 接入（每次一个功能）

```
用户: "ernie-image 模型代码已接入（DiT/VAE/TextEncoder），
     现在接入 Text-to-Image 功能，执行 diffsynth-pipeline"
```

agent 会：
1. 分析目标库的 Pipeline 实现，理解数据流和参数签名
2. 创建 DiffSynth Pipeline 文件，设计 Unit 链（Scheduler/VAE/DiT/TextEncoder 等）
3. 生成推理示例脚本到 `model_inference/` 目录
4. 运行端到端一致性验证：原库输出 vs DiffSynth 输出

**后续功能追加**：

```
用户: "ernie-image 的 Text-to-Image Pipeline 已接入，
     现在追加 Image-to-Image 功能，执行 diffsynth-pipeline"
```

agent 会在已有 Pipeline 文件中追加 Unit 和参数，而不是创建新文件。

> Pipeline 文件按系列组织，不按功能拆分。一个模型系列一个 Pipeline 文件。

---

#### Step 3a: 低显存推理（可选）

```
用户: "ernie-image 的 Pipeline 已接入（Text-to-Image），
     现在接入低显存推理，执行 diffsynth-pipeline-lowvram"
```

agent 会：
1. 在 `vram_management_module_maps.py` 中注册 module_map
2. 为每个已接入功能生成 `model_inference_low_vram/` 脚本
3. 注入 `vram_config`，实现 CPU offload

---

#### Step 3b: 训练模块接入（可选）

```
用户: "ernie-image 的 Pipeline 已接入（Text-to-Image），
     现在接入训练模块，执行 diffsynth-pipeline-training"
```

agent 会：
1. 确认/创建 `InputXXXEmbedder` Unit
2. 创建 `train.py` + `.sh` 配置 + 验证脚本
3. 准备数据集
4. LoRA debug 验证 → 完整训练验证

---

#### Step 4: 代码风格化

```
用户: "ernie-image 的 Pipeline 和低显存/训练模块已全部接入，
     现在统一代码风格，执行 diffsynth-style"
```

统一新建/修改文件的代码风格：
- ModelConfig 一行写完
- `extra_kwargs` 一行写完
- Pipeline dict 按参数组换行
- Import 紧凑组织

> 纯视觉层面修改，不改变任何运行时行为。

---

#### Step 5: 全量测试

```
用户: "ernie-image 的代码已风格化，推理脚本在 model_inference/ 和 model_inference_low_vram/，
     现在运行全量测试，执行 diffsynth-testing"
```

agent 会原样运行所有推理脚本（不修改任何代码），收集输出和报错。

**策略**：
- mtime 快照检测 + 立即移动，不删除文件
- 失败不阻塞，全部运行完统一报告
- 默认跳过模型下载

**产出物**：
- 测试报告：`packages/ernie-image/.sisyphus/tests/latest/test_report.md`
- 输出审查：`packages/ernie-image/.sisyphus/tests/latest/{script_name}/`

**你需要做什么**：检查成功脚本的输出文件质量，确认失败脚本是否需要修复。

---

#### Step 6: 文档生成

```
用户: "ernie-image 的推理脚本已全部测试通过，
     现在生成中英文文档，执行 diffsynth-docs"
```

agent 会生成中英文文档 + 更新 README + 更新 index.rst。

**信息源**：从已接入的代码提取（Pipeline 参数签名、推理脚本列表、训练脚本路径），不虚构内容。

**需要你提供的**（可选，不阻塞生成）：
- 一句话模型描述

---

#### Step 7: PR 描述生成

```
用户: "ernie-image 的文档已生成，
     现在扫描代码生成 PR 描述，执行 diffsynth-pr"
```

agent 会扫描实际代码（Pipeline 文件、模型组件、推理脚本），生成 PR 描述 Markdown 文件。

**重要**：
- 以实际代码为准，不直接复制蓝图中的功能列表
- 仅生成 Markdown 文件，不执行任何 git 操作
- 你 Review 确认后再自行创建分支、提交、推送

**产出物**：`packages/ernie-image/.sisyphus/pr/description.md`

---

#### Step 8: PR Review 反馈处理

```
用户: "ernie-image 的 PR 已提交，PR 号是 #1389，
     现在处理 PR Review 反馈，执行 diffsynth-pr-review"
```

agent 会：
1. 抓取 PR 上的所有 Review 评论
2. 分析每条建议，生成修改蓝图
3. 向你展示修改规划，确认后执行
4. 针对性修复 + 验证（推理 smoke test / 训练 10 步）

> 本地代码就是 PR 分支，直接修改。不需要 checkout 或 fetch。

---

### 跳过不需要的 step

不是每次接入都需要走完所有步骤。

**version_upgrade：只注册新 hash**

```
用户: "LTX-2.3 升级，模型结构没变，请先分析目标库，执行 diffsynth-analyze-target"

→ 蓝图确认：无依赖变化、无结构变化

用户: "ernie-image 蓝图已确认，无结构变化，执行模型代码接入，执行 diffsynth-model-code"

→ 判断：结构无变化 → 只注册新 hash → 完成

用户: "ernie-image 环境需要更新吗？执行 diffsynth-environment"

→ 判断：无新增依赖 → 跳过 → 完成
```

**不需要训练**

```
用户: "这个模型不支持训练，跳过训练，ernie-image 的 Pipeline 已接入，直接进入代码风格化，执行 diffsynth-style"

→ 不执行 diffsynth-pipeline-training
→ 从 Step 3 直接到 Step 4
```
