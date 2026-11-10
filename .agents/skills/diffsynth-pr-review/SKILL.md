---
name: diffsynth-pr-review
description: >
  Review PR feedback comments, generate a modification blueprint, and implement targeted
  fixes for DiffSynth-Studio model integration PRs. The user has already submitted a PR
  and reviewers have left comments. Use this skill to analyze those comments, generate a
  modification blueprint, and implement targeted fixes on the local codebase.
  Use when the user provides a PR URL, PR number, or mentions PR review comments, feedback,
  or suggestions that need to be addressed on a model/feature integration PR. This skill supports both PR URLs
  (e.g., https://github.com/modelscope/DiffSynth-Studio/pull/1389) and bare PR numbers.
  The local codebase is already the PR branch -- just work with what's on disk.
  If changes involve training or test files, this skill adds minimal smoke tests to verify them.
  SCOPE: Only for PRs that integrate a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for reviews of any other PR — bug fixes, refactoring, performance tuning, or general code
  changes.
---

# DiffSynth-Studio: PR Review 反馈处理

读取 PR 上的 Review 反馈，分析每条建议，生成修改蓝图，等用户确认后做针对性修改。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**PR 输入**：用户需提供以下任一形式的 PR 标识（用于抓取 Review 反馈）：
- 完整 PR URL（如 `https://github.com/modelscope/DiffSynth-Studio/pull/1389`）
- 仅 PR 编号（如 `1389`，默认仓库为 `modelscope/DiffSynth-Studio`）

## 核心原则

### 1. 反馈分类优先

PR 上的反馈可能来自人类 Reviewer 或自动化工具（如 Gemini Code Assist、GitHub Copilot）。在开始修改前，必须先分类整理：

| 类别 | 说明 | 示例 |
|------|------|------|
| Bug / 正确性 | 代码存在运行时错误或逻辑错误 | 空值未处理导致崩溃、设备不匹配 |
| 代码质量 | 不影响运行但违背项目规范或最佳实践 | 硬编码常量、缺少 docstring |
| 功能增强 | 建议性改进，非必须修复 | 添加额外参数、优化性能 |
| 风格 / 格式 | 纯格式问题 | 缩进、命名风格 |

### 2. 以代码为准，不盲从蓝图

蓝图是接入前的计划文档，实际代码可能已发生变化。分析反馈和制定修改方案时，**以磁盘上的实际代码为准**。

### 2. 不盲从 review 建议，以原有代码意图为准

对每条 review 建议（尤其是 AI 工具自动生成的），**先追溯原有代码的编写逻辑，再决定是否采纳**。PR review 中的建议往往只看到了局部，不一定理解代码在整体 pipeline 中的角色。具体做法：

1. **追溯原有代码的意图**：读取修改涉及的文件（不仅是变更行，还要读上下文和调用方），理解这段代码当时为什么这么写——它服务于什么功能、有什么依赖、和上下游如何交互。如果原有代码是有意为之且功能正确，保留原有逻辑
2. **判断建议是否合理**：review 建议可能不理解项目上下文，提出看似正确但实际会破坏原有逻辑的修改。特别警惕：改变参数默认值、删除"冗余"代码、重命名变量等建议，它们可能忽略了上下游的隐式依赖
3. **只在必修时才改**：确认真实存在 bug、规范违背、或明显可改进的点才采纳；如果只是风格偏好或建议者没有理解原有意图，可以不采纳并在蓝图中说明「为什么不改」的理由

**核心判断标准**：如果原有代码能正常工作、有明确的编写目的，即使 review 建议看起来"更好"，也不要做不必要的改动。改动带来的风险往往大于收益。

### 3. 整体决定最小测试策略

测试不是逐条反馈决定——而是**整体评估所有采纳的修改涉及哪些模块**，然后统一决定：

| 判断条件 | 测试类型 | 要求 |
|----------|---------|------|
| 有任一修改涉及推理相关代码（pipeline、模型、VAE、scheduler、脚本） | 推理最小测试 | 根据修改内容规划需要的推理脚本，完整运行每个脚本，记录执行过程并保存输出 |
| 有任一修改涉及训练相关代码（training script、loss、optimizer、data loader） | 训练最小测试 | 运行 LoRA 训练，执行 10 个 step，不验证代码 |

**规则**：
- 推理测试和训练测试各自只需整体决定「做」或「不做」，不需要为每条修改单独创建测试
- **整体规划测试**：根据采纳的修改涉及哪些模块、哪些脚本，规划需要运行哪些已有脚本来覆盖这些修改。测试脚本数量不限，以能覆盖所有修改为准。如果修改涉及多个不同的推理功能（如文生图和图生图），则运行多个对应的推理脚本；如果只涉及单一功能，运行一个脚本即可。核心原则是：分析修改内容 → 规划需要的脚本 → 确保选中的脚本集合运行后能覆盖所有修改
- **最小测试的含义**：「最小」指的是**用最少的脚本数量和训练 step 覆盖所有修改**，不是降低测试质量。推理脚本必须完整运行其全部逻辑，**严禁篡改脚本的任何运行参数**（如 height、width、inference_steps、num_inference_steps、guidance_scale、num_steps、batch_size 等），不得缩短、跳过、改小任何参数来减少运行时间。训练测试只做 10 个 step 是唯一的简化，除此之外所有参数保持脚本原始配置
- **推理最小测试**：
  - 复用已有的推理脚本（如 `examples/{model_name}/model_inference/` 下的脚本），**完整运行整个脚本**
  - 必须符合 [diffsynth-testing](../diffsynth-testing/SKILL.md) 的方案：执行过程要有记录
- **训练最小测试**：
  - 参考 [diffsynth-pipeline-training](../diffsynth-pipeline-training/SKILL.md) 的 "LoRA 完整训练" 模式
  - 启动 `examples/{series}/model_training/lora/{ModelName}.sh`，观察 loss 正常输出
  - 等待约 10 个 step 后 kill 停掉训练
  - **脚本为双阶段（含两条 `accelerate launch`）时**：第一阶段 `sft:data_process` 不产生 loss，必须先跑完，进入 `sft:train` 后再计 step
  - **只做 LoRA 训练**（不做 full training），**不需要运行验证脚本**

### 4. 参考其他 skill 的约定

本 skill 的目标是完成最终的代码修改，修改后的代码必须融入 DiffSynth-Studio 的整体生态。因此在执行修改时，应参考其他 skill 的约定和规范：

- 修改模型文件、state_dict_converter、model_configs.py 注册时，参考 [diffsynth-model-code](../diffsynth-model-code/SKILL.md) 的模型组件规范
- 修改 Pipeline 文件时，参考 [diffsynth-pipeline](../diffsynth-pipeline/SKILL.md) 的 Pipeline 编写规范
- 修改训练相关代码时，参考 [diffsynth-pipeline-training](../diffsynth-pipeline-training/SKILL.md)
- 修改推理脚本、测试脚本时，参考 [diffsynth-testing](../diffsynth-testing/SKILL.md) 的运行方式（注意：testing skill 只负责运行脚本验证，不涉及代码内容修改）
- 所有修改完成后，确保代码风格与 [diffsynth-style](../diffsynth-style/SKILL.md) 一致

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pr-review-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pr-review-report.md << EOF

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
# 从 CLAUDE.md 或 config.yaml 获取模型名称
MODEL_NAME="{model-name}"
BLUEPRINT_PATH="packages/${MODEL_NAME}/.sisyphus/integration-blueprints/${MODEL_NAME}-blueprint.md"
```

**本 skill 必须从蓝图报告中读取的信息：**

| 蓝图信息 | 用途 |
|---------|------|
| 基本信息表中的 `Conda 环境名称` | 测试环境。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。** |
| Pipeline 功能规划表 | Pipeline 类名、系列名称、推理脚本路径 |
| 训练相关信息（如有） | 训练脚本路径、训练方式 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始处理 PR 反馈前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/pr-review-plan.md`。** 基于 PR 信息和已确认的修改范围，明确反馈来源、分类策略、修改方案、测试规划和执行顺序。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示，确认后再开始处理反馈。**

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 3. 获取 PR 反馈、分析反馈与生成修改蓝图

> 📖 开始前：重读本步骤描述，确认流程与报告路径

本步骤包含两个连续动作：抓取 PR 反馈 → 分析反馈并生成修改蓝图。

#### 3a. 获取 PR 反馈

使用 `curl` 请求 GitHub 公开 API 获取 PR 反馈。**公开仓库不需要任何认证**，`curl` 系统自带，无额外依赖。

```bash
API="https://api.github.com/repos/modelscope/DiffSynth-Studio"

# 1. 获取 PR 基本信息（标题、body、变更统计、分支名、diff_url）
curl -s "$API/pulls/{PR_NUMBER}"

# 2. 获取 PR-level Reviews（Gemini Code Assist、Copilot 等工具的总体评审）
curl -s "$API/pulls/{PR_NUMBER}/reviews"

# 3. 获取逐行代码评审评论（inline comments，包含具体文件和行号）
curl -s "$API/pulls/{PR_NUMBER}/comments"

# 4. 获取 PR 正文讨论区的普通评论
curl -s "$API/issues/{PR_NUMBER}/comments"

# 5. 获取完整 diff（通过 PR 信息中的 diff_url 字段）
DIFF_URL=$(curl -s "$API/pulls/{PR_NUMBER}" | python3 -c "import json,sys; print(json.load(sys.stdin)['diff_url'])")
curl -sL "$DIFF_URL"
```

**收集内容**：
1. PR 的基本信息（标题、文件变更列表）
2. 所有 Review 评论（包括 AI 工具自动生成的反馈）
3. 普通 PR 评论（人类 Reviewer 的手动评论）
4. 完整的 diff 变更内容

将原始反馈保存到 `packages/{model-name}/.sisyphus/pr-review/raw-feedback.md`。

#### 3b. 分析反馈与生成修改蓝图

对每一条 Review 反馈进行分析并生成合并后的分析蓝图文档。

1. **分类**：按照核心原则中的分类表归类（Bug / 正确性、代码质量、功能增强、风格 / 格式）
2. **定位**：指出反馈涉及的具体文件路径和代码行
3. **评估影响**：判断该反馈是否合理、是否需要修改
4. **确定修改方案**：每条反馈对应一个具体的修改动作
5. **生成蓝图**：读取当前代码，写出修改后代码，规划测试

输出到 `packages/{model-name}/.sisyphus/pr-review/analysis-modification-blueprint.md`。文件标题为「PR Review 反馈分析与修改蓝图」，包含以下章节：

```markdown
# PR Review 反馈分析与修改蓝图

## 反馈分析表
| 序号 | 来源 | 类别 | 文件 | 行号 | 问题描述 | 是否采纳 | 修改方案 | 引入风险 |
|------|------|------|------|------|----------|----------|----------|----------|
| 1 | Gemini Code Assist | Bug / 正确性 | xxx.py | 123 | ... | 是 | ... | {采纳后可能破坏的原有逻辑/功能} |

> **是否采纳**的判断标准：
> - **Bug / 正确性**：一律采纳
> - **代码质量**：评估修改成本，低成本的采纳
> - **功能增强**：仅当实现简单且不影响核心功能时采纳，否则标注为「建议后续迭代」
> - **风格 / 格式**：如果项目有统一的代码风格规范则采纳，否则可忽略

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pr-review |
| 执行时间 | {timestamp} |
| PR 地址 | {PR URL} |
| 反馈总数 | {N} 条 |
| 采纳数量 | {M} 项 |

## 修改方案

按修改项逐一描述每个改动的前后对比。

### 修改项 1: {简短标题}

- **对应反馈**: #{序号} ({来源})
- **类别**: {Bug / 代码质量 / 功能增强 / 风格格式}
- **涉及文件**: `{path/to/file.py}`
- **当前代码**: （粘贴当前代码片段）
- **修改后代码**: （粘贴修改后的代码）
- **修改说明**: {简要说明为什么这样改}

---

### 修改项 2: {简短标题}

- **对应反馈**: #{序号} ({来源})
- ...（同上结构）

---

## 测试规划

根据采纳的修改涉及的模块，整体决定需要运行哪些测试脚本。

### 推理测试脚本
| 脚本路径 | 用途 | 覆盖的修改项 |
|----------|------|-------------|
| `examples/{model_name}/model_inference/xxx.py` | 文生图推理 | 修改项 1、修改项 3 |
| `examples/{model_name}/model_inference/yyy.py` | 图生图推理 | 修改项 5 |

- 决策：执行 / 不执行
- 原因：{哪些修改涉及推理代码}
- 执行目录：`packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/`
- 输出审查：`packages/{model-name}/.sisyphus/tests/$(date +%Y%m%d_%H%M%S)/`
- **严禁篡改脚本参数**：必须原样完整运行，不得修改 height、inference_steps 等任何参数
- 符合 diffsynth-testing 方案：执行记录 + 输出保存

### 训练测试脚本
| 脚本路径 | 覆盖的修改项 |
|----------|-------------|
| `examples/{model_name}/model_training/lora/{ModelName}.sh` | 修改项 2 |

- 决策：执行 / 不执行
- 原因：{哪些修改涉及训练代码}
- 运行方式：`cd {diffsynth_root} && CUDA_VISIBLE_DEVICES=0 bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log`
- 观察日志输出，确认 loss 正常打印，运行够 10 个 step 后 kill 停掉训练
- 双阶段脚本：先跑完 `sft:data_process`，进入 `sft:train` 后再计 step
- 不运行验证脚本，只确认训练能跑起来

### 测试覆盖确认
- 修改项 1: 被 `xxx.py` 覆盖
- 修改项 2: 被 `train_lora.py` 覆盖
- 修改项 3: 被 `xxx.py` 覆盖
- ...
- 未覆盖的修改项: {如有，说明原因}
```

**蓝图生成后，必须展示给用户并等待确认**，不要自动开始修改。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 4. 用户确认

> 📖 开始前：重读本步骤描述，确认流程与报告路径

将修改蓝图展示给用户，包括：

1. 每条反馈的分类和修改方案
2. 每条修改涉及的具体文件改动（diff 预览）
3. 如果需要，最小测试脚本的计划

**必须等用户确认后才能继续。** 如果用户要求调整某些修改方案，重新修改蓝图。

用户确认后，在蓝图文件末尾追加确认记录：

**用户确认**：
- 确认时间: {timestamp}
- 确认修改项数量: {N} 项
- 用户备注: {如果有}

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 5. 执行修改

> 📖 开始前：重读本步骤描述，确认流程与报告路径

按照蓝图逐一执行修改。

**修改规则**：

1. **逐条执行**：按蓝图中的修改项顺序执行，完成一项后再执行下一项
2. **保留中间状态**：修改过程中不要提交 git commit，所有修改完成后统一提交
3. **修改前先读取文件**：不要凭记忆修改，每次都先 Read 目标文件确认当前内容
4. **记录每次修改**：每次修改后追加记录到执行日志
5. **保持风格一致**：修改后的代码应遵循 [diffsynth-style](../diffsynth-style/SKILL.md) 的代码规范，保持紧凑、美观、可读

修改完成后，保存修改摘要到 `packages/{model-name}/.sisyphus/pr-review/modification-summary.md`，内容包含：

- 已修改的文件列表
- 每个文件的变更说明
- 对应反馈编号

查看变更确认：

```bash
git diff --stat
git diff
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 6. 执行最小测试（如需要）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**整体评估**：根据蓝图中「测试规划」的结论，执行对应的测试。

#### 推理最小测试（如需执行）
- 直接使用已有的推理脚本（如 `examples/{model_name}/model_inference/xxx.py`）
- **完整运行整个脚本代码**，不跳过任何步骤
- 必须符合 [diffsynth-testing](../diffsynth-testing/SKILL.md) 的方案：
  - 初始化执行日志目录：`packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/`
  - 输出审查目录：`packages/{model-name}/.sisyphus/tests/$(date +%Y%m%d_%H%M%S)/`
  - 采用 mtime 检测 + 立即移动策略收集输出文件到 `outputs/generated/{script_name}/`
  - 保存运行日志和测试结果
- 不验证输出结果的正确性，只确认整个流程能跑通不报错

#### 训练最小测试（如需执行）

参考 [diffsynth-pipeline-training](../diffsynth-pipeline-training/SKILL.md) 的 "LoRA 完整训练" 模式，但简化为最小验证：

**先判断脚本是单阶段还是双阶段**：

```bash
grep -c 'accelerate launch' examples/{series}/model_training/lora/{ModelName}.sh
```

- 结果为 `1` → 单阶段，按下方「单阶段」执行
- 结果 `>1` → 双阶段（拆分训练），按下方「双阶段」执行

**单阶段**：

1. 启动 LoRA 训练脚本，后台运行并 tee 日志：
   ```bash
   cd {diffsynth_root}
   CUDA_VISIBLE_DEVICES=0 bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log &
   TRAIN_PID=$!
   ```
2. 观察日志，确认 loss 正常输出，等待约 10 个 step
3. 停掉训练：
   ```bash
   kill $TRAIN_PID 2>/dev/null
   wait $TRAIN_PID 2>/dev/null
   ```

**双阶段**：第一阶段是 `--task "sft:data_process"`，只做前处理缓存、**不产生 loss**。在这个阶段 kill 等于完全没有覆盖训练代码。

1. 先让第一阶段跑完（数据集很小，通常很快），不要中途 kill
2. 进入第二阶段（`--task "sft:train"`）后再观察 loss，等约 10 个 step 后 kill
3. 判断阶段的方法：日志中出现第二条 `accelerate launch` 对应的输出，或出现 loss 数值即表示已进入训练阶段
4. 为缩短第一阶段耗时，可临时把两阶段的 `--dataset_repeat` 都改为 `1`、`--num_epochs` 改为 `1`（改动仅用于本次冒烟测试，不要提交）

**两种情况共同要求**：

- **只做 LoRA 训练**，不做 full training
- 不运行验证脚本，只确认训练能跑起来

**注意**：
- 不需要创建新的 smoke test 脚本，直接复用已有的推理/训练脚本
- 根据修改内容规划需要的脚本数量，确保所有修改均被覆盖，脚本数量不限
- 推理测试必须有执行记录和输出保存，符合 diffsynth-testing 的方案

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

### 7. 验证修改

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有修改和测试完成后，执行最终验证：

```bash
# 1. 检查所有修改文件是否存在语法错误
for f in $(git diff --name-only -- '*.py'); do
  python -m py_compile "$f" 2>&1 || echo "FAIL: $f has syntax errors"
done

# 2. 确认所有报告文件已生成
for f in \
  "packages/{model-name}/.sisyphus/pr-review/analysis-modification-blueprint.md" \
  "packages/{model-name}/.sisyphus/pr-review/modification-summary.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 3. 检查报告文件完整性
for f in \
  "packages/{model-name}/.sisyphus/skill_work_report/pr-review-report.md" \
  "packages/{model-name}/.sisyphus/user_report/pr-review-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 4. 查看最终变更
echo "=== 修改统计 ==="
git diff --stat
```

如有缺失，立即补充。如果有最小测试脚本，确认所有测试均通过。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-review-report.md`

## 输出

### 执行日志

所有执行过程保存到：

- **PR Review 执行日志**: `packages/{model-name}/.sisyphus/pr-review/`
  - `analysis-modification-blueprint.md` — 反馈分析表 + 修改蓝图（合并）
  - `modification-summary.md` — 修改摘要
  - `raw-feedback.md` — 原始 PR 反馈

- **推理最小测试**（如执行）：遵循 diffsynth-testing 的目录规范
  - 执行日志: `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/`
  - 输出审查: `packages/{model-name}/.sisyphus/tests/$(date +%Y%m%d_%H%M%S)/`
  - 测试报告: `packages/{model-name}/.sisyphus/tests/latest/test_report.md`
  - 输出文件按脚本名分目录存放: `tests/latest/{script_name}/`

- **训练最小测试**（如执行）：
  - 执行日志: `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/outputs/lora_train.log`

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/pr-review-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# PR Review 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pr-review |
| 执行时间 | {timestamp} |
| PR 地址 | {PR URL} |
| 反馈总数 | {N} 条 |
| 采纳数量 | {M} 项 |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取 PR Review 所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、Pipeline 功能规划、推理/训练脚本路径
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

**注意事项**：
- Conda 环境名称必须从蓝图报告的「基本信息」表格中获取，不自行推断。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。**

---

### Step 1: 初始化执行日志目录

**目标**：创建标准化的执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review/` 目录及子目录
- 用于存放后续的执行日志、测试输出和脚本

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示完整处理规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/pr-review-plan.md`
- 向用户展示反馈来源、修改范围、测试规划、执行顺序
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/pr-review-plan.md`

**注意事项**：
- 执行计划需明确列出涉及的所有文件和测试脚本

---

### Step 3: 获取 PR 反馈、分析反馈与生成修改蓝图

**目标**：抓取 PR 上的所有 review 反馈，分类分析，生成合并后的分析蓝图文档（含反馈分析表 + 修改方案 + 测试规划）。

**执行内容**：
- 使用 `curl` 请求 GitHub 公开 API 获取 PR 基本信息、reviews、inline comments、普通评论
- 使用 `curl -sL` 通过 diff_url 获取完整代码变更
- 将原始反馈保存到 `packages/{model-name}/.sisyphus/pr-review/raw-feedback.md`
- 按照 Bug/代码质量/功能增强/风格格式分类反馈
- 追溯原有代码意图，评估是否采纳，制定修改方案
- 读取当前代码，写出修改后代码，规划测试
- 输出到 `analysis-modification-blueprint.md`
- 展示给用户等待确认

**产出物**：
- `raw-feedback.md`
- `analysis-modification-blueprint.md`（反馈分析表 + 修改蓝图，描述怎么改、怎么测）

**注意事项**：
- 蓝图包含反馈分析表、基本信息表、修改方案（每个修改项前后代码对比）、测试规划章节
- 蓝图生成后必须展示给用户并等待确认

---

### Step 4: 用户确认

**目标**：等待用户确认修改蓝图。

**执行内容**：
- 展示每条反馈的分类和修改方案
- 展示 diff 预览
- 用户确认后在蓝图文件中追加确认记录

**产出物**：用户确认记录

**注意事项**：
- 必须等用户确认后才能继续
- 用户要求调整时重新修改蓝图

---

### Step 5: 执行修改

**目标**：按蓝图逐一执行修改。

**执行内容**：
- 按蓝图顺序逐条修改，修改前 Read 文件确认当前内容
- 保持代码风格与 diffsynth-style 一致
- 修改后保存修改摘要

**产出物**：修改后的代码文件、`modification-summary.md`

**注意事项**：
- 修改过程中不要提交 git commit
- 不要凭记忆修改，每次都先 Read 目标文件

---

### Step 6: 补充最小测试（如需要）

**目标**：根据蓝图「测试规划」的整体决定，执行推理和/或训练最小测试，确保所有修改均被覆盖。

**执行内容**：

**6a. 推理最小测试**（当有任一修改涉及推理代码时执行）：
- 初始化执行日志目录：`packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/`
- 初始化输出审查目录：`packages/{model-name}/.sisyphus/tests/$(date +%Y%m%d_%H%M%S)/`
- 创建 `latest` 软链接指向本次目录
- 采用 mtime 检测 + 立即移动策略（参考 diffsynth-testing 方案）：
  1. 运行前：递归扫描 DiffSynth-Studio 根目录，记录所有输出文件的 mtime
  2. 运行：在 DiffSynth-Studio 根目录下完整运行推理脚本（`DIFFSYNTH_SKIP_DOWNLOAD=true`）
  3. 运行后：检测新增或 mtime 更新的文件，立即移动到 `outputs/generated/{script_name}/` 独立子目录
  4. 保存运行日志（stdout/stderr）到执行日志目录
- 不验证输出结果正确性，只确认流程能跑通不报错

**6b. 训练最小测试**（当有任一修改涉及训练代码时执行）：

参考 diffsynth-pipeline-training 的 "LoRA 完整训练" 模式，简化为最小验证：
1. 启动 LoRA 训练脚本，后台运行并 tee 日志：
   ```bash
   cd {diffsynth_root}
   CUDA_VISIBLE_DEVICES=0 bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log &
   TRAIN_PID=$!
   ```
2. 观察日志，确认 loss 正常输出，等待约 10 个 step
3. 停掉训练：`kill $TRAIN_PID 2>/dev/null`
4. 双阶段脚本：第一阶段 `sft:data_process` 不产生 loss，先跑完再在 `sft:train` 阶段计 step
5. 不做 full training，不运行验证脚本
6. 训练日志保存到 `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr-review-testing/outputs/lora_train.log`

**6c. 测试覆盖确认**：
- 确认推理测试运行后覆盖了所有推理相关修改
- 确认训练测试运行后覆盖了所有训练相关修改
- 如果有修改未被任何测试覆盖，在测试报告中说明原因

**产出物**：
- 推理测试：执行日志（`execution-logs/`）、输出审查（`tests/latest/{script_name}/`）、运行日志（`{script_name}_output.log`）
- 训练测试：训练日志（`lora_train.log`）
- 测试报告（`tests/latest/test_report.md`）

**注意事项**：
- 推理测试和训练测试各自只需整体决定「做」或「不做」，不需要为每条修改单独创建测试
- 根据修改涉及的模块和功能，规划需要运行的脚本列表，数量不限，确保运行后能覆盖所有修改
- **推理测试严禁篡改脚本参数**：不得修改或缩短 height、width、inference_steps、num_inference_steps、guidance_scale 等任何参数，必须原样完整运行
- 推理测试必须符合 diffsynth-testing 方案（mtime 检测 + 立即移动 + 输出保存），不创建新的 smoke test 脚本
- 训练测试只做 LoRA，参考 diffsynth-pipeline-training 的 "LoRA 完整训练" 模式，运行 10 step 后 kill 停掉，不运行验证脚本

---

### Step 7: 验证修改

**目标**：确认所有修改文件和报告文件完整性。

**执行内容**：
- 检查所有修改文件语法
- 检查报告文件完整性
- 查看最终变更统计

**产出物**：验证通过确认
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pr-review-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-7 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 获取 PR 反馈、分析反馈、生成修改蓝图 | Step 3 |
| 4 | 用户确认 | Step 4 |
| 5 | 执行修改 | Step 5 |
| 6 | 补充最小测试（如需要） | Step 6 |
| 7 | 验证修改 | Step 7 |

### 向用户报告

修改完成后，向用户报告**必须写入文件** `packages/{model-name}/.sisyphus/user_report/pr-review-report.md`。文件内容按以下模板生成：

```bash
cat > packages/{model-name}/.sisyphus/user_report/pr-review-report.md << 'OUTER_EOF'
## ✅ PR Review 反馈处理完成

执行日志: `packages/{model-name}/.sisyphus/pr-review/`

### 修改概览
- PR 地址: {PR URL}
- 反馈总数: {N} 条
- 已采纳修改: {M} 项
- 修改文件: {K} 个

### 修改详情
详见 `packages/{model-name}/.sisyphus/pr-review/modification-summary.md`

### 最小测试
{如果有测试脚本，列出每个测试的状态：通过/失败}

### 后续操作建议
所有修改已在本地完成。请 Review 变更内容，确认无误后手动执行 git 提交和推送。
OUTER_EOF
```

## 注意事项

1. **不要自动提交 git commit**：所有修改完成后等待用户确认，由用户自行提交
2. **反馈来源识别**：区分人类 Reviewer 和自动化工具的反馈，人类反馈通常优先级更高
3. **冲突处理**：如果多条反馈互相矛盾（如一条建议增加参数、另一条建议删除同一参数），向用户说明冲突，等用户决策
4. **范围控制**：只修改反馈中明确提到的问题，不要借机做额外的重构或改进
5. **最小测试整体决策**：「最小」指用最少的脚本数量和训练 step 覆盖所有修改，不代表可以降低测试质量。推理测试必须符合 diffsynth-testing 方案（有执行记录、有输出保存），严禁篡改脚本的任何运行参数（height、width、inference_steps 等）。训练测试参考 diffsynth-pipeline-training 的 "LoRA 完整训练" 模式，运行约 10 step 后 kill 停掉，不运行验证脚本
6. **测试整体覆盖**：蓝图中必须明确列出涉及的脚本、每个脚本覆盖哪些修改、以及还有哪些修改未被覆盖。根据修改内容规划需要的脚本数量，确保选中脚本集合运行后能覆盖所有修改
