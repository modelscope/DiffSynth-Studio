---
name: diffsynth-pr
description: >
  Generate a PR description markdown file for a completed model integration in DiffSynth-Studio.
  Use this skill after all testing is done and the user has confirmed the results, or whenever
  the user wants to generate a PR description for review. This skill only outputs a markdown
  file — it does NOT create branches, commit code, or push to remote.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for PRs from any other development — bug fixes, refactoring, performance tuning, or
  general code changes.
---

# DiffSynth-Studio: PR 描述生成

读取接入信息，生成 PR 描述 Markdown 文件。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

## 工作流程

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
| 基本信息表中的 `Conda 环境名称` | PR 信息 |
| 接入类型 | PR 描述中的集成类型说明 |
| Pipeline 功能规划表 | 接入的功能列表 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pr"
mkdir -p ${EXEC_LOG_DIR}/{outputs,scripts,checkpoints}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

### 2. 以实际代码为准，校验蓝图信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**蓝图是接入前的计划文档，实际代码在开发过程中可能已经变化。生成 PR 描述时必须以磁盘上的实际代码为准。**

**校验流程**：

1. **Pipeline 文件**：扫描 `{diffsynth_root}/diffsynth/pipelines/` 下与本次模型相关的文件，逐行读取 `__call__` 签名和 Pipeline 类名，**不要**依赖蓝图中的功能规划表
2. **模型组件**：扫描 `{diffsynth_root}/diffsynth/models/` 下与本次模型相关的文件，**不要**依赖蓝图中的组件清单
3. **State dict converter**：扫描 `{diffsynth_root}/diffsynth/utils/state_dict_converters/` 下与本次模型相关的文件
4. **推理脚本**：扫描 `{diffsynth_root}/examples/` 下与本次模型相关的脚本（含 `model_inference/`、`model_inference_low_vram/`、`model_training/`），**不要**依赖蓝图中的推理规划
5. **model_configs.py**：读取 `{diffsynth_root}/diffsynth/configs/model_configs.py` 中新增的注册项
6. **显存管理配置**：读取 `{diffsynth_root}/diffsynth/configs/vram_management_module_maps.py` 中新增的 module_map 条目
7. **文档**：读取 `{diffsynth_root}/docs/zh/Model_Details/` 和 `docs/en/Model_Details/` 下本次新增或修改的文档，以及 `docs/{zh,en}/index.rst`、`docs/{zh,en}/README.md` 的改动
8. **根 README**：读取 `{diffsynth_root}/README.md` 和 `README_zh.md` 的改动（更新历史、全局模型表、基础框架分类列表）
9. **依赖变更**：读取 `{diffsynth_root}/pyproject.toml` 的实际内容

> 用 `git status` / `git diff --stat` 核对实际改动文件清单，避免漏项。

**发现不一致时**：蓝图中的功能名、组件名、Pipeline 类名等可能与实际代码不同，**一律以实际代码为准**，不修正代码去匹配蓝图。仅在报告中注明主要差异供用户参考。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

### 3. 收集接入信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**以实际代码为准**，收集生成 PR 描述所需的所有信息：

1. **接入类型**：从蓝图报告读取（new_series / version_upgrade）
2. **功能列表**：从实际 Pipeline 文件的类名和 `__call__` 签名提取（见 Step 2 扫描结果）
3. **模型组件**：从 `{diffsynth_root}/diffsynth/models/` 下实际存在的文件提取
4. **Pipeline 文件**：从 `{diffsynth_root}/diffsynth/pipelines/` 下实际存在的文件提取
5. **State dict converter**：从 `{diffsynth_root}/diffsynth/utils/state_dict_converters/` 下实际存在的文件提取
6. **推理与训练脚本**：从 `{diffsynth_root}/examples/` 下实际存在的脚本提取
7. **显存管理配置**：从 `vram_management_module_maps.py` 实际新增的条目提取
8. **文档改动**：从 `docs/{zh,en}/Model_Details/`、`index.rst`、`docs/{zh,en}/README.md` 的实际改动提取
9. **根 README 改动**：从 `README.md` / `README_zh.md` 的实际改动提取
10. **依赖变更**：从 `{diffsynth_root}/pyproject.toml` 实际内容中提取
11. **模型信息**：从 `model_configs.py` 实际注册项中提取

> 不要直接复制蓝图中的功能列表或组件清单到 PR 描述中。蓝图是计划，代码是事实。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

### 4. 生成 PR 描述

> 📖 开始前：重读本步骤描述，确认流程与报告路径

完整 PR 描述模板见 [references/pr-template.md](references/pr-template.md)。

根据收集到的信息，填充 PR 描述模板，输出到：

```
packages/{model-name}/.sisyphus/pr/description.md
```

**语言要求：PR 描述必须全部使用英文。** 包括标题、摘要、功能列表、组件说明、依赖列表、注意事项等所有内容。不要使用中文或其他非英文语言。DiffSynth-Studio 是面向全球开发者的开源项目，PR 描述使用英文才能让所有贡献者和使用者都能理解。

PR 描述应聚焦于**新功能介绍**，不包含接入过程的测试等细节：
- 接入的模型名称和功能列表（支持哪些 pipeline）
- 模型组件清单（组件名、文件名、参数量）
- 新增的依赖包
- 注意事项（硬件要求、已知限制等）

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

### 5. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在 PR 描述生成完成后，执行最终验证：

```bash
# 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/pr-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/pr-report.md" \
  "packages/{model-name}/.sisyphus/user_report/pr-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 检查 PR 描述文件是否存在
if [ ! -f "packages/{model-name}/.sisyphus/pr/description.md" ]; then
  echo "WARNING: PR 描述文件缺失"
fi
```

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pr-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的验证脚本
- **输出目录**: `outputs/` - 命令执行日志
- **PR 目录**: `packages/{model-name}/.sisyphus/pr/` - PR 描述文件

### PR 描述

输出到 `packages/{model-name}/.sisyphus/pr/description.md`，用户可自行复制使用。

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/pr-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# PR 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pr |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取 PR 描述生成所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、接入类型、Pipeline 功能规划表
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_pr/` 目录及子目录

**产出物**：执行日志目录

---

### Step 2: 以实际代码为准，校验蓝图信息

**目标**：对比蓝图计划与实际代码，确保 PR 描述基于最新代码状态。

**执行内容**：
- 扫描 `{diffsynth_root}/diffsynth/pipelines/` 下与本次模型相关的文件，逐行读取 `__call__` 签名和 Pipeline 类名
- 扫描 `{diffsynth_root}/diffsynth/models/` 下与本次模型相关的文件
- 扫描 `{diffsynth_root}/examples/` 下与本次模型相关的推理脚本
- 读取 `model_configs.py` 中新增的注册项
- 读取 `pyproject.toml` 的实际内容
- 发现蓝图与实际代码不一致时，以实际代码为准

**产出物**：代码状态确认清单，主要差异记录

**注意事项**：
- 蓝图是计划，代码是事实。不要修正代码去匹配蓝图
- 仅在报告中注明主要差异供用户参考

---

### Step 3: 收集接入信息

**目标**：收集生成 PR 描述所需的所有信息。

**执行内容**：
- 从蓝图报告提取接入类型
- 从实际 Pipeline 文件提取功能列表、类名、参数签名
- 从 `diffsynth/models/` 和 `diffsynth/pipelines/` 提取实际存在的文件
- 从 `examples/` 提取实际存在的推理脚本
- 从 `pyproject.toml` 提取依赖变更
- 从 `model_configs.py` 提取模型信息

**产出物**：信息收集清单

---

### Step 4: 生成 PR 描述

**目标**：根据收集到的信息生成完整的 PR 描述。

**执行内容**：
- 参考 `references/pr-template.md` 模板
- 填入模型名称、接入的功能列表、模型组件清单、新增依赖、注意事项
- 功能列表、组件清单等信息必须以实际代码为准，不要直接复制蓝图
- **PR 描述必须全部使用英文**，不要混入中文
- 输出到 `packages/{model-name}/.sisyphus/pr/description.md`

**产出物**：
- `packages/{model-name}/.sisyphus/pr/description.md`

---

### Step 5: 最终验证

**目标**：确认所有报告文件和输出文件完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查 PR 描述文件是否存在
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## PR 规划

### PR 描述要点
- **PR 描述必须全部使用英文**，包括所有标题、描述、注释
- 模型名称和接入的功能列表（支持哪些 pipeline）
- 模型组件清单（组件名、参数量）
- 新增依赖包
- 注意事项（硬件要求、已知限制等）
- 聚焦新功能介绍，不包含接入过程的测试、验证等细节
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pr-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-5 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 以实际代码为准，校验蓝图信息 | Step 2 |
| 3 | 收集接入信息 | Step 3 |
| 4 | 生成 PR 描述 | Step 4 |
| 5 | 最终验证 | Step 5 |

每完成一个步骤，执行：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pr-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

PR 描述生成完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/pr-report.md << 'OUTER_EOF'
## ✅ PR 描述生成完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 📋 PR 描述

- 文件路径: `packages/{model-name}/.sisyphus/pr/description.md`
- 接入类型: {new_series / version_upgrade}

### ⚠️ 安全提醒

本 skill 仅生成 PR 描述 Markdown 文件，**不执行任何 git 操作**（不创建分支、不提交、不推送）。请用户自行处理代码版本管理并创建 PR，以确保安全。

请 Review PR 描述内容，确认无误后手动执行 git 操作。

### 📦 数据集上传提醒

PR 描述中使用的推理样例数据集（`data/diffsynth_example_dataset/{series}/{FeatureName}/`）**不会包含在代码 PR 中**，需要单独上传到 ModelScope 数据集仓库：

1. **确认数据集目录存在**：`data/diffsynth_example_dataset/{series}/{FeatureName}/`
2. **上传到 ModelScope**：`modelscope upload --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset`
3. **确认推理脚本中的数据集路径**与实际上传路径一致

> 数据集是独立的，不随代码 PR 一起提交。忘记上传数据集会导致推理脚本无法正常获取样例数据。

### 📦 代码推送提醒

本 integration 生成的 DiffSynth-Studio 代码位于 `packages/{model-name}/DiffSynth-Studio/` 目录下，这些代码**不会自动包含在 PR 中**，需要你先推送到自己 fork 的 DiffSynth-Studio 仓库的对应模型分支：

    cd packages/{model-name}/DiffSynth-Studio

    # 1. 先将 origin 重定向到你自己的 fork（如果尚未配置）
    git remote set-url origin https://github.com/{your-username}/DiffSynth-Studio.git

    # 2. 将本地 main 分支推送到 fork 的模型分支（如 ernie-image）
    git push origin main:{model-branch}

    # 3. （可选）添加官方仓库为 upstream，方便后续同步
    git remote add upstream https://github.com/modelscope/DiffSynth-Studio.git

> `{model-branch}` 是当前接好的模型分支名（如 `ernie-image`）。本地代码保持在 main 分支，push 时映射到远程的模型分支即可。模型代码和 Pipeline 文件的引用链接依赖于你的 fork 仓库中该分支上是否存在对应文件。
OUTER_EOF
```
