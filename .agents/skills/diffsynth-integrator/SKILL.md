---
name: diffsynth-integrator
description: >
  Orchestrate the full model integration workflow for DiffSynth-Studio, from analyzing an external
  codebase to generating a PR description. Use this skill when a user wants to integrate a complete
  model (not just one step). This skill coordinates the 11 individual diffsynth-* skills:
  analyze-target, environment, model-code, pipeline, pipeline-lowvram, pipeline-training, style,
  testing, docs, pr, and pr-review. Each step is triggered explicitly by the user — there is no automatic
  state management or step continuation.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke this skill or any diffsynth-* integration skill for other development — bug fixes,
  refactoring, performance tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 全流程编排

编排从目标库分析到 PR 创建的完整接入流程。

## 配置

[config.yaml](config.yaml) 只有一个有效配置项 `env_clone_source`，其余内容是目录结构约定的注释。**路径不从 config.yaml 读取**，全部按下方目录结构规范从 `packages/{model-name}/` 推导。

### 目录结构规范

所有模型集成工作都在 `{workspace_root}/packages/{model-name}/` 目录下进行，结构如下：

```
packages/{model-name}/
├── {target-library}/          # 目标模型库
│   └── models/                # 真实目录
│       └── model-owner/
│           └── Model-Name/ → ~/.cache/modelscope/hub/models/model-owner/Model-Name
├── DiffSynth-Studio/         # DiffSynth 框架
│   └── models/                # 真实目录
│       └── model-owner/
│           └── Model-Name/ → ~/.cache/modelscope/hub/models/model-owner/Model-Name
└── .sisyphus/                # 中间报告和日志
    ├── plans/                # Plan 文件（执行前制定）
    ├── skill_work_report/    # 渐进式步骤报告（与 workflow step 一一对应）
    ├── user_report/          # 向用户报告（必须写入文件）
    ├── execution-logs/       # 执行日志（scripts/, outputs/, checkpoints/）
    ├── tests/                # 测试结果 + 输出审查
    ├── integration-blueprints/  # 蓝图报告 + 可行性分析
    ├── code-reports/         # 代码阅读报告
    ├── reports/              # Pipeline 设计蓝图
    ├── pr/                   # PR 描述（description.md）
    └── pr-review/            # PR Review 反馈
```

**重要规则：**
- DiffSynth-Studio 必须 clone 到 `packages/{model-name}/DiffSynth-Studio/`，与目标库在同一目录下
- 不允许在 `packages/` 根目录下直接放置 DiffSynth-Studio，这会破坏多模型隔离

### 配置项

来自 [config.yaml](config.yaml)：

- `env_clone_source` — 用于 clone 的源环境名称（如 "diffsynth"）

以下不是配置项，而是按目录结构约定推导的占位符：

- `{diffsynth_root}` = `packages/{model-name}/DiffSynth-Studio/`
- `{target_path}` = `packages/{model-name}/{target-library}/`（具体名称从蓝图报告获取）
- `{sisyphus_root}` = `packages/{model-name}/.sisyphus/`

**模型软链接**：`models/` 是真实目录，内部每个模型 repo 单独软链接到 `~/.cache/modelscope/hub/models/`。目标库和 DiffSynth 各自拥有独立的 `models/` 目录，但软链接指向同一个 ModelScope 缓存路径。

**目标库路径不由 config.yaml 管理。** 当用户提到接入某个模型时（如"接入 ACE-Step-1.5"），agent 应自动在 `{workspace_root}/packages/` 下查找对应目录（如 `packages/ACE-Step-1.5/`）。如果找到多个匹配，向用户确认。

**环境名称由 skill 自动生成。** 格式为 `{model_name}-diffsynth`（如 `ace-step-diffsynth`），不写入 config.yaml。

## 流程概览

| Step | Skill | 说明 | 人工确认点 |
|------|-------|------|-----------|
| 0 | `diffsynth-analyze-target` | 分析目标库，生成接入蓝图 + 代码阅读报告 | ✅ 确认蓝图 |
| 1 | `diffsynth-environment` | 双向环境验证：DiffSynth OK → Target OK → DiffSynth still OK | — |
| 2 | `diffsynth-model-code` | 模型代码接入 + 一致性测试 | — |
| 3 | `diffsynth-pipeline` | Pipeline 接入（每次调用处理一个功能） | — |
| 3a | `diffsynth-pipeline-lowvram` | 低显存推理脚本 + VRAM 管理配置 | — |
| 3b | `diffsynth-pipeline-training` | 训练模块接入 + 训练验证 | — |
| 4 | `diffsynth-style` | 代码风格化：统一 pipeline/模型/脚本的代码风格 | — |
| 5 | `diffsynth-testing` | 全量推理脚本测试 + 结果保存 | ✅ 确认输出质量 |
| 6 | `diffsynth-docs` | 中英文文档生成 + README 更新 | — |
| 7 | `diffsynth-pr` | 生成 PR 描述 Markdown 文件（不执行 git 操作） | ✅ Review PR 描述 |
| 8 | `diffsynth-pr-review` | 处理 PR Review 反馈，按评论针对性修复 | — |

## 执行方式

每个 step 由用户显式启动，不自动推进。但任何 step 执行前，**必须先确保 config.yaml 已填充**。

### 初始化规则

当用户首次提到接入某个模型时（如"接入 ACE-Step-1.5"），agent 应：
1. 从用户输入中提取模型名称
2. 在 `{workspace_root}/packages/` 下查找对应目录（如 `packages/ACE-Step-1.5/`）
3. 如果找到多个匹配，向用户确认
4. 将 `target_path` 记录到上下文（不写入 config.yaml），后续步骤从蓝图报告中获取

这样后续所有 step 都从蓝图报告中获取目标库路径，config.yaml 只管理全局环境配置。

### 典型流程

```
用户: "接入 Ernie-Image"
  → agent 在 packages/ 下找到 packages/Ernie-Image/
  → 自动 clone DiffSynth-Studio（如不存在）
  → 执行 diffsynth-analyze-target
  → 生成蓝图报告 → 等待用户确认

用户: "蓝图没问题，执行环境创建"
  → 执行 diffsynth-environment
  → 从 config.yaml 读取 clone 源，从蓝图读取环境名称和依赖差异
  → 双向验证通过

用户: "现在接入模型代码"
  → 执行 diffsynth-model-code
  → 从蓝图读取组件清单和接入类型
  → 一致性测试通过

用户: "接入 Text-to-Image Pipeline"
  → 执行 diffsynth-pipeline（首次：创建 Pipeline 文件 + 推理脚本）
  → E2E 验证通过

用户: "接入 Image-to-Image Pipeline"
  → 再次执行 diffsynth-pipeline（后续：追加 Unit 和脚本）
  → E2E 验证通过

用户: "接入低显存推理"
  → 执行 diffsynth-pipeline-lowvram
  → 注册 module_map + 生成低显存脚本

用户: "接入训练模块"（可选）
  → 执行 diffsynth-pipeline-training
  → 创建训练模块 + 训练验证通过

用户: "代码风格化"
  → 执行 diffsynth-style
  → 统一代码风格

用户: "运行推理脚本测试"
  → 执行 diffsynth-testing
  → 全量测试 + 结果保存 → 等待用户确认输出质量

用户: "生成文档"
  → 执行 diffsynth-docs
  → 生成中英文文档 + 更新 README

用户: "生成 PR 描述"
  → 执行 diffsynth-pr
  → 以实际代码为准生成 PR 描述 → 等待用户 Review

用户: "处理 PR Review 反馈"
  → 执行 diffsynth-pr-review
  → 分析评论 → 生成修改蓝图 → 针对性修复
```

### 上下文传递机制

| 信息类型 | 存储位置 | 读取方 |
|---------|---------|--------|
| 环境配置 | `config.yaml` | diffsynth-environment |
| 目标库路径 | 蓝图报告 + 对话上下文 | 所有 skill |
| 蓝图报告 | `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md` | Step 1-7 |
| 代码阅读报告 | `packages/{model-name}/.sisyphus/code-reports/code-report.md` | 供人阅读 |
| 可行性分析 | `packages/{model-name}/.sisyphus/integration-blueprints/feasibility-analysis.md` | 供人阅读 |
| 测试结果 | `packages/{model-name}/.sisyphus/tests/latest/test_report.md` | 人工 Review |
| PR 描述 | `packages/{model-name}/.sisyphus/pr/description.md` | 人工 Review |
| 模型路径 | `~/.cache/modelscope/hub/models/` | Step 1-5 |
| **Conda 环境名称** | 蓝图报告「基本信息」表格 | Step 1-5（**单一数据源**，不自行生成） |
| 共享参考文档 | `diffsynth-integrator/references/` | 所有 skill |

## 关键规则

1. **蓝图驱动**：Step 1-5 都参考 Step 0 生成的蓝图，不猜测
2. **按需执行**：每个 step 内部会判断是否需要执行，不做不必要的事
3. **用户控制**：每个 step 由用户显式启动，不自动推进到下一步
4. **双向验证**：Step 1 环境创建遵循 "DiffSynth OK → Target OK → DiffSynth still OK" 验证序列
5. **人工确认点**：Step 0（蓝图）、Step 5（测试结果）、Step 7（PR 描述）必须用户确认
6. **失败回滚**：任何 step 失败时，停止流程，报告错误，等待用户决策

## 共享参考文档

所有 diffsynth-* skill 共享以下参考文档（位于 `diffsynth-integrator/references/`）：

| 文档 | 用途 | 依赖方 |
|------|------|--------|
| [pipeline-template.md](../diffsynth-pipeline/references/pipeline-template.md) | Pipeline 架构唯一定义 | analyze-target, model-code, pipeline, docs, testing |
| [exec-log-init.md](references/exec-log-init.md) | 执行日志初始化模板 | environment, model-code, pipeline, testing, pr |
| [blueprint-contract.md](references/blueprint-contract.md) | Blueprint 契约规范 | 所有 skill |
| [execution-traceability.md](references/execution-traceability.md) | 执行留痕规范 | 所有 skill |
| [step-report.md](references/step-report.md) | 渐进式步骤报告格式 | 所有 skill |

**修改共享文档时**，需同步更新所有依赖方的引用。

## 各 Skill 详细工作流程

所有 11 个 diffsynth-* skill 的详细工作流程（具体到三级子步骤）已整理到 [skill-workflows.md](skill-workflows.md)。该文档包含每个 skill 的完整步骤拆解、操作细节和输出文件清单，可作为执行参考。

## 完整框架文档

- Pipeline 架构定义：[pipeline-template.md](../diffsynth-pipeline/references/pipeline-template.md)
- Blueprint 契约规范：[blueprint-contract.md](references/blueprint-contract.md)
- 执行日志初始化：[exec-log-init.md](references/exec-log-init.md)
- 渐进式步骤报告：[step-report.md](references/step-report.md)
- 执行留痕规范：[execution-traceability.md](references/execution-traceability.md)
