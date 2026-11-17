# Blueprint 契约规范

> **共享契约**：本文档定义了 Blueprint 报告的统一结构，是所有 diffsynth-* skill 的数据源契约。

## 概述

Blueprint 报告由 `diffsynth-analyze-target` skill 生成，输出到 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`。

**完整格式定义**见 [blueprint-format.md](../../diffsynth-analyze-target/references/blueprint-format.md)。

本文档聚焦于**各 skill 与 Blueprint 的契约关系**——每个 skill 需要什么、从哪里获取。

## 单一数据源原则

| 信息类型 | 存储位置 | 读取方 | 生成方 |
|---------|---------|--------|--------|
| 目标库路径 | 蓝图报告「基本信息」表格 | 所有 skill | diffsynth-analyze-target |
| Conda 环境名称 | 蓝图报告「基本信息」表格 | environment, model-code, pipeline, pipeline-lowvram, pipeline-training, style, testing, docs, pr-review | diffsynth-analyze-target（自动生成） |
| 接入类型 | 蓝图报告「基本信息」表格 | model-code, style, docs, pr | diffsynth-analyze-target |
| 模型组件清单 | 蓝图报告第 2 节 | model-code | diffsynth-analyze-target |
| 依赖差异分析 | 蓝图报告第 1 节 | environment | diffsynth-analyze-target |
| Pipeline 功能规划 | 蓝图报告第 3 节 | pipeline | diffsynth-analyze-target |
| 模型下载命令清单 | 蓝图报告第 6 节 | environment | diffsynth-analyze-target |
| 功能-脚本映射 | 蓝图报告第 4.1 节 | environment, testing | diffsynth-analyze-target |
| 低显存推理信息 | 蓝图报告第 4.2 节 | pipeline-lowvram | diffsynth-analyze-target |
| 训练信息 | 蓝图报告第 4.3 节 | pipeline-training | diffsynth-analyze-target |

## Conda 环境名称责任链

**生成规则**：`{model_name}-diffsynth`（蛇形命名）
- 示例：`"ACE-Step-1.5"` → `ace-step-diffsynth`

**责任链**：
1. **生成**：`diffsynth-analyze-target` 在 Step 0 自动生成，写入蓝图报告「基本信息」表格
2. **存储**：蓝图报告是唯一存储位置，不写入 config.yaml
3. **读取**：后续所有 skill 从蓝图报告读取，**不自行生成**

**为什么不由 config.yaml 管理**：环境名称与具体模型绑定，属于项目级配置而非全局配置。

## 各 Skill 蓝图需求速查

### diffsynth-environment 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 依赖差异分析 | 第 1 节 | 确定需要安装哪些包 |
| DiffSynth 验证脚本 | 第 1.4 节 | 环境创建前后验证 DiffSynth |
| 目标库验证脚本 | 第 1.4 节 | 验证目标库在共享环境中能否运行 |
| 模型下载命令清单 | 第 6 节 | 模型预检查和软链接创建 |

### diffsynth-model-code 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 接入类型 | 基本信息 | 决定新建文件还是只注册 hash |
| 模型组件清单 | 第 2 节（含 2.0/2.0.1/2.1/2.2） | 每个组件的类名、签名、权重 key 模式 |
| 外部依赖组件分析 | 第 2.0.1 节 | 判断哪些组件需要重构/包装 |
| 模型耦合分析 | 第 2.0 节 | 拆分建议 |
| 一致性测试规划 | 第 2.2.1 节 | 确定哪些测试必须执行 |

### diffsynth-pipeline 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 功能名称 | 第 3 节 | 本次要接入的功能 |
| 原库推理脚本路径 | 第 3 节 | E2E 测试对比 |
| 输入输出、特殊处理 | 第 3 节 | 设计 Unit 和 `__call__` 参数 |
| 依赖的模型组件 | 第 3 节 | `from_pretrained` 中的 model_configs |

### diffsynth-docs 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 接入类型 | 基本信息 | 新建文档还是更新已有文档 |
| 功能列表 | 第 3 节 | 确定推理脚本表格行数 |
| Pipeline 类名 | 基本信息 | 定位 Pipeline 文件 |
| 与已有版本的差异 | 第 4 节（基本信息表格） | version_upgrade 时填写 |

### diffsynth-testing 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 推理脚本路径 | 第 4.1 节 | 确定测试范围 |
| 低显存推理脚本路径 | 第 4.1 节 | 可选测试 |
| 训练验证脚本路径 | 第 4.1 节 | 可选测试 |

### diffsynth-pipeline-lowvram 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| Pipeline 类名 | 基本信息 | 确认 Pipeline 结构 |
| 功能列表 | 第 3 节 | 确定需要生成哪些低显存推理脚本 |
| 模型组件列表 | 第 2 节 | 确定需要为哪些组件注册 module_map |
| 低显存信息 | 第 4.2 节 | 组件显存占用分析、优化建议 |

### diffsynth-pipeline-training 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 训练支持 | 第 3 节（功能表） | 判断是否生成训练模块 |
| 训练信息 | 第 4.3 节 | 训练类型、数据集、参数、InputEmbedder 需求 |
| Scheduler 训练策略 | 第 3.1 节 | 训练时 timestep 策略 |

### diffsynth-style 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| Conda 环境名称 | 基本信息 | 验证风格化后的文件仍能正常 import |
| 模型组件清单 | 第 2 节 | 确定需要风格化的模型文件范围 |
| 接入类型 | 基本信息 | 区分 new_series / version_upgrade 的处理范围 |

### diffsynth-pr 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| 接入类型 | 基本信息 | PR 描述中的集成类型说明 |
| Pipeline 功能规划 | 第 3 节 | 接入的功能列表（最终以实际代码为准） |

> 注意：PR 描述生成时必须以实际代码为准，蓝图信息仅作参考。蓝图中的功能名、组件名、Pipeline 类名等可能与实际代码不同，一律以实际代码为准。

### diffsynth-pr-review 需要

| 字段 | 蓝图章节 | 用途 |
|------|---------|------|
| Conda 环境名称 | 基本信息 | 执行最小测试时运行 Python / 训练脚本 |

除环境名外**不依赖蓝图**：PR Review 标识由用户提供（PR URL 或 PR 号），修改基于本地磁盘代码，修改蓝图由本 skill 自行生成（`pr-review/analysis-modification-blueprint.md`）。

## Blueprint 格式变更流程

当需要修改 Blueprint 结构时：

1. 更新 `diffsynth-analyze-target/references/blueprint-format.md`
2. 更新本文档中的对应章节
3. 通知所有受影响的 skill 更新"蓝图需求"部分
4. 确保向后兼容（旧蓝图仍能工作）

---

*本文档是共享参考，修改时需同步更新所有依赖此文档的 skill。*
