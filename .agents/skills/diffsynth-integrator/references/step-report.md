# 渐进式步骤报告规范

> **共享规范**：每个 diffsynth-* skill 在执行过程中，每完成一个小步骤（sub-task），就追加一条记录到自己的步骤报告文件中。用户可随时查看了解当前进度。

## 报告位置

每个 skill 有独立的报告文件，统一输出到 `skill_work_report/` 目录：

```
packages/{model-name}/.sisyphus/skill_work_report/
├── analyze-target-report.md       # diffsynth-analyze-target 的步骤报告
├── environment-report.md          # diffsynth-environment 的步骤报告
├── model-code-report.md           # diffsynth-model-code 的步骤报告
├── pipeline-{feature}-report.md   # diffsynth-pipeline 的步骤报告（每个功能一个）
├── lowvram-report.md              # diffsynth-pipeline-lowvram 的步骤报告
├── pipeline-training-report.md    # diffsynth-pipeline-training 的步骤报告
├── style-report.md                # diffsynth-style 的步骤报告
├── testing-report.md              # diffsynth-testing 的步骤报告
├── docs-report.md                 # diffsynth-docs 的步骤报告
├── pr-report.md                   # diffsynth-pr 的步骤报告
└── pr-review-report.md            # diffsynth-pr-review 的步骤报告
```

## 报告结构

```markdown
# {Skill Name} 执行报告

> 自动生成，记录本 skill 执行过程中的每个步骤。

## 基本信息

| 字段 | 值 |
|------|-----|
| 模型名称 | {name} |
| Skill | {skill_name} |
| 开始时间 | {timestamp} |
| 状态 | 进行中 / 已完成 / 失败 |

## 步骤对照表

| Step | 名称 | 对应 Workflow | 状态 |
|------|------|---------------|------|
| 1 | {步骤名称} | Step {N} | ✅ / ⬜ |
| 2 | {步骤名称} | Step {N} | ✅ / ⬜ |
| ... | ... | ... | ... |

---

## 执行时间线

### Step 1: {步骤名称}
- **状态**: ✅ 完成
- **完成时间**: {timestamp}
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: `{文件路径}`

### Step 2: {步骤名称}
- **状态**: ✅ 完成
- **完成时间**: {timestamp}
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: `{文件路径}`

...

## 执行摘要

> 所有步骤完成后生成。

- **总步骤数**: {N}
- **成功**: {N}
- **失败**: {N}
- **跳过**: {N}
- **总耗时**: {duration}
```

## 生成规则

### 首次创建（skill 开始时）

每个 skill 执行的第一步，创建报告文件：

```bash
mkdir -p packages/{model-name}/.sisyphus/skill_work_report

cat > packages/{model-name}/.sisyphus/skill_work_report/{skill-report-filename}.md << EOF
# {Skill Name} 执行报告

> 自动生成，记录本 skill 执行过程中的每个步骤。

## 基本信息

| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | {skill_name} |
| 开始时间 | \$(date -Iseconds) |
| 状态 | 进行中 |

## 步骤对照表

| Step | 名称 | 对应 Workflow | 状态 |
|------|------|---------------|------|
| 1 | {步骤1名称} | Step {N} | ⬜ |
| 2 | {步骤2名称} | Step {N} | ⬜ |
| ... | ... | ... | ... |

---

## 执行时间线
EOF
```

### 每完成一个步骤后追加

**关键规则**：每完成一个有意义的子步骤，立即追加记录。不要等到 skill 全部完成才写。

```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/{skill-report-filename}.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败 / ⬜ 跳过
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述做了什么}
- **关键结果**: {1-2 句话说明结果，如"识别出 3 个模型组件：DiT、VAE、TextEncoder"}
- **输出文件**: \`{文件路径}\`
EOF
```

同时更新步骤对照表中的状态：
```bash
sed -i 's/| {N} | {步骤名称} | ... | ⬜ |/| {N} | {步骤名称} | ... | ✅ |/' \
  packages/{model-name}/.sisyphus/skill_work_report/{skill-report-filename}.md
```

### 全部完成后追加摘要

```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/{skill-report-filename}.md << EOF

---

## 执行摘要

- **总步骤数**: {N}
- **成功**: {N}
- **失败**: {N}
- **跳过**: {N}
- **总耗时**: {duration}
EOF

# 更新状态为已完成
sed -i 's/| 状态 | 进行中 |/| 状态 | 已完成 |/' packages/{model-name}/.sisyphus/skill_work_report/{skill-report-filename}.md
```

## 各 Skill 的步骤划分

### diffsynth-analyze-target

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 读取下游 Skill 全文 | 读取 6 个下游 skill 的 SKILL.md |
| 2 | 分析目标库概览 | 读取项目结构、依赖、README |
| 3 | 识别模型组件 | 找到所有模型定义文件，提取类名、签名 |
| 4 | 模型耦合分析 | 分析训练/推理耦合、条件/DiT 耦合等 |
| 5 | 外部依赖组件分析 | 判断哪些是 diffusers/transformers/自有源码 |
| 6 | 功能规划 | 识别所有功能，按难度排序 |
| 7 | 依赖对比 | 对比目标库与 DiffSynth 的依赖差异 |
| 8 | 生成蓝图报告 | 输出 blueprint.md |
| 9 | 生成代码阅读报告 | 输出 code-report.md |

### diffsynth-environment

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 创建 conda 环境 | clone 源环境 |
| 2 | 创建模型软链接 | 建立统一模型路径结构 |
| 3 | 安装 DiffSynth 并验证 | pip install + 运行验证脚本 |
| 4 | 分析目标库依赖 | 对比依赖差异 |
| 5 | 逐步安装目标库依赖 | 分批安装，每批验证 |
| 6 | 模型预检查 | 检查所需模型是否已下载 |
| 7 | 验证目标库能运行 | 运行目标库验证脚本 |
| 8 | 验证 DiffSynth 仍然正常 | 最终双向验证 |

### diffsynth-model-code

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 + manifest.json |
| 2 | 确定接入方式 | new_series / version_upgrade，为每个组件制定接入方案 |
| 3 | 执行接入方案 | 新建/修改/复用模型文件（每个组件一个子步骤） |
| 4 | 编写 state_dict_converter | 如需要 |
| 5 | 注册到 model_configs.py | 计算 hash，添加注册条目 |
| 6 | ModelPool 加载测试 | 前置验证注册链正确性（所有情况都必须执行） |
| 7 | 一致性验证 | Forward 一致性对比（所有情况都必须执行） |

### diffsynth-pipeline

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 + manifest.json |
| 2 | 确定本次接入的功能 | 从蓝图读取功能信息 |
| 3 | 创建/更新 Pipeline 文件 | 首次创建或追加 Unit |
| 4 | 创建推理示例脚本 | model_inference/{feature}.py |
| 5 | 创建训练模块 | 如首次且支持训练 |
| 6 | 一致性验证 | 原库 vs DiffSynth 输出对比 |
| 7 | 验证完成 | Code Review 或逐步测试定位问题 |

### diffsynth-pipeline-training

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 制定执行计划 | 训练模块架构、数据集方案、训练脚本清单等 |
| 3 | 分析训练需求 | 确定 SeriesName、series、ModelName、任务类型 |
| 4 | 准备数据集 | 下载或复用已有样例数据集 |
| 5 | 创建训练代码 | TrainingModule、train.py、.sh 脚本、验证脚本 |
| 6 | Debug 测试 | 降参快速验证 |
| 7 | 训练验证 | 完整运行正式训练 |

### diffsynth-testing

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 收集测试范围 | 扫描推理脚本、验证脚本 |
| 2 | 运行推理脚本 | 运行所有 model_inference/*.py |
| 3 | 运行低显存推理脚本 | 运行所有 model_inference_low_vram/*.py |
| 4 | 运行训练验证脚本 | 运行 validate_full/validate_lora |
| 5 | 收集一致性测试结果 | 从之前 skill 收集 |
| 6 | 生成测试报告 | 输出 test_report.md |

### diffsynth-pipeline-lowvram

| Step | 名称 | 说明 |
|------|------|------|
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 制定执行计划 | 低显存脚本架构、module_map 注册规划等 |
| 3 | 注册 module_map | 在 vram_management_module_maps.py 中注册映射 |
| 4 | 生成低显存推理脚本 | 为每个功能生成 model_inference_low_vram/ 脚本 |
| 5 | 验证正确性 | 静态检查 + 动态运行 |

### diffsynth-style

| Step | 名称 | 说明 |
|------|------|------|
| 0 | 读取蓝图信息 | 从蓝图获取 conda 环境名、组件清单、接入类型 |
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 制定执行计划 | 风格化范围与规则规划 |
| 3 | 确定需要风格化的文件 | 根据用户指定或自动查找 |
| 4 | 自动备份并确认 | 备份 + 逐条确认将应用的规则 |
| 5 | 逐文件应用风格规则 | 纯格式修改，不改逻辑 |
| 6 | 验证 | import 正常、测试不变 |
| 7 | 更新蓝图 — 补充测试脚本清单 | 供 diffsynth-testing 消费 |
| 8 | 最终验证 | 确认所有报告文件完整性 |

### diffsynth-docs

| Step | 名称 | 说明 |
|------|------|------|
| 0 | 读取蓝图信息 | 从蓝图获取 conda 环境名、系列名等 |
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 制定执行计划 | 文档生成规划 |
| 3 | 判断接入类型 | new_series / version_upgrade |
| 4 | 读取参考文档 | 读取 2-3 篇现有文档提取风格 |
| 5 | 从代码提取信息 | 读取 Pipeline、推理脚本、训练脚本等 |
| 6 | 向用户收集缺失信息 | 一句话模型描述等 |
| 7 | 生成/更新文档 | 按固定模板生成中英文文档 |
| 8 | 更新 docs/ 内的注册点 | index.rst、docs/*/README.md、按需 Setup.md |
| 9 | 更新仓库根 README | 更新历史、全局模型表、基础框架分类列表 |
| 10 | 静态验证 | 检查链接、表格、参数一致性 |
| 11 | 最终验证 | 确认所有报告文件完整性 |

### diffsynth-pr

| Step | 名称 | 说明 |
|------|------|------|
| 0 | 读取蓝图信息 | 从蓝图获取接入类型等 |
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 以实际代码为准，校验蓝图信息 | 扫描 Pipeline/模型/converter/脚本/配置/文档实际文件 |
| 3 | 收集接入信息 | 从实际代码提取 PR 描述所需信息 |
| 4 | 生成 PR 描述 | 按模板填充，输出到 pr/description.md |
| 5 | 最终验证 | 确认所有报告文件完整性 |

### diffsynth-pr-review

| Step | 名称 | 说明 |
|------|------|------|
| 0 | 读取蓝图信息 | 从蓝图获取 conda 环境名等 |
| 1 | 初始化执行日志目录 | 创建执行日志目录结构 |
| 2 | 制定执行计划 | 反馈处理规划 |
| 3 | 获取 PR 反馈、分析反馈与生成修改蓝图 | 抓取 Review 评论并逐条分析 |
| 4 | 用户确认 | 展示修改蓝图，等待确认 |
| 5 | 执行修改 | 按确认后的蓝图做针对性修改 |
| 6 | 执行最小测试（如需要） | 推理 smoke test + LoRA 训练最小测试 |
| 7 | 验证修改 | 确认修改覆盖全部反馈，报告文件完整 |

## 报告内容填写原则

1. **面向用户**：使用自然语言，避免技术细节堆砌
2. **即时更新**：每完成一个步骤立即追加，不要等到最后
3. **结果导向**：重点说明"完成了什么"和"结果如何"
4. **可追溯**：每个结论都指向具体的输出文件
5. **简洁**：每个步骤 3-5 行，不要长篇大论

---

*本文档是共享参考，修改时需同步更新所有依赖此文档的 skill。*
