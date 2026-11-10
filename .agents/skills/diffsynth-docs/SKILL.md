---
name: diffsynth-docs
description: >
  Generate or update documentation for a model being integrated into DiffSynth-Studio. Use this
  skill when documentation needs to be created or updated as part of an integration — whether for a
  completely new model series or just adding a new variant/feature to an existing series. Generates
  both Chinese (zh) and English (en) documentation, and updates the corresponding index.rst files.
  Information comes from multiple sources: already integrated model/pipeline code, user
  specification, or blueprint report.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — general doc edits, bug fixes, refactoring, performance
  tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 文档生成

根据已接入的模型代码生成文档。文档使用**固定模板结构**，功能驱动体现在：根据实际接入的功能，在模板的对应章节填入内容。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

## 固定模板结构

**所有模板结构（中英文完整文档模板、固定内容、章节映射、关键规则）已移至 [references/doc-templates.md](references/doc-templates.md)。**

生成文档前，请先读取该参考文件。

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
| 基本信息表中的 `Conda 环境名称` | 验证环境 |
| 基本信息表中的 `目标库路径` | 文档参考 |
| 接入类型 | new_series / version_upgrade，决定文档生成策略 |
| Pipeline 功能规划表 | Pipeline 类名、系列名称、功能列表 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_docs"
mkdir -p ${EXEC_LOG_DIR}/{outputs,scripts,checkpoints}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始生成文档前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/docs-plan.md`。** 基于蓝图信息和代码状态，明确接入类型、参考文档、信息提取清单、用户收集项和文档输出路径。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始生成文档。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 3. 判断接入类型

> 📖 开始前：重读本步骤描述，确认流程与报告路径

从蓝图报告中读取 `接入类型`（new_series / version_upgrade）。无蓝图时向用户确认。

**蓝图路径**: `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 4. 读取参考文档

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**首先读取文档模板**：`references/doc-templates.md`，其中包含中英文完整文档结构、固定内容模板、章节名称映射、关键规则。这是生成文档的主要依据。

**new_series**：在理解模板结构的基础上，读取 `docs/zh/Model_Details/` 下 2-3 篇参考文档，提取章节组织风格、代码示例风格、参数说明风格。同时读取 `docs/en/Model_Details/` 下对应英文参考文档，提取英文描述风格。如对模板中某章节的写法不清楚，可多读几篇参考文档确认。

**version_upgrade**：读取现有的 `docs/zh/Model_Details/{Model-Name}.md` 和 `docs/en/Model_Details/{Model-Name}.md`，了解已有结构。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 5. 从代码提取信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**必须读取的文件**：

| 文件路径 | 提取什么 | 填入模板哪个章节 |
|---------|---------|----------------|
| `{diffsynth_root}/diffsynth/pipelines/{model}_*.py` | Pipeline 类名、`__call__` 参数签名 | 快速开始、模型推理参数 |
| `{diffsynth_root}/examples/{model}/model_inference/*.py` | 推理脚本列表（每个脚本 = 表格一行）、完整代码 | 快速开始（选最基础的）、模型总览表格行 + 链接 |
| `{diffsynth_root}/examples/{model}/model_inference_low_vram/*.py` | 低显存推理脚本列表 | 模型总览表格链接 |
| `{diffsynth_root}/examples/{model}/model_training/full/*.sh` | 全量训练脚本列表 | 模型总览表格链接 |
| `{diffsynth_root}/examples/{model}/model_training/validate_full/*.py` | 验证脚本列表 | 模型总览表格链接 |
| `{diffsynth_root}/examples/{model}/model_training/lora/*.sh` | LoRA 训练脚本列表 | 模型总览表格链接 |
| `{diffsynth_root}/examples/{model}/model_training/validate_lora/*.py` | LoRA 验证脚本列表 | 模型总览表格链接 |
| `{diffsynth_root}/examples/{model}/model_training/train.py` | 训练专有参数（argparse 中非通用参数部分） | 模型训练参数 → 专有参数 |
| `{diffsynth_root}/examples/{model}/model_training/special/` | 特殊训练脚本列表 | 特殊训练脚本章节 |
| `{diffsynth_root}/diffsynth/configs/model_configs.py` | 模型 ID（补充到表格的"模型 ID"列） | 模型总览表格的模型 ID 列 |

**注意**：通用训练参数使用固定模板，不需要从 `train.py` 提取。只需提取 `{Model-Name} 专有参数`。

**提取规则**：
- Pipeline 参数说明优先从代码 docstring 提取
- docstring 缺失时，从参数名和类型推断
- 无法推断时，标注 `TODO: 待补充说明`
- 专有参数说明参考 `train.py` 中 argparse 的 `help` 参数

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 6. 向用户收集缺失信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

以下信息必须用户提供：
- 一句话模型描述

以下信息如代码中无法提取，向用户询问：
- 特殊功能的业务含义
- DeepSpeed ZeRO 3 训练说明（如支持）
- 训练技巧（如有）

**不阻塞生成**：用户暂不提供的字段留空或标注 TODO。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 7. 生成/更新文档

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**开始前强制要求**：必须先完整读取 `references/doc-templates.md`，确认中文/英文文档结构、固定内容模板、章节名称映射、关键规则后再动笔。不要凭记忆或猜测生成。

**new_series**：
1. 按固定模板结构构建中文文档
2. 安装、通用训练参数、样例数据集命令使用固定模板
3. 快速开始代码从最基础的推理脚本完整复制（含 VRAM 配置、from_pretrained、调用、保存）
4. 模型总览表格每行对应 `model_inference/` 下的一个推理脚本，模型 ID 从 `model_configs.py` 补充
5. 推理参数从 `Pipeline.__call__` 签名逐行提取
6. 训练参数：通用参数用固定模板 + 专有参数从代码提取
7. 输出中文文档到 `{diffsynth_root}/docs/zh/Model_Details/{Model-Name}.md`
8. **同步生成英文文档**：章节标题和描述使用英文（见 `references/doc-templates.md` 中的章节名称映射表），代码部分保持不变，输出到 `{diffsynth_root}/docs/en/Model_Details/{Model-Name}.md`

**version_upgrade**：
1. 读取现有中文文档 `docs/zh/Model_Details/{Model-Name}.md` 和英文文档 `docs/en/Model_Details/{Model-Name}.md`
2. 在模型总览表格中追加新模型行
3. 保持已有结构和内容不变

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 8. 更新 docs/ 内的注册点

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**new_series** 需要注册到 8.1 和 8.2 两处，缺一处新模型就不会出现在文档导航里；8.3 按需触发。

#### 8.1 index.rst

在 `docs/zh/index.rst` 和 `docs/en/index.rst` 的"模型详解"(`Model_Details`) toctree 区块末尾追加新文档条目：

```rst
# docs/zh/index.rst — 在"模型详解"区块最后追加
.. toctree::
   :maxdepth: 2
   :caption: 模型详解

   Model_Details/FLUX
   Model_Details/Wan
   ...
   Model_Details/{Model-Name}   ← 新增
```

```rst
# docs/en/index.rst — 在"Model Details"区块最后追加
.. toctree::
   :maxdepth: 2
   :caption: Model Details

   Model_Details/FLUX
   Model_Details/Wan
   ...
   Model_Details/{Model-Name}   ← 新增
```

**version_upgrade**：文档条目已存在，不需要修改 `index.rst`。

#### 8.2 文档主页

在 `docs/zh/README.md` 的 `## Section 2: 模型详解` 和 `docs/en/README.md` 的 `## Section 2: Model Details` 列表末尾各追加一行。两边链接完全相同，只有章节标题不同：

```markdown
* [{Model-Series-Name}](./Model_Details/{Model-Name}.md)
```

列表按接入时间顺序排列，**追加到末尾**，不要按字母序插入。

**version_upgrade**：条目已存在，不需要修改文档主页。

#### 8.3 可选依赖说明（条件触发）

`docs/zh/Pipeline_Usage/Setup.md` 和 `docs/en/Pipeline_Usage/Setup.md` 列出了 `pip install` 的可选依赖组。**仅当新模型需要某个可选依赖时**才改这里：

- **复用已有依赖组**：把模型名加入该组的举例。例如音频模型需要 `[audio]`，上游写法是「`[audio]`: 用于音频模型的支持，例如 ACE-Step、MiniMax-Music3 等」，新音频模型追加到这个举例列表即可
- **需要专属依赖组**：在「特定模型的依赖」/「Model-specific dependencies」子项下新增一条，格式参考已有的 `[infiniteyou]`、`[ses]`、`[nexusgen]`

依赖组本身在 `pyproject.toml` 的 `[project.optional-dependencies]` 中定义，**不属于本 skill 范围**——若发现需要新增依赖组而 `pyproject.toml` 中还没有，向用户报告，由 `diffsynth-environment` 或 `diffsynth-model-code` 处理。

模型不需要任何可选依赖时（纯图像/视频模型的常见情况），跳过本小节。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 9. 更新仓库根 README

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**对象是仓库根目录的 `README.md` / `README_zh.md`**，与 Step 8.2 的文档主页（`docs/*/README.md`）是不同文件，两者都要更新。

**详细步骤见 [references/doc-templates.md](references/doc-templates.md) 中的 "README 更新模板" 章节。**

需要改两处：更新历史（9.1）和全局模型表（9.2）。

#### 9.1 更新历史 — new_series 和 version_upgrade 都需要

1. 在 `README.md` 的 `## Update History` 章节中，在最新一条记录上方插入新条目
2. 在 `README_zh.md` 的 `## 更新历史` 章节中，在最新一条记录上方插入新条目
3. 格式参考已有条目：`- **{Date}** {简短描述} + [文档链接](/docs/zh/Model_Details/{Model-Series-Name}.md) + [代码链接](/examples/{model}/)`
4. version_upgrade 时，描述应体现是"新增 XX 功能/变体"而非"开源"
5. **new_series 模板**：
   - 英文：`- **{Month Day, Year}** {Model-Series-Name} open-sourced, welcome a new member to the {image/video/audio} model family! ...`
   - 中文：`- **{Year}年{Month}月{Day}日** {Model-Series-Name} 开源，欢迎加入{图像/视频/音频}生成模型家族！...`

**模型类型判断**（用于更新历史措辞，以及 9.3 的分类归属，不影响表格位置）：
- **图像模型**：Pipeline 文件名含 `image`（如 `ernie_image`、`flux`、`qwen_image`、`z_image`、`joyai_image`），或功能为 text-to-image、image editing、image-to-image 等
- **视频模型**：Pipeline 文件名含 `video`（如 `wan_video`、`ltx2_audio_video`、`step_video`），或功能为 text-to-video、video editing、image-to-video 等
- **音频模型**：Pipeline 文件名含 `audio` / `music`（如 `ace_step`、`minimax_music3`），或功能为 text-to-music、text-to-audio 等
- **不确定时**：从蓝图报告中的 Pipeline 功能规划表判断，或向用户确认

> 音视频联合生成的模型（如 `minimax_h3_audio_video`、`ltx2_audio_video`）归入**视频**类——上游 MiniMax-H3 和 LTX-2 都在视频生成分类下。

#### 9.2 全局模型表 — new_series 和 version_upgrade 都需要

README 末尾有一张覆盖全部模型的表格，**没有**按图像/视频分区，也**没有** per-model 的 `<details>` 区块和 Quick Start 代码：

- `README.md` → `## All Supported Models`
- `README_zh.md` → `## 全部支持的模型`

表格 **8 列**，比文档页的模型总览表多一个位于最前的「基础架构」列：

```
| 基础架构 | 模型 ID | 推理 | 低显存推理 | 全量训练 | 全量训练后验证 | LoRA 训练 | LoRA 训练后验证 |
| Architecture | Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
```

**插入位置**：
- 表格按**基础架构**分块，同一架构的行必须连续，块内顺序与文档页模型总览表一致
- **架构已在表中**（如新模型基于 LTX-2、Wan 等）→ 把新行插入该架构已有的连续块内
- **全新架构** → 在表头分隔行下方另起新块，即放到**表格最前面**。架构块按接入时间**由新到旧**排列（当前首块 MiniMax-Music3，末块 Stable Diffusion）
- version_upgrade 同理：插入该架构已有的块内，不要追加到表格末尾

**行内容**：
1. 第 1 列填**基础架构名**，不是模型系列名。上游把 JoyAI-Echo 归 `LTX-2`、MOVA 与 LongCat-Video 归 `Wan`。同一架构多行重复填同一个值
2. 第 2 列起与该模型文档页 `## 模型总览` 表格的对应行一致——**链接的脚本必须相同**
3. 链接使用根相对路径 `/examples/{model}/...`，不用绝对 URL
4. 不支持的功能填 `-`

**⚠️ 不要求表头逐字相同**：README 全局表的英文列头是 `Validation After Full Training` / `Validation After LoRA Training`，而文档页用的是 `Full Training Validation` / `LoRA Training Validation`。一致性要求只针对**行内容（链接到哪些脚本）**，不针对列头措辞和列数。

#### 9.3 基础框架分类列表 — 仅 new_series

`README.md` 的 `## Basic Framework` / `README_zh.md` 的 `## 基础框架` 章节里，"Model overview:" / "模型一览："之后是一份**按类别分组的模型列表**。新系列在对应类别下登记。

类别：`Image generation` / `图像生成`、`Video generation` / `视频生成`、`Audio generation` / `音频生成`，各类别内**新的在前**。`Evaluation models` / `评测模型` 是**扁平单行条目**（无缩进子项），不适用于普通模型。

> 该列表不追求覆盖全部模型——上游 MoVA 在全局表有行但不在此列表中。是否登记可向用户确认。

条目格式（注意中英文标点和 ReadTheDocs locale 不同）：

```markdown
    - {Model-Series-Name}: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/{Model-Name}.html), [Example code](/examples/{model}/)
```

```markdown
    - {Model-Series-Name}：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/{Model-Name}.html)、[样例代码](/examples/{model}/)
```

要点：
1. 文档链接用 **ReadTheDocs HTML URL**（`.html` 结尾），不是仓库内 `.md` 路径——这与更新历史条目的链接形式不同
2. locale 段：英文 `/en/latest/`，中文 `/zh-cn/latest/`
3. 中文用全角冒号 `：` 和顿号 `、`；英文用 `:` 和 `,`
4. 缩进为 4 个空格（嵌在类别项之下）

**该章节顶部的精选快速开始表格不要动。** "快速开始，体验热门和最新模型"下面那张 8 列表格是人工精选的子集（当前只有 MiniMax-H3、ACE-Step、Z-Image、Krea-2），并非每个模型都在内——最新的 MiniMax-Music3 就不在。是否收录由维护者决定，本 skill 不自动添加，如认为该模型值得收录，在向用户报告时提一句即可。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 10. 静态验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

检查：
- [ ] 中文文档 `docs/zh/Model_Details/{Model-Name}.md` 存在且格式正确
- [ ] 英文文档 `docs/en/Model_Details/{Model-Name}.md` 存在且格式正确
- [ ] 中英文文档章节标题一一对应（见 `references/doc-templates.md` 中的章节名称映射表）
- [ ] 文档中所有代码链接指向的文件实际存在
- [ ] 模型总览表格中的行数与 `model_inference/` 下的推理脚本数量一致
- [ ] 推理参数列表与 Pipeline `__call__` 签名一致
- [ ] 训练专有参数与 `train.py` argparse 一致
- [ ] 通用训练参数使用固定模板（未擅自修改）
- [ ] 安装章节使用固定模板
- [ ] 样例数据集命令存在
- [ ] 训练结尾固定链接存在
- [ ] 模板章节顺序正确（安装 → 快速开始 → 模型总览 → 模型推理 → 模型训练）
- [ ] `docs/zh/index.rst` 的 Model_Details 区块包含新模型条目
- [ ] `docs/en/index.rst` 的 Model Details 区块包含新模型条目
- [ ] `docs/zh/README.md` 的 Section 2 模型列表包含新模型条目
- [ ] `docs/en/README.md` 的 Section 2 模型列表包含新模型条目
- [ ] `README.md` 的 `## All Supported Models` 表格包含新模型的行
- [ ] `README_zh.md` 的 `## 全部支持的模型` 表格包含新模型的行
- [ ] 同系列的行在表格中连续成块（未被其他系列的行打断）
- [ ] （new_series）新系列块位于表格最前，紧跟表头分隔行
- [ ] （version_upgrade）新变体行插入在该系列已有块内，未追加到表格末尾
- [ ] 全局表各行链接的脚本与文档页模型总览表一致
- [ ] `README.md` / `README_zh.md` 的更新历史已插入新条目，位于最新一条上方
- [ ] （new_series）`README.md` 的 `## Basic Framework` 模型总览列表在对应类别下已登记，用 ReadTheDocs `.html` 链接
- [ ] （new_series）`README_zh.md` 的 `## 基础框架` 模型总览列表同上，locale 为 `/zh-cn/latest/`，标点为全角
- [ ] （如需可选依赖）`docs/zh/Pipeline_Usage/Setup.md` 和 en 版的依赖组说明已更新

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

### 11. 最终验证（内部清理）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

所有文档步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/docs-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/docs-report.md" \
  "packages/{model-name}/.sisyphus/user_report/docs-report.md"; do
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

## 蓝图需求

本 skill 需要从蓝图报告中获取：
- **接入类型**：new_series / version_upgrade
- **功能列表**：用于确定需要生成哪些推理脚本对应的表格行（每个功能 = 一个推理脚本 = 表格一行）
- **Pipeline 类名**：用于定位对应的 Pipeline 文件
- **与已有版本的差异**（version_upgrade 时）

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/docs-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的验证脚本
- **输出目录**: `outputs/` - 命令执行日志

### 生成的文档

- **new_series**：
  - 中文文档：`{diffsynth_root}/docs/zh/Model_Details/{Model-Name}.md`
  - 英文文档：`{diffsynth_root}/docs/en/Model_Details/{Model-Name}.md`
  - `index.rst` 更新：`{diffsynth_root}/docs/zh/index.rst` + `{diffsynth_root}/docs/en/index.rst`
  - 文档主页更新：`{diffsynth_root}/docs/zh/README.md` + `{diffsynth_root}/docs/en/README.md`（Section 2 模型列表追加条目）
  - 可选依赖说明（如需）：`{diffsynth_root}/docs/{zh,en}/Pipeline_Usage/Setup.md`
  - `README.md` 更新：`{diffsynth_root}/README.md`（全局模型表新增行 + 基础框架分类列表登记 + 更新历史条目）
  - `README_zh.md` 更新：`{diffsynth_root}/README_zh.md`（同上）
- **version_upgrade**：
  - 更新中英文现有文档（仅在表格中追加新行），无需修改 `index.rst` 和文档主页
  - `README.md` 更新：全局模型表在该系列已有块内追加新变体行 + 更新历史条目
  - `README_zh.md` 更新：同上

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/docs-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Docs 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-docs |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取文档生成所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、接入类型、Pipeline 功能规划表
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_docs/` 目录及子目录

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示文档生成规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/docs-plan.md`
- 向用户展示接入类型、参考文档、信息提取清单
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/docs-plan.md`

---

### Step 3: 判断接入类型

**目标**：确定是 new_series 还是 version_upgrade。

**执行内容**：
- 从蓝图报告读取接入类型
- 无蓝图时向用户确认

**产出物**：接入类型确认

---

### Step 4: 读取参考文档

**目标**：读取文档模板和参考文档，提取风格。

**执行内容**：
- 读取 `references/doc-templates.md` 文档模板
- new_series：读取 2-3 篇参考文档提取风格
- version_upgrade：读取现有文档了解结构

**产出物**：参考文档风格确认

---

### Step 5: 从代码提取信息

**目标**：从代码中提取文档所需信息。

**执行内容**：
- 读取 Pipeline 文件提取类名、参数签名
- 读取推理脚本列表和用法
- 读取训练脚本路径和参数
- 从 model_configs.py 提取模型 ID

**产出物**：信息提取清单

---

### Step 6: 向用户收集缺失信息

**目标**：向用户收集代码中无法提取的信息。

**执行内容**：
- 请求一句话模型描述
- 请求特殊功能说明

**产出物**：用户补充信息

**注意事项**：
- 用户暂不提供的字段留空或标注 TODO，不阻塞生成

---

### Step 7: 生成/更新文档

**目标**：生成中英文文档。

**执行内容**：
- 按模板结构构建中文文档
- 安装、通用训练参数使用固定模板
- 快速开始代码从推理脚本完整复制
- 同步生成英文文档（章节标题翻译为英文）
- 输出到 `docs/zh/` 和 `docs/en/`

**产出物**：
- `docs/zh/Model_Details/{Model-Name}.md`
- `docs/en/Model_Details/{Model-Name}.md`

---

### Step 8: 更新 docs/ 内的注册点

**目标**：在 index.rst、文档主页和（如需）可选依赖说明中登记新模型。

**执行内容**：
- 在 `docs/zh/index.rst` 的 Model_Details toctree 区块追加条目
- 在 `docs/en/index.rst` 的 Model Details toctree 区块追加条目
- 在 `docs/zh/README.md` 和 `docs/en/README.md` 的 Section 2 模型列表末尾追加条目
- 如模型需要可选依赖，更新 `docs/{zh,en}/Pipeline_Usage/Setup.md`
- version_upgrade 时跳过（条目已存在）

**产出物**：更新后的 index.rst、docs/*/README.md、（如需）Setup.md

---

### Step 9: 更新仓库根 README

**目标**：更新仓库根目录的 README.md 和 README_zh.md（与 Step 8 的文档主页是不同文件）。

**执行内容**：
- 更新历史条目（new_series 和 version_upgrade 都需要）
- new_series：在全局模型表最前插入新系列块
- new_series：在「基础框架」模型总览列表的对应类别下登记（ReadTheDocs 链接）
- version_upgrade：在该系列已有块内追加新变体行

**产出物**：更新后的 README.md 和 README_zh.md

---

### Step 10: 静态验证

**目标**：验证文档格式和内容正确性。

**执行内容**：
- 检查中英文文档存在且格式正确
- 检查章节标题一一对应
- 检查代码链接有效性
- 检查模型总览表格行数与推理脚本数一致
- 检查推理参数与 Pipeline __call__ 签名一致

**产出物**：验证通过确认

---

### Step 11: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 文档规划

### 文档架构设计
- **接入类型**: {new_series / version_upgrade}
- **输出路径**:
  - 中文: `docs/zh/Model_Details/{Model-Name}.md`
  - 英文: `docs/en/Model_Details/{Model-Name}.md`
  - index.rst: `docs/zh/index.rst` + `docs/en/index.rst`
- **参考文档**: 已有的中英文文档路径（提取风格）

### 信息提取清单
- Pipeline 类名、参数
- 推理脚本路径和用法
- 训练脚本路径（如有）
- 模型配置和下载方式

### 用户收集项
- 一句话描述
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/docs-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-11 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 判断接入类型 | Step 3 |
| 4 | 读取参考文档 | Step 4 |
| 5 | 从代码提取信息 | Step 5 |
| 6 | 向用户收集缺失信息 | Step 6 |
| 7 | 生成/更新文档 | Step 7 |
| 8 | 更新 docs/ 内的注册点 | Step 8 |
| 9 | 更新仓库根 README | Step 9 |
| 10 | 静态验证 | Step 10 |
| 11 | 最终验证 | Step 11 |

每完成一个步骤，执行：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/docs-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

文档生成完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/docs-report.md << 'OUTER_EOF'
## ✅ 文档生成完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 📁 生成的文档

- 文档路径: `{diffsynth_root}/docs/zh/Model_Details/{Model-Name}.md`
- 接入类型: {new_series / version_upgrade}

请确认文档内容是否符合预期。

### 📦 代码推送提醒

文档已生成完毕，但文档中引用的代码链接指向的是**你 fork 的 DiffSynth-Studio 仓库**，而非官方仓库。在创建 PR 之前，请确保已将 DiffSynth-Studio 代码推送到你 GitHub fork 仓库的对应模型分支：

    # 1. 先将 origin 重定向到你自己的 fork（如果尚未配置）
    git remote set-url origin https://github.com/{your-username}/DiffSynth-Studio.git

    # 2. 将本地 main 分支推送到 fork 的模型分支（如 ernie-image）
    git push origin main:{model-branch}

    # 3. （可选）添加官方仓库为 upstream，方便后续同步
    git remote add upstream https://github.com/modelscope/DiffSynth-Studio.git

> `{model-branch}` 是当前接好的模型分支名（如 `ernie-image`）。本地代码保持在 main 分支，push 时映射到远程的模型分支即可。
OUTER_EOF
```

## 文档生成原则

- **固定结构**：章节顺序、表格列定义、训练参数分类全部固定
- **功能驱动**：根据实际接入的功能，在固定结构的对应位置填入内容
- **代码为源**：参数、用法、脚本路径都从代码提取，不虚构
- **用户补充**：描述、图片、业务说明由用户提供
- **增量最小**：version_upgrade 只做必要修改，不动已有内容
- **链接格式区分**：
  - **README**（`README.md` / `README_zh.md`）中的代码链接：使用**本地相对路径**（如 `/examples/{model}/model_inference/xxx.py`）
  - **模型文档**（`docs/zh/Model_Details/*.md` / `docs/en/Model_Details/*.md`）中的代码链接：使用**完整 GitHub URL**（如 `https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/{model}/model_inference/xxx.py`）
  - 详见 `references/doc-templates.md` 中的"GitHub 链接前缀"章节
