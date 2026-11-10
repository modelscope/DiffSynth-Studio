---
name: diffsynth-pipeline-training
description: >
  Create training modules and run training verification for an integrated model in DiffSynth-Studio.
  Use this skill after the Pipeline (inference) is fully integrated and E2E tested, or whenever
  the user wants to create training scripts (train.py, full/lora configs, validation scripts)
  or run training to verify the training pipeline works. Can create training for the whole series
  at once — one call handles all training module creation and verification.
  Trigger only when adding training support (LoRA, full training, dataset preparation, training
  verification) as part of a model or feature integration.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — running/fine-tuning with existing training scripts, bug fixes,
  refactoring, performance tuning, or general code changes.
---

# DiffSynth-Studio: Pipeline 训练接入

为已接入推理 Pipeline 的模型系列创建训练模块并完成训练验证。**整个系列共用一个训练脚本**，一次调用完成全部训练模块的创建和验证。

## ⚠️ 核心约束：只改训练相关 Unit

**训练接入时，只能修改与训练直接相关的 Unit（如 `InputXXXEmbedder` 的 `pipe.scheduler.training` 分支、新增训练专用 Unit），绝对禁止修改 Pipeline 的其他任何内容。**

推理 Pipeline 已经过 E2E 测试验证，任何对非训练相关代码的改动都可能破坏已做好的推理流程。

| 允许修改 | 禁止修改 |
|---------|---------|
| `InputXXXEmbedder` 的 `pipe.scheduler.training` 分支逻辑 | DiT/VAE/TextEncoder 等模型组件 |
| 新增 `InputXXXEmbedder` Unit（如缺失） | 推理用 Unit 的 `pipe.scheduler.training=False` 分支 |
| 训练脚本（`train.py`、`.sh` 配置、验证脚本） | Pipeline 的 Unit 链顺序、`__call__` 逻辑 |
| `get_pipeline_inputs` 映射逻辑 | 推理示例脚本 |

**判断标准**：如果改动的代码只在 `pipe.scheduler.training=True` 时执行，可以改；如果改动会影响 `pipe.scheduler.training=False` 的推理路径，不要改。

## ⚠️ 训练使用单卡运行

**所有训练脚本（LoRA、全量、debug 测试）均使用单 GPU 运行，不使用多卡分布式训练。**

通过 `CUDA_VISIBLE_DEVICES` 环境变量在**执行层面**限制使用单卡。不要在训练脚本中写入 `CUDA_VISIBLE_DEVICES` 或 `--num_processes` 等限制——脚本本身应保持干净，GPU 的选择由执行命令控制。

| 训练类型 | GPU 数量 | 执行命令示例 |
|---------|---------|-------------|
| LoRA 训练 | 单卡 | `CUDA_VISIBLE_DEVICES=0 accelerate launch` |
| 全量训练 | 单卡 | `CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ...` |
| Debug 测试 | 单卡 | `CUDA_VISIBLE_DEVICES=0 accelerate launch` |

不要修改训练脚本以添加多卡相关配置。如果用户有多卡需求，由用户自行决定如何调度。

## ⚠️ Step 7/8 执行规则：串行 + 异常清理

**Step 7（Debug 测试）和 Step 8（训练验证）中的所有训练任务必须逐个串行执行，严禁多进程/并行同时运行。** 训练任务会占用大量 GPU 显存和计算资源，并行执行会导致 OOM、训练崩溃或结果不可复现。

| 规则 | 说明 |
|------|------|
| **串行执行** | 每个训练任务必须等前一个完全结束后才能开始下一个，不允许同时启动多个训练进程 |
| **Step 7 内部顺序** | 7a 单阶段 debug →（如 OOM）7b 双阶段 debug → 7c 清理，按序执行，不跳步不并行 |
| **Step 8 内部顺序** | 8a LoRA 完整训练 →（如 OOM）8b 双阶段 LoRA → 8c 全量训练冒烟，严格按序，8c 仅在 8a 成功后执行 |
| **异常清理** | 任何训练进程异常退出时，必须先清理残留子进程后再继续或报告失败 |

**异常清理模板**（每次训练命令前后使用）：

```bash
# 训练开始前：清理可能残留的 GPU 进程
cleanup_residual() {
    # 清理已退出的子进程
    wait 2>/dev/null
    # 检查是否有残留的 accelerate/python 训练进程
    for pid in $(pgrep -f "accelerate.*train.py" 2>/dev/null); do
        echo "WARNING: 发现残留训练进程 PID=$pid，正在清理..."
        kill $pid 2>/dev/null
        sleep 1
        kill -9 $pid 2>/dev/null
    done
}
cleanup_residual

# 训练命令
CUDA_VISIBLE_DEVICES=0 conda run -n {conda_env_name} bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# 训练结束后：确保所有子进程已回收
wait 2>/dev/null

# 检查退出状态
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: 训练异常退出 (exit code=$TRAIN_EXIT_CODE)"
    cleanup_residual
    # 记录错误，根据情况决定是否继续
fi
```

**关键约束**：
- **禁止在后台执行任何训练进程**：Step 7/8 中所有训练命令必须在前台同步运行，严禁使用 `&` 将训练任务放入后台
- 禁止使用 `parallel`、`xargs -P`、多进程 Python 脚本等方式并行执行训练
- 全量训练冒烟测试（8c）如需要限时停止，使用 `timeout` 命令在前台控制，**不得**使用 `&` 后台启动 + `kill` 的方式
- 任何命令执行后必须 `wait` 确保无残留子进程，发现残留进程必须先清理再继续

## ⚠️ accelerate config `num_processes` 恢复规则

**`accelerate_config_zero3.yaml` 中的 `num_processes: 8` 是标准生产配置。**

在全量训练冒烟测试（Step 8c）中，为加快验证速度，可以临时将 `num_processes` 改为 `1`（单卡运行）。**但测试结束后，必须立即将 `num_processes` 改回 `8`，恢复原始生产配置。**

| 阶段 | `num_processes` | 说明 |
|------|----------------|------|
| 正常状态（创建时 / 测试后） | `8` | 标准生产配置，用于实际多卡训练 |
| 全量训练冒烟测试（Step 8c） | `1` | 临时改为单卡，加快验证速度 |

**操作流程**：
```bash
# Step 8c 开始前：临时改为单卡
sed -i 's/^num_processes: 8/num_processes: 1/' examples/{series}/model_training/full/accelerate_config_zero3.yaml

# 执行冒烟测试...
timeout 120 conda run -n {conda_env_name} bash examples/{series}/model_training/full/{ModelName}.sh ...

# Step 8c 结束后：立即恢复为 8（无论测试成功或失败）
sed -i 's/^num_processes: 1/num_processes: 8/' examples/{series}/model_training/full/accelerate_config_zero3.yaml
```

**关键**：
- 必须在 Step 8c 完成后立即恢复，不可遗忘
- 即使冒烟测试中途报错中断，也必须先恢复 `num_processes` 再继续后续步骤
- LoRA 训练和 Debug 测试不涉及此配置修改，保持原样即可

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径**：
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `target_path`: `packages/{model-name}/{target-library}/`（从蓝图报告获取）
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**必需前提**：
- `diffsynth-pipeline` 已完成推理 Pipeline 接入
- E2E 端到端测试已通过
- Pipeline 中已实现 `InputXXXEmbedder` Unit

## 训练架构

DiffSynth 的训练框架天然保证推理-训练一致性。**能推理就能训练**——Pipeline 通过 `InputXXXEmbedder` 和 `scheduler.training` 标志自动切换训练/推理行为。

### 训练数据流

```
数据集 (data) → get_pipeline_inputs() → inputs_shared, inputs_posi, inputs_nega
    → transfer_data_to_device() → Unit 链执行（InputXXXEmbedder 是关键）
    → FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi) → loss
```

### `InputXXXEmbedder` 训练分支

`InputXXXEmbedder` 的训练分支编写是步骤 5.1 的核心工作，详细内容见下方步骤 5.1 章节。简要要点：
- 已有 `InputXXXEmbedder`：只需确认/补全 `pipe.scheduler.training=True` 分支
- 没有 `InputXXXEmbedder`：按标准实现创建
- **绝对禁止**修改推理路径（`pipe.scheduler.training=False`）的任何逻辑

### Loss 函数所需字段

| Loss | 必需字段 |
|------|---------|
| `FlowMatchSFTLoss` | `input_latents` |
| `FlowMatchSFTAudioVideoLoss` | `input_latents` + `audio_input_latents` |

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pipeline-training-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pipeline-training-report.md << EOF

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
| 基本信息表中的 `Conda 环境名称` | 训练环境。**本 skill 执行的所有 Python/accelerate 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 或 `conda run -n {conda_env_name} accelerate ...` 形式。** |
| 基本信息表中的 `目标库路径` | 训练参考 |
| Pipeline 功能规划表 | Pipeline 类名、系列名称 |
| 训练相关信息（如有） | 训练方式（full/lora）、数据集格式 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pipeline_training"
mkdir -p ${EXEC_LOG_DIR}/{outputs,checkpoints,scripts}
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始创建训练模块前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/pipeline-training-plan.md`。** 基于蓝图信息和已有的推理 Pipeline，明确训练模块架构、数据集方案、训练脚本清单、InputEmbedder 训练分支、Loss 函数选择、Debug 测试方案和完整训练方案。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、参考」，详见 Plan 模板章节。

**Plan 越详细越好。** 详细的 Plan 能帮助执行者理解每个步骤的具体内容、注意事项和产出物，减少执行过程中的猜测和错误。每个步骤的描述应具体到可以直接照着执行的程度，不要省略关键细节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始编写代码。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 3. 分析训练需求

> 📖 开始前：重读本步骤描述，确认流程与报告路径

确定：{SeriesName}（与 Pipeline 类名一致）、{series}（小写目录名）、{ModelName}（与训练脚本文件名一致）。具体训练任务类型和数据集字段在 Step 4 中分析。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 4. 训练任务分析与数据集确认

> 📖 开始前：重读本步骤描述，确认流程与报告路径

DiffSynth 的训练框架遵循**训推一致**原则：**能推理就能训练**。Pipeline 通过 `InputXXXEmbedder` 和 `scheduler.training` 标志自动切换训练/推理行为，无需编写独立的训练 Pipeline。

本步骤的核心任务是：从已有的推理 Pipeline 出发，分析训练所需的数据字段，确认映射关系，构建数据集。

#### 4a. 训练任务分析

**原则**：推理的输入和输出决定了训练的字段。

| 推理行为 | 对应训练字段 | 说明 |
|---------|-------------|------|
| 推理接受 `prompt`，输出 `image` | 数据集需要 `prompt` + `image` 列 | `image` 在训练代码中会被映射到 `input_image` 参数 |
| 推理接受 `prompt`，输出 `video` | 数据集需要 `prompt` + `video` 列 | `video` 在训练代码中会被映射到 `input_video` 参数 |
| 推理接受额外参数（如 `edit_image`） | 数据集需要对应字段（如 `edit_image` 列） | 通过 `--extra_inputs` 配置，由 `parse_extra_inputs` 自动注入 |
| 推理的 `noise`/`rand_device` | **不需要数据集提供** | 由框架在训练时自动生成，不属于数据集字段 |

**分析方法**：

1. 打开 `pipeline.py`，列出 Pipeline 中所有 Unit 及其 `input_params` 声明
2. 区分哪些是**推理时由用户传入的参数**（如 `prompt`、`input_image`、`edit_image`），哪些是**框架内部参数**（如 `noise`、`rand_device`、`cfg_scale`、`tiled`）
3. 推理时用户传入的参数 = 训练时数据集需要提供的字段

以 LTX-2 视频模型为例：
```
推理: prompt → video
训练: 数据集需 prompt + video 列
```

以图像编辑模型为例：
```
推理: prompt + edit_image → edited_image
训练: 数据集需 prompt + edit_image + image 列
```

#### 4b. 确认字段与映射关系

**核心原则：训推一致，数据集列名 = Unit `input_params`。** 训练不需要额外的映射层。Pipeline Unit 链中声明什么参数名，数据集就用什么列名。

**数据流**：`data["edit_image"]` → `parse_extra_inputs` 直接注入 → `inputs_shared["edit_image"]` → Unit `process(self, pipe, edit_image, ...)`

核心字段（`image`/`video`）在 `get_pipeline_inputs` 中被手动映射到 `input_image`/`input_video`，这是因为数据集复用约定（见 4d），**额外字段不做此类映射**。

**数据集复用时的列名处理**：相同任务类型可以复用数据集，但如果列名与当前模型的 Unit `input_params` 不一致，**修改数据集文件的列名**，不修改代码做适配。

**字段确认清单**：
- [ ] 列出 Pipeline 所有 Unit 的 `input_params`
- [ ] 标记哪些是用户传入参数（需要数据集字段），哪些是框架参数（不需要）
- [ ] 确认数据集 `metadata.csv` 列名与 Unit `input_params` 完全一致
- [ ] 复用数据集时，如有列名不一致，修改 `metadata.csv` 列名使其匹配

**⚠️ 输出：本次任务的列名规划**

完成上述分析后，必须明确输出本次任务的列名规划表，格式如下：

```
## 列名规划

**任务类型**: {例如：图像编辑 / 基础图像生成 / 视频生成 / 音视频联合}

| 列名 | 来源 | 说明 | 是否数据文件 |
|------|------|------|-------------|
| image | InputImageEmbedder | 输入图像 | 是 |
| edit_image | EditImageEmbedder | 编辑目标/参考图像 | 是 |
| prompt | TextEmbedder | 文本 prompt | 否 |

**data_file_keys**: "image,edit_image"（包含文件路径的列）
**extra_inputs**: "edit_image"（需注入 Pipeline 的列，与 Unit input_params 一致）
```

要点：
- `data_file_keys` = 所有"是否数据文件"为**是**的列名，逗号分隔
- `extra_inputs` = 除了 `prompt`、`image`/`video` 之外，所有需要注入 Pipeline 的列名（同一列可同时出现在 `data_file_keys` 和 `extra_inputs` 中，分别负责文件加载和值注入）
- 此规划表将直接用于 Step 6 训练脚本的 `--data_file_keys` 和 `--extra_inputs` 参数配置

#### 4c. 确定数据集构建方案

根据 4a 和 4b 的分析结果，确定：

**数据集存储路径**（与训练脚本的目录结构严格对应）：
```
{diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
├── metadata.csv          # 数据索引文件（列名由 4b 确认）
├── images/               # 图像文件（图像任务）
└── videos/               # 视频文件（视频任务）
```

其中 `{series}/{ModelName}/` 与训练脚本所在的目录结构严格对应。例如训练脚本位于 `examples/qwen_image/model_training/lora/Qwen-Image-Edit.sh`，则数据集路径为 `qwen_image/Qwen-Image-Edit/`。

**metadata.csv 列定义**：根据 4b 的字段确认结果，列出所有必需列和可选列。

| 任务类型 | 必需列 | 可选列 |
|---------|--------|--------|
| 基础图像生成（T2I） | `image`, `prompt` | - |
| 图像编辑 | `image`, `edit_image`（列名以 Unit `input_params` 为准）, `prompt` | - |
| 视频生成（T2V） | `video`, `prompt` | `frame_rate` |
| 音视频联合（T2AV） | `video`, `prompt`, `input_audio` | `frame_rate` |
| 多图像输入 | `image`（JSON 数组）, `prompt` | - |

**数据来源方案**（按优先级，三选一）：
1. **复用已有数据集**：**默认首选方案**（见 4d），同类型任务直接复制已有数据集
2. **从 Pipeline 推理输出创建样例数据集**：当没有可复用数据集时使用（见 4d），利用已完成推理 Pipeline 的生成结果作为训练数据
3. **用户提供数据集**：当以上两种方案均不可用时，请用户提供数据集后再继续

**⚠️ 重要**：数据集的列名、文件路径格式必须在本文档中明确记录，后续 Step 5 创建训练代码时，`get_pipeline_inputs` 方法、`UnifiedDataset` 的 `data_file_keys` 和 `special_operator_map` 都将依赖此处的定义。

#### 4d. 构建数据集

**⚠️ 必须先 `cd` 到 DiffSynth-Studio 根目录**，确保 `./data/diffsynth_example_dataset` 路径相对于 DiffSynth-Studio 根目录。

**下载样例数据集**（推荐，可获取所有系列的数据）：

```bash
# 必须先在 DiffSynth-Studio 根目录下执行
cd {diffsynth_root}
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

**方案一：复用已有数据集**（同类型任务，默认首选）：

```bash
# 示例：为基础图像生成任务复用 Qwen-Image 数据集
mkdir -p {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}
cp -r {diffsynth_root}/data/diffsynth_example_dataset/qwen_image/Qwen-Image/* \
       {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
```

各任务类型可复用的源数据集：

| 新模型任务 | 复用源数据集 |
|-----------|-------------|
| 基础 T2I | `qwen_image/Qwen-Image/` 或 `z_image/Z-Image/` |
| 图像编辑 | `qwen_image/Qwen-Image-Edit/` |
| T2V 视频 | `wanvideo/Wan2.1-T2V-14B/` |
| T2AV 音视频 | `ltx2/LTX-2.3-T2AV-splited/` |

**⚠️ 复用后必须检查 metadata.csv 列名**：

复制完成后，将源数据集的 `metadata.csv` 列名与 4b 输出的**列名规划表**逐一对比：

```bash
# 查看源数据集列名
head -1 {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv
```

- **如果列名与规划表完全一致**：无需额外操作，直接进入下一步
- **如果列名不一致**：修改 `metadata.csv` 的列名使其与规划表（即 Unit `input_params`）完全匹配。**修改数据集文件的列名，不修改代码做适配。** 重命名仅改变列名以对齐语义，数据内容本身不变。

```bash
# 示例：源数据集列名为 ref_image，而目标 Unit 的 input_params 是 edit_image
# 语义相同，仅名称不同——直接替换列名即可
cd {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
sed -i 's/^ref_image,/edit_image,/' metadata.csv
```

**方案二：从 Pipeline 推理输出创建样例数据集**（当没有可复用数据集时使用）：

由于推理 Pipeline 已经完成 E2E 测试，推理输出（生成的图像/视频及对应的 prompt）可以直接用作训练样例数据。这保证了训练数据与模型能力的一致性。

1. **定位推理输出**：
   ```bash
   # 推理输出位于 Pipeline 执行日志目录
   PIPELINE_EXEC_LOG="packages/{model-name}/.sisyphus/execution-logs/latest/outputs/"
   # 或查找最近的 pipeline 推理输出
   find packages/{model-name}/.sisyphus/execution-logs/ -name "*.jpg" -o -name "*.png" -o -name "*.mp4" | head -20
   ```

2. **获取推理 prompt**：从 Pipeline 推理脚本或执行日志中提取使用的 prompt。
   - 查看推理脚本：`packages/{model-name}/.sisyphus/execution-logs/*/scripts/*.py` 或 `*.sh`
   - 查看执行日志中的 prompt 记录：`grep -i "prompt" packages/{model-name}/.sisyphus/execution-logs/*/outputs/*.log`

3. **创建数据集目录**：
   ```bash
   mkdir -p {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/{images|videos}/
   ```

4. **复制推理输出作为数据文件**：
   ```bash
   # 图像任务：复制推理生成的图像
   cp ${PIPELINE_EXEC_LOG}/*.jpg {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/images/
   # 或复制为编号格式
   for i in ${PIPELINE_EXEC_LOG}/*.jpg; do cp "$i" "{diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/images/$(printf '%03d' $((++n))).jpg"; done

   # 视频任务：复制推理生成的视频
   cp ${PIPELINE_EXEC_LOG}/*.mp4 {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/videos/
   ```

5. **编写 metadata.csv**，列名必须与 4b 确认的字段一致，prompt 使用推理时的实际 prompt：
   ```bash
   cd {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
   # 根据实际推理输出数量和 prompt 编写 metadata.csv
   ```

6. **⚠️ 向用户展示创建结果，等待确认**：

   数据集创建完成后，向用户展示以下内容，等待用户 check 确认后再继续后续步骤：

   ```
   ## 样例数据集已创建，请确认

   **路径**: `{diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/`

   **文件清单**:
   - metadata.csv（{N} 条样例）
   - images/001.jpg, images/002.jpg, ...（或 videos/）

   **metadata.csv 内容**:
   ```
   {显示 metadata.csv 内容}
   ```

   **数据来源**: Pipeline 推理输出（执行日志: packages/{model-name}/.sisyphus/execution-logs/latest/outputs/）
   **Prompt 来源**: 推理时使用的实际 prompt

   请确认数据集内容无误，如有需要调整的地方请告知。
   ```

   - **用户确认后**：继续后续 Step
   - **用户要求修改**：根据反馈调整 metadata.csv 或数据文件，再次展示确认

**方案三：用户提供数据集**（当前两种方案均不可用时）：

当推理输出也不存在或不足时，向用户说明以下信息后，请用户提供数据集再继续：

- 当前模型的任务类型（T2I 图像生成 / T2V 视频生成 / T2AV 音视频 / 图像编辑等）
- 数据集需要的字段（根据 4b 确认结果）
- 建议的数据量（至少 2-5 条样例即可用于训练验证）
- 数据格式要求（图像：jpg/png，视频：mp4，prompt：文本描述）

**metadata.csv 格式示例**：

基础图像生成：
```csv
image,prompt
images/001.jpg,a beautiful sunset
images/002.jpg,a mountain landscape
```

视频生成：
```csv
video,prompt,frame_rate
videos/001,a person walking,24
videos/002,a cat running,24
```

图像编辑（列名以 Unit `input_params` 为准）：
```csv
image,edit_image,prompt
images/001.jpg,images/001_edit.jpg,change the sky to night
```

详见 [references/dataset-guidelines.md](references/dataset-guidelines.md)。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 5. 创建训练代码（Python）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**本步骤仅创建 Python 代码**（InputEmbedder 训练分支 + train.py），`.sh` 训练脚本在 Step 6 中创建。

按以下顺序逐步创建：

**步骤 5.1：编写 InputEmbedder 训练分支**

训练接入的核心工作是确保 `InputXXXEmbedder` 在 `pipe.scheduler.training=True` 时返回正确的 `input_latents`。
- 已有 `InputXXXEmbedder`：检查其 `pipe.scheduler.training` 分支是否完整，只需补全训练分支
- 没有 `InputXXXEmbedder`：需要新增
- **禁止**修改推理路径（`pipe.scheduler.training=False`）的任何逻辑

InputEmbedder 的标准实现模式和 LTX-2 参考代码见 [references/training-code-templates.md](references/training-code-templates.md)。

**核心要求**：
- 推理模式（`training=False`）**不返回 `input_latents`**
- 训练模式返回包含 `input_latents` 的 dict
- VAE 编码复用 Pipeline 已有的 `preprocess_xxx`、`vae.encode` 等方法
- `output_params` 声明了所有可能返回的 key
- 不要盲目添加 VAE 不支持的 Tile 参数
- 不要在 `InputXXXEmbedder` 中处理特殊功能输入（编辑图像、mask 等）

**步骤 5.2：创建 `{SeriesName}TrainingModule` 类**

继承 `DiffusionTrainingModule`，实现三个方法：
- `__init__`：加载 Pipeline、设置训练模式、定义 `task_to_loss`（大部分可复用模板）
- `get_pipeline_inputs`：将数据集字段映射到 Pipeline 输入（核心规则如下）
  - `inputs_posi` 放文本条件（如 `{"prompt": data["prompt"]}`）
  - `inputs_nega` 必须为空（`{"negative_prompt": ""}`）
  - `inputs_shared` 放数据 + 框架控制参数（`cfg_scale=1`、`use_gradient_checkpointing` 等）
  - 核心字段 key 名（`input_image`/`input_video`）必须与对应 Embedder Unit 的 `input_params` 一致
  - 额外字段通过 `parse_extra_inputs` 自动注入
  - 图像和视频任务的差异仅在核心字段映射（`input_image` vs `input_video` + `num_frames`）
- `forward`：入口函数（**标准实现，不要修改**）

完整模板见 [references/training-code-templates.md](references/training-code-templates.md)。需配置项清单及决定方案见 [references/training-scripts.md](references/training-scripts.md)。

**步骤 5.3：创建 `train.py`**

完整的训练代码模板（含图像任务、视频任务、验证脚本）见 [references/training-code-templates.md](references/training-code-templates.md)。

**创建流程**：
1. 选择图像或视频任务模板，替换 `{series}`、`{SeriesName}` 等占位符
2. 确认 `get_pipeline_inputs` 中的字段映射与 Step 4 分析的数据集字段一致
3. 确认 `task_to_loss` 和 `launcher_map` 仅保留 SFT 相关条目（`sft:data_process`、`sft`、`sft:train`）
4. 保存为 `examples/{series}/model_training/train.py`

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 6. 创建训练脚本（.sh）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

本步骤创建 `.sh` 训练脚本。完整模板、参数分析和需配置项决定方案见 [references/training-scripts.md](references/training-scripts.md)。

#### 6a. 脚本结构

所有 `.sh` 训练脚本遵循统一结构：数据集下载注释 → 固定参数 → 需配置参数。

**完整模板见 [references/training-scripts.md](references/training-scripts.md)。以下概述结构差异。**

LoRA vs 全量训练脚本差异对照：

| 差异项 | LoRA 脚本 | Full 脚本 |
|--------|-----------|-----------|
| `accelerate launch` | 直接运行 | `--config_file .../accelerate_config_zero3.yaml` |
| `--learning_rate` | `1e-4` | `1e-5` |
| `--num_epochs` | `5` | `2` |
| `--dataset_repeat` | `50`（图像）/ `100`（视频） | `50`（图像）/ `100`（视频） |
| `--lora_base_model` | 需配置 | 不存在 |
| `--lora_target_modules` | `""`（固定） | 不存在 |
| `--lora_rank` | `32` | 不存在 |
| `--trainable_models` | 不存在 | `"dit"` |
| `--output_path` | `*_lora` | `*_full` |

**完全固定参数**（两种训练脚本相同，直接复制）：
- `--learning_rate`：LoRA `1e-4`，Full `1e-5`
- `--lora_rank`：`32`（仅 LoRA）
- `--lora_target_modules`：`""`（仅 LoRA）— 交给框架自动探测注入层。**必须显式写出这一行**，省略会退回 parser 默认值 `"q,k,v,o,ffn.0,ffn.2"`（Wan 专用层名），非 Wan 架构启动即报 `ValueError`；探测盲区与兜底取值见 [references/training-scripts.md](references/training-scripts.md) §3
- `--use_gradient_checkpointing`：强制包含
- `--trainable_models`：`"dit"`（仅 Full）
- `--find_unused_parameters`：始终包含

**需根据模型配置的参数**（共 10 项，见下方决定方案）：
- `{series}` / `{ModelName}` — Step 3 分析结果
- `--dataset_base_path` / `--dataset_metadata_path` — 基于 series 和 ModelName 拼接
- `--model_id_with_origin_paths` — 从 pipeline.py ModelConfig 列表复制
- `--lora_base_model` — Pipeline 中主扩散模型的属性名（如 self.dit → "dit"）
- `--remove_prefix_in_ckpt` — 与 lora_base_model 对应，格式 `"pipe.{属性名}."`
- `--data_file_keys` — metadata.csv 中文件路径列
- `--extra_inputs` — 需要注入 Pipeline 的额外列
- 数据集下载行 — 基于 series/ModelName
- `--max_pixels` / `--height/--width/--num_frames` — 图像 vs 视频
- `--dataset_repeat` — 图像 50，视频 100
- `--output_path` — 基于 ModelName 和训练类型

#### 6b. LoRA 训练脚本

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "{series}/{ModelName}/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
accelerate launch examples/{series}/model_training/train.py \
    --learning_rate 1e-4 --num_epochs 5 \
    --lora_rank 32 \
    --lora_target_modules "" \
    --use_gradient_checkpointing --find_unused_parameters

# ===== 需配置参数（按下方说明逐一替换） =====
    --dataset_base_path "./data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    --model_id_with_origin_paths "..." \
    --lora_base_model "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --output_path "./models/train/{ModelName}_lora" \
    --data_file_keys "image" \
    --extra_inputs "edit_image"
```

**需配置参数说明**：
- `--remove_prefix_in_ckpt` 与 `--lora_base_model` 对应：`"dit"` → `"pipe.dit."`，`"video_dit"` → `"pipe.video_dit."`，`"vace"` → `"pipe.vace."`
- `--data_file_keys`：metadata.csv 中包含文件路径的列（image/video/audio），逗号分隔
- `--extra_inputs`：需要注入 Pipeline 的额外列名（如 edit_image, input_audio）。**同一列名可能同时出现在 data_file_keys 和 extra_inputs 中**
- `--max_pixels`：视频任务改用 `--height`/`--width`/`--num_frames`
- 无额外输入时 `--extra_inputs` 可省略

完整模板见 [references/training-scripts.md](references/training-scripts.md) §1.1。

#### 6c. 全量训练脚本

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "{series}/{ModelName}/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
accelerate launch --config_file examples/{series}/model_training/full/accelerate_config_zero3.yaml \
    examples/{series}/model_training/train.py \
    --learning_rate 1e-5 --num_epochs 2 \
    --trainable_models "dit" --use_gradient_checkpointing --find_unused_parameters

# ===== 需配置参数（按下方说明逐一替换） =====
    --dataset_base_path "./data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    --model_id_with_origin_paths "..." \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --output_path "./models/train/{ModelName}_full" \
    --data_file_keys "image" \
    --extra_inputs "edit_image"
```

**`accelerate_config_zero3.yaml`**（如目录中不存在则创建）：
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

#### 6d. `--data_file_keys` 与 `--extra_inputs` 配置

这两个参数常被混淆，理解其作用后配置很简单：

| 参数 | 作用 | 示例 |
|------|------|------|
| `--data_file_keys` | metadata.csv 中**包含文件路径的列**，会被 `UnifiedDataset` 通过 `special_operator_map` / `main_data_operator` 加载为 tensor/PIL | `"image"`, `"video,input_audio"`, `"image,edit_image"` |
| `--extra_inputs` | metadata.csv 中**需要注入 Pipeline 推理流程**的额外列名（不包括 image/video/prompt），值通过 `parse_extra_inputs` 进入 `inputs_shared` 供 Pipeline Units 使用 | 无额外输入省略, `"edit_image"`, `"input_audio,input_image"` |

**同一列名可同时出现在两个参数中**——前者负责文件加载，后者负责值注入：

| 任务 | data_file_keys | extra_inputs |
|------|---------------|-------------|
| 基础 T2I | `"image"` | 无 |
| 图像编辑 | `"image,edit_image"` | `"edit_image"` |
| 多层图像 | `"image,layer_input_image"` | `"layer_num,layer_input_image"` |
| 音视频联合 | `"video,input_audio"` | `"input_audio"` |
| 音视频+参考图 | `"video,input_audio"` | `"input_audio,input_image"` |

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 7. Debug 测试（快速验证）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

⚠️ **串行执行**：Step 7 所有子步骤按序执行，严禁并行。异常退出时先清理残留进程。详见上方「Step 7/8 执行规则：串行 + 异常清理」章节。

**⚠️ 训练代码稳定性**：DiffSynth 的训练模板代码经过验证，**若无必要，不要修改模板代码**。如果训练出现问题，必须先按以下顺序排查：

1. **检查数据集**：确认 `metadata.csv` 列名与 Unit `input_params` 一致、数据文件路径正确、数据格式正确
2. **检查训练脚本配置**：确认 `--dataset_base_path`、`--dataset_metadata_path`、`--extra_inputs` 等参数正确；确认 `--lora_target_modules ""` 这一行仍然存在（被删掉不会触发自动探测，而是退回 parser 默认的 Wan 层名列表）
3. **确认以上两者无误后**，再排查是否是训练模板代码的问题

只有确认数据集和训练脚本配置都无误，且问题依然存在时，才考虑修改训练模板代码。

**⚠️ 优先使用 LoRA 训练进行 debug**，全量训练仅在 LoRA 跑通后验证使用。

**⚠️ LoRA OOM 自动切换到双阶段训练**：如果 debug 过程中出现 OOM，**自动将 LoRA 脚本改写为双阶段训练**（详见 [references/split-training.md](references/split-training.md)）。全量训练保持单阶段方案，不做双阶段切换。

#### 7a. 单阶段 debug 测试

1. 基于 LoRA 训练脚本创建 debug 脚本 `examples/{series}/model_training/lora/{ModelName}-test.sh`：将 `--num_epochs` 改为 `1`、`--dataset_repeat` 改为 `1`
2. 运行：`cd {diffsynth_root} && CUDA_VISIBLE_DEVICES=0 conda run -n {conda_env_name} bash examples/{series}/model_training/lora/{ModelName}-test.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_test.log`
3. 确认：无报错退出 + 输出目录有 LoRA checkpoint
4. 确认日志里 `LoRA will be patched at [...]` 非空，且覆盖到 attention / FFN 层。**为空**说明该 DiT 的 block 容器不是 `nn.ModuleList`（自动探测失效），需按 [references/training-scripts.md](references/training-scripts.md) §3.2–§3.4 改为显式列表；**非空但 loss 不降**则检查是否漏掉了关键层

#### 7b. OOM 处理：切换到双阶段训练

如果 7a 出现 OOM（日志中包含 `OutOfMemoryError`、`CUDA out of memory`）：

1. **将 `lora/{ModelName}.sh` 改写为双阶段脚本**（详见 [references/split-training.md](references/split-training.md)）：
   - 第一阶段（`--task "sft:data_process"`）：`--model_id_with_origin_paths` 保持完整，用 `--offload_models` 填入 DiT 条目，**`--dataset_repeat` 改为 `1`**，输出到 `"./models/train/{ModelName}-LoRA-splited-cache"`
   - 第二阶段（`--task "sft:train"`）：`--model_id_with_origin_paths` 保持完整，用 `--offload_models` 填入 TextEncoder + VAE 条目，`--dataset_base_path` 指向缓存路径，**删除 `--dataset_metadata_path`**（进入 `load_from_cache` 模式），**`--dataset_repeat` 保持原值不变**（50 或 100），输出到 `"./models/train/{ModelName}-LoRA-splited"`

2. **创建双阶段 debug 脚本** `lora/{ModelName}-test.sh`：
   - 基于双阶段脚本，将 `--num_epochs` 改为 `1`，**两阶段的 `--dataset_repeat` 都改为 `1`**（debug 专用）
   - `--output_path` 使用 `LoRA-splited-cache` 和 `LoRA-splited` 后缀

3. 运行双阶段 debug：
   ```bash
   cd {diffsynth_root} && CUDA_VISIBLE_DEVICES=0 conda run -n {conda_env_name} bash examples/{series}/model_training/lora/{ModelName}-test.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_test.log
   ```

4. 确认：两阶段均正常完成 + 输出目录有 LoRA checkpoint

#### 7c. 清理

Debug 测试脚本（`{ModelName}-test.sh`）保留在 DiffSynth-Studio 仓库的 `examples/{series}/model_training/lora/` 目录下，方便后续调试复用。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 8. 训练验证（完整运行）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

⚠️ **串行执行**：Step 8 所有子步骤按序执行，严禁并行。异常退出时先清理残留进程。详见上方「Step 7/8 执行规则：串行 + 异常清理」章节。

**⚠️ 此步骤使用真实训练参数，禁止任何形式的降参偷懒。**

Debug 测试（Step 7）可以用降参方式快速验证链路是否正确，但 **Step 8 训练验证是唯一产出真实 checkpoint 的环节**，必须使用正式训练脚本中的全部默认参数运行。

**严格禁止的行为**：
- 降低 `--num_epochs`（LoRA 必须 5 轮，Full 必须 2 轮）
- 降低 `--dataset_repeat`（保持 50 不变）
- 降低 `--max_pixels` / 分辨率
- 降低学习率或更改 optimizer 配置

Debug 阶段的降参脚本 `lora/{ModelName}-test.sh` **仅用于验证链路**，不可替代 Step 8。

---

Debug 测试通过后，运行完整的正式训练流程，产出真实 checkpoint 供用户确认。全量训练只做冒烟测试，验证启动和脚本路径。

**8a. LoRA 完整训练**：
1. 运行：`cd {diffsynth_root} && CUDA_VISIBLE_DEVICES=0 conda run -n {conda_env_name} bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log`
2. 确认：训练正常完成，输出目录包含完整 LoRA checkpoint 文件
3. **⚠️ LoRA 推理验证：必须输出验证图像** — 使用训练的 LoRA 进行推理测试，**将验证图像保存到 `${EXEC_LOG_DIR}/outputs/lora_result.jpg`，此文件是用户审查训练效果的依据，不可省略**。

   **⚠️ 验证脚本的 prompt 必须来自 `metadata.csv` 第一行的 prompt 列**（配上第一行对应的输入文件），不可随意编造或使用其他来源的 prompt。验证的目的是确认 LoRA 是否学到了训练集的内容，因此必须用训练集本身的数据进行测试。

   ```bash
   cd {diffsynth_root}
   # 使用验证脚本加载 LoRA 进行推理（参考 references/training-code-templates.md 中的验证脚本模板）
   # prompt 必须用 metadata.csv 第一行的内容，输入文件也用第一行对应的数据
   # 必须保存验证图像到 ${EXEC_LOG_DIR}/outputs/lora_result.jpg
   ```
4. 将验证图像保存到 `${EXEC_LOG_DIR}/outputs/lora_result.jpg`

**8b. LoRA OOM 处理**：如果 8a 出现 OOM，自动切换到双阶段训练：

1. **将 `lora/{ModelName}.sh` 改写为双阶段脚本**（详见 [references/split-training.md](references/split-training.md)）：
   - 第一阶段（`--task "sft:data_process"`）：模型列表保持完整，`--offload_models` 填入 DiT 条目，**`--dataset_repeat` 改为 `1`**，输出到 `"./models/train/{ModelName}-LoRA-splited-cache"`
   - 第二阶段（`--task "sft:train"`）：模型列表保持完整，`--offload_models` 填入 TextEncoder + VAE 条目，`--dataset_base_path` 指向缓存路径，**删除 `--dataset_metadata_path`**，**`--dataset_repeat` 保持原值不变**（50 或 100），输出到 `"./models/train/{ModelName}-LoRA-splited"`

2. 重新运行双阶段 LoRA 训练：
   ```bash
   cd {diffsynth_root} && CUDA_VISIBLE_DEVICES=0 conda run -n {conda_env_name} bash examples/{series}/model_training/lora/{ModelName}.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/lora_train.log
   ```

3. **⚠️ 必须输出验证图像** — 使用第二阶段输出的 LoRA 进行推理，将验证图像保存到 `${EXEC_LOG_DIR}/outputs/lora_result.jpg`。

   **⚠️ 验证脚本的 prompt 必须来自 `metadata.csv` 第一行的 prompt 列**（配上第一行对应的输入文件），不可随意编造。验证的目的是确认双阶段训练后的 LoRA 是否学到了训练集的内容，因此必须用训练集本身的数据。

4. 在报告中记录：单阶段 OOM → 切换到双阶段训练成功

**8c. 全量训练**（Smoke Test）：

仅在 LoRA 完整训练成功后执行。全量训练耗时较长，此处只做冒烟测试：验证训练能正常启动、运行若干 step 后正常终止。

**⚠️ 冒烟测试前临时修改 `num_processes`：** 为加快验证速度，先将 `accelerate_config_zero3.yaml` 中的 `num_processes` 从 `8` 改为 `1`（单卡运行），**测试结束后必须立即改回 `8`**。详见上方「accelerate config `num_processes` 恢复规则」章节。

1. **临时改为单卡**：
   ```bash
   cd {diffsynth_root}
   sed -i 's/^num_processes: 8/num_processes: 1/' examples/{series}/model_training/full/accelerate_config_zero3.yaml
   ```

2. 启动测试（**前台运行，使用 `timeout` 限时停止，禁止 `&` 后台执行**）：
   ```bash
   cd {diffsynth_root}
   timeout 120 conda run -n {conda_env_name} bash examples/{series}/model_training/full/{ModelName}.sh > ${EXEC_LOG_DIR}/outputs/full_train.log 2>&1
   FULL_EXIT_CODE=$?
   # timeout 退出码 124 = 超时到达（预期行为），0 = 正常完成，其他 = 异常
   wait 2>/dev/null
   # 确认无残留进程
   for pid in $(pgrep -f "accelerate.*train.py" 2>/dev/null); do
       kill $pid 2>/dev/null
       sleep 0.5
       kill -9 $pid 2>/dev/null
   done
   wait 2>/dev/null
   grep -i "loss" ${EXEC_LOG_DIR}/outputs/full_train.log | tail -5
   ```

3. **⚠️ 恢复 `num_processes` 为 8（无论测试成功或失败，必须执行）**：
   ```bash
   sed -i 's/^num_processes: 1/num_processes: 8/' examples/{series}/model_training/full/accelerate_config_zero3.yaml
   ```

4. 确认：日志中包含 loss 输出、训练正常启动、无报错
5. **OOM 处理**：如果单卡仍然 OOM：
   - 记录 OOM 日志到 `${EXEC_LOG_DIR}/outputs/full_train_oom.log`
   - 停止全量训练冒烟测试，标记为 `⏭️ OOM 跳过`
   - 在渐进式报告和用户报告中**明确记录 OOM 原因**
   - 向用户说明 OOM 情况，由用户决定是否调整显存策略
6. 记录验证结果：训练能否正常启动、checkpoint 路径是否正确

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

### 9. 推理脚本数据范式对齐

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**目的**：确保推理脚本与新建数据集的字段范式一致，并使用 Python modelscope API 下载数据集。

当新建的数据集中除了 `prompt` 和 `image`（或 `video`）之外，还包含其他非文本类数据字段（如 `edit_image`、`input_image`、`mask`、`depth` 等）时，必须更新推理脚本以使用数据集中的实际数据作为推理输入。

**为什么需要做这一步**：训练数据集定义了模型的输入输出范式。如果推理脚本仍使用硬编码参数或简化输入，会导致推理脚本无法真实反映模型的训练-推理一致性，也方便后续用户复现和测试。

#### 9a. 判断是否需要更新

检查 Step 4 中创建的 `metadata.csv`，列出除文本字段外的所有列：

```bash
# 查看数据集 metadata.csv 的所有列
head -1 {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv
```

- 如果只有 `prompt` + `image`（或 `video`），**跳过本步骤**，直接进入 Step 10
- 如果包含额外的非文本数据字段（如 `edit_image`、`mask_image`、`depth_image`、`input_audio` 等），**继续执行以下步骤**

#### 9b. 添加 modelscope 数据集下载

在推理脚本中添加 `dataset_snapshot_download`：

```python
from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="{series}/{ModelName}/*"
)
```

#### 9c. 修改推理脚本参数

**⚠️ 验证脚本的 prompt 必须使用训练集的第一条数据（metadata.csv 中的第一行）。** 验证的目的是确认训练后的 LoRA 是否学到了训练集中的内容，因此 prompt 和所有输入必须来自训练集本身，而不是随意编造或从其他地方取。

**如何从训练集取第一条数据：**

1. **打开 `metadata.csv`**，找到第一行数据（除去表头）。该文件的编码通常支持 pandas 读取，第一行对应的数据编号一般为 `001`。

2. **读取 `prompt` 列的值**，这就是验证脚本中 `prompt` 变量的内容。该列在 `metadata.csv` 中通常名为 `prompt`。

3. **读取第一条数据对应的所有输入文件路径**（`image`、`edit_image`、`video` 等），从 `dataset_base_path` 下的相对路径加载。

4. **直接 hardcode 到验证脚本中**：

```python
from PIL import Image

dataset_base_path = "data/diffsynth_example_dataset/{series}/{ModelName}"

# ===== prompt 来自 metadata.csv 第一条数据 =====
# 打开 metadata.csv 查看第一行的 prompt 列内容，原样填入
prompt = "此处填写 metadata.csv 第一行 prompt 列的实际内容"

# ===== 输入文件来自第一条数据（001）对应的路径 =====
# 根据 metadata.csv 第一行中各字段对应的文件路径，从 dataset_base_path 加载
edit_image = Image.open(f"{dataset_base_path}/images/001_edit.jpg")  # 路径按 metadata.csv 第一行实际值填写
```

**常见错误：**
- **不要**随意编造 prompt，必须从 `metadata.csv` 第一行读取
- **不要**使用推理示例脚本中的 prompt（那是示例数据，不是训练集数据）
- **不要**使用多条数据混合，只取第一条

完整步骤：

1. **硬编码数据集路径和输入数据**：

   ```python
   from PIL import Image

   dataset_base_path = "data/diffsynth_example_dataset/{series}/{ModelName}"
   ```

2. **加载第一条数据（001）的对应字段**：

   ```python
   # 以图像编辑为例：只需加载 edit_image 作为额外输入
   edit_image = Image.open(f"{dataset_base_path}/images/001_edit.jpg")
   prompt = "此处填写 metadata.csv 第一行 prompt 列的实际内容"  # ⚠️ 必须从 metadata.csv 第一行读取
   ```

   常见数据类型加载方式：
   - **图像**：`Image.open(f"{dataset_base_path}/images/001.jpg")`
   - **视频**：`VideoData(f"{dataset_base_path}/videos/001.mp4", height=H, width=W)`
   - **音频**：`from diffsynth.utils.data.audio import read_audio` → `read_audio(f"{dataset_base_path}/audio/001.wav", start_time=0, duration=duration)`

3. **传入推理调用**：

   ```python
   result = pipe(
       prompt=prompt,
       edit_image=edit_image,
   )
   ```

**修改原则**：
- 默认使用第一条数据（编号 `001`），hardcode 到代码中
- 推理脚本的输入参数必须与 `metadata.csv` 中的非文本字段一一对应
- 保留原始推理脚本的核心逻辑（Pipeline 初始化、输出保存等），仅替换输入数据来源
- 可以参考同 series 已有验证脚本（`examples/{series}/model_training/validate_lora/`）中类似字段的使用方式
- **prompt 必须来自 `metadata.csv` 第一行的 prompt 列**，不可编造

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

## 常见问题排查

| 问题 | 排查方向 |
|------|---------|
| `input_latents` 为 None | `data["image"]` 加载是否正确、metadata.csv 路径是否正确、`main_data_operator` 是否配置 |
| Loss 为 NaN | 学习率过高（降 10 倍）、VAE 编码含 NaN、dtype 不匹配（确保 `bfloat16`） |
| 显存不足 | 按优先级逐步尝试：① 确认 `--use_gradient_checkpointing` 已添加 → ② 追加 `--use_gradient_checkpointing_offload` → ③ 减小 `--max_pixels` 或 `--lora_rank` → ④ 最后才尝试 deepspeed zero3 配置。**不要一上来就试 deepspeed**，纯 accelerate + gradient checkpointing 通常已足够 |
| 训练/推理输出不一致 | `InputXXXEmbedder` 的 `training` 分支、`split_pipeline_units`、数据预处理一致性 |
| 推理脚本无法加载数据集字段 | 确认 `metadata.csv` 列名与 Pipeline Unit `input_params` 一致、数据文件路径相对路径正确 |

### 10. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有训练步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/pipeline-training-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/pipeline-training-report.md" \
  "packages/{model-name}/.sisyphus/user_report/pipeline-training-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 2. 检查 debug 测试脚本是否存在于 DiffSynth-Studio 仓库中
for script in \
  "examples/{series}/model_training/lora/{ModelName}-test.sh" \
  "examples/{series}/model_training/lora/{ModelName}.sh" \
  "examples/{series}/model_training/full/{ModelName}.sh"; do
  if [ -f "$script" ]; then
    echo "OK: $script 已存在"
  else
    echo "WARNING: 训练脚本缺失: $script"
  fi
done

# 3. 检查执行日志和验证输出
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pipeline_training"
for output in \
  "$EXEC_LOG_DIR/outputs/lora_test.log" \
  "$EXEC_LOG_DIR/outputs/lora_train.log" \
  "$EXEC_LOG_DIR/outputs/lora_result.jpg" \
  "$EXEC_LOG_DIR/outputs/full_train.log"; do
  if [ -f "$output" ]; then
    echo "OK: $output 已保存"
  else
    echo "WARNING: 输出文件缺失: $output"
  fi
done
```

如有缺失，立即补充。

**向用户展示验证图像**：

所有检查通过后，向用户展示训练验证结果：

- 验证图像：`${EXEC_LOG_DIR}/outputs/lora_result.jpg`（LoRA 训练后的推理输出）
- 训练日志：`${EXEC_LOG_DIR}/outputs/lora_train.log`
- 等待用户确认训练效果后，继续后续步骤

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-training-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pipeline_training/`
- **输出目录**: `outputs/` - 命令执行日志、验证图像
- **脚本目录**: `scripts/` - 保存的训练脚本、验证脚本
- **检查点目录**: `checkpoints/` - 训练产出的 checkpoint

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/pipeline-training-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Pipeline Training 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pipeline-training |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：**目标**（做什么）、**执行内容**（怎么做）、**产出物**（交付什么）、**参考**（去哪里找更多细节）。

> 详细执行规则和约束请参见 SKILL.md 对应 Step 章节。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取训练模块接入所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、接入类型、训练相关信息
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

**参考**：SKILL.md Step 0 · 蓝图报告「基本信息」表格

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构、manifest.json 和 latest 软链接。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_pipeline_training/` 目录及子目录（scripts/、outputs/、checkpoints/）
- 创建 `manifest.json` 记录执行元信息
- 创建 `latest` 软链接指向当前执行日志目录

**产出物**：执行日志目录

**参考**：SKILL.md Step 1 · `references/execution-traceability.md`（执行日志初始化模板）

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示训练模块接入规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/pipeline-training-plan.md`
- 向用户展示训练架构设计、数据集方案、Debug 测试方案
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/pipeline-training-plan.md`

**参考**：SKILL.md Step 2

---

### Step 3: 分析训练需求

**目标**：确定训练模块的架构需求。

**执行内容**：
- 分析 Loss 类型（FlowMatchSFTLoss / FlowMatchSFTAudioVideoLoss）
- 确定数据集类型（图像/视频/多模态）
- 确认 InputEmbedder 是否需要新增训练分支

**产出物**：训练需求确认

**参考**：SKILL.md Step 3 · `references/split-training.md`（训练架构分析）

---

### Step 4: 准备数据集

**目标**：准备训练数据集。

**执行内容**：
- 下载 diffsynth_example_dataset
- 确认数据集格式和字段映射
- 构建 metadata.csv 和目录结构

**产出物**：数据集

**参考**：SKILL.md Step 4 · `references/dataset-guidelines.md`（数据集构建规范）

---

### Step 5: 创建训练代码（Python）

**目标**：创建训练模块和训练 Python 代码。

**执行内容**：
- 编写 InputEmbedder 的 `pipe.scheduler.training` 训练分支
- 创建 TrainingModule 类（{SeriesName}TrainingModule）
- 创建 train.py 训练脚本
- 创建推理验证脚本

**产出物**：
- `examples/{series}/model_training/train.py`
- InputEmbedder 训练分支修改

**参考**：SKILL.md Step 5 · `references/training-code-templates.md` · `references/training-scripts.md`

---

### Step 6: 创建训练脚本（.sh）

**目标**：创建 .sh 训练脚本。

**执行内容**：
- 创建 LoRA 训练脚本 `lora/{ModelName}.sh`
- 创建全量训练脚本 `full/{ModelName}.sh`
- 创建 `accelerate_config_zero3.yaml`（如需要）

**产出物**：
- `examples/{series}/model_training/lora/{ModelName}.sh`
- `examples/{series}/model_training/full/{ModelName}.sh`
- `examples/{series}/model_training/full/accelerate_config_zero3.yaml`

**参考**：SKILL.md Step 6 · `references/training-scripts.md`（训练脚本模板）

---

### Step 7: Debug 测试

**目标**：降参训练验证代码可运行。

**执行内容**：
- 使用 num_epochs=1, dataset_repeat=1 运行 LoRA 训练
- 验证代码无报错
- 清理 debug 脚本

**产出物**：Debug 测试通过确认

**参考**：SKILL.md Step 7

---

### Step 8: 训练验证

**目标**：完整训练 + 推理测试验证训练效果。

**执行内容**：
- LoRA 完整训练（5 epochs）+ 推理验证
- 全量训练冒烟测试（仅验证启动）

**产出物**：训练验证结果

**参考**：SKILL.md Step 8

---

### Step 9: 推理脚本数据范式对齐

**目标**：确保推理脚本与新建数据集的字段范式一致，并支持数据集自动下载。

**执行内容**：
- 检查 `metadata.csv` 是否包含除 `prompt` + `image`（或 `video`）之外的额外非文本字段
- 如有额外字段（如 `edit_image`），修改推理脚本使用数据集中的实际数据作为输入
- 在推理脚本中添加 `modelscope download` 数据集下载逻辑
- 如无额外字段，跳过本步骤

**产出物**：更新后的推理脚本

**参考**：SKILL.md Step 9

---

### Step 10: 最终验证

**目标**：确认所有报告文件和 debug/训练输出完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查 debug 测试脚本和训练脚本是否存在于 DiffSynth-Studio 仓库中
- 检查执行日志目录中的输出图像和日志
- 如有缺失，立即补充

**产出物**：验证通过确认

**参考**：SKILL.md Step 10

---

## 训练架构设计

- **TrainingModule 类名**: {SeriesName}TrainingModule
- **Loss 函数**: {FlowMatchSFTLoss / FlowMatchSFTAudioVideoLoss}
- **数据集类型**: {图像/视频/多模态}
- **LoRA 训练**: rank={N}, target_modules=[...]
- **InputEmbedder 训练分支**: 是否需要新增

## 验证方案
- **Debug 测试**: num_epochs=1, dataset_repeat=1，验证代码可运行
- **训练验证**: LoRA 推理测试，验证训练效果
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pipeline-training-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-10 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 分析训练需求 | Step 3 |
| 4 | 准备数据集 | Step 4 |
| 5 | 创建训练代码（Python） | Step 5 |
| 6 | 创建训练脚本（.sh） | Step 6 |
| 7 | Debug 测试 | Step 7 |
| 8 | 训练验证 | Step 8 |
| 9 | 推理脚本数据范式对齐 | Step 9 |
| 10 | 最终验证 | Step 10 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pipeline-training-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败 / ⬜ 跳过
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

训练验证完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/pipeline-training-report.md << OUTER_EOF
## ✅ 训练接入与验证完成

执行日志: \`packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pipeline_training/\`

### 📁 生成的文件

- 训练模块: `examples/{series}/model_training/train.py`
- LoRA 训练: `examples/{series}/model_training/lora/{ModelName}.sh`
- 全量训练: `examples/{series}/model_training/full/{ModelName}.sh`
- 验证脚本: `examples/{series}/model_training/validate_lora/{ModelName}.py`、`validate_full/{ModelName}.py`

### 📊 训练验证结果

| 阶段 | 状态 | 日志 | 验证图像 |
|------|------|------|---------|
| Debug 测试 | ✅ / ❌ | `lora_test.log` | - |
| LoRA 完整训练（5 epochs） | ✅ / ❌ | `lora_train.log` | `outputs/lora_result.jpg` |
| 全量训练（Smoke Test） | ✅ / ⏭️ 跳过 / ❌ | `full_train.log` | 路径验证通过 / 失败 |

### 📥 Checkpoint 位置

- LoRA: `./models/train/{ModelName}_lora/`
- 全量: `./models/train/{ModelName}_full/`

请检查验证图像质量，确认训练结果是否符合预期。
OUTER_EOF
```
