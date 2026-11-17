---
name: diffsynth-pipeline
description: >
  Create inference pipelines for an integrated model in DiffSynth-Studio.
  Use this skill after model code is integrated, or when, during a model/feature integration, the
  user wants to add a new pipeline (like Text-to-Video, Image-to-Audio, etc.), add inference
  examples, or run end-to-end consistency tests between the original library's example scripts and
  the DiffSynth pipeline. This skill handles ONE feature at a time — invoke it separately for each
  feature you want to integrate.
  Note: Training functionality has been split into the `diffsynth-pipeline-training` skill.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — bug fixes or refactoring of existing pipelines, performance
  tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: Pipeline 接入

为接入的模型创建推理 Pipeline 和推理示例脚本。**每次调用处理一个功能**，按从简单到难的顺序依次调用本 skill。

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `target_path`: `packages/{model-name}/{target-library}/`（从蓝图报告获取）
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**目标库路径**、**本次要接入的功能信息**和**Conda 环境名称**从蓝图报告中获取：
- 目标库路径：蓝图报告「基本信息」表格
- 功能信息：蓝图报告「Pipeline 功能规划」章节
- **Conda 环境名称**：蓝图报告「基本信息」表格中的 `Conda 环境名称` 字段（如 `ace-step-diffsynth`）。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。**

## Pipeline 设计原则

### 总则

- **每次调用处理一个功能。** 不依赖蓝图中的完整功能列表，每次只接入一个功能。按从简单到难的顺序，逐次调用本 skill。
- **Pipeline 文件按系列组织，不按功能拆分。** 一个模型系列对应一个 Pipeline 文件（如 `z_image.py` 包含 `ZImagePipeline`），所有功能通过 `__call__` 的不同参数和 `units` 链的不同组合来实现。
- **执行留痕，过程可追溯。** 所有测试脚本先保存再执行，所有命令输出保存到日志，关键检查点明确标记。遵循 [execution-traceability.md](../diffsynth-integrator/references/execution-traceability.md) 规范。
- **前提条件**：本 skill 假设 `diffsynth-model-code` 已完成所有组件的接入。

### 原则 1：每个 Unit 和 model_fn 必须有目标库参考

**⚠️ 严格原则：每个 Unit 和 model_fn 必须有目标库中的对应参考，输出必须一致。**

Pipeline 的每一个操作都不能凭空编写，必须在目标库的推理流程中找到对应的操作。

| 组件 | 约束 | 原因 |
|------|------|------|
| **每个 PipelineUnit 的 `process` 方法** | 必须在目标库中找到对应的推理操作，输出与目标库一致 | 每个 Unit 对应原库推理流程中的一个具体步骤 |
| **model_fn 函数** | 必须在目标库中找到 DiT/模型的前向调用代码，输出与目标库一致 | model_fn 是目标库模型 forward 的直接映射 |
| **`__call__` 的整体流程** | 必须与目标库的完整推理流程在逻辑上等价 | Pipeline 是原库推理流程的结构化重组 |
| **NoiseInitializer** | 唯一例外：只需保证相同 seed/shape 下与原库噪声一致 | 噪声生成是框架行为，不对应目标库的模型逻辑 |

**执行要求**：
1. **编写每个 Unit 之前**：先在目标库的推理脚本/代码中找到对应的操作。记录代码位置、输入输出形状、中间变量。
2. **编写每个 Unit 之后**：验证该 Unit 的输出与目标库对应操作的输出完全一致（数值相同或 `allclose(atol=1e-5)`）。
3. **禁止凭空编写**：❌ "我觉得这里应该做这样的变换" → 没有目标库参考。✅ "目标库在 `inference.py:123` 对 latent 做了 `latents * scale + shift`，我在 Unit 里做同样的操作" → 有明确参考。

**为什么这条原则是必须的**：Pipeline 的目标是与原库输出一模一样，不是"看起来差不多"。如果每个 Unit 都有对应参考且输出一致，最终 Pipeline 的输出自然一致。

**version_upgrade 下的适用范围**：本原则约束的是**本次为新变体新增或修改**的 Unit 和 model_fn 分支。已有的、正在服务其他已验证变体的 Unit 不在范围内——不要以"重新验证一致性"为名改写它们的逻辑（见「原则 5」）。

### 原则 2：Pipeline 是编排者，不是模型实现者

**Pipeline 应该做的事**：
- ✅ 调用条件编码器（TextEncoder、AudioEncoder 等）准备条件
- ✅ 初始化噪声 latent
- ✅ 运行去噪循环（调用 DiT + Scheduler）
- ✅ 调用 VAE 解码输出
- ✅ 处理多模态输入输出的组合逻辑
- ✅ 管理 VRAM（offload、cpu 缓存等）

**Pipeline 不应该做的事**：
- ❌ 实现模型的前向传播逻辑（这是模型文件的事）
- ❌ 包含注意力机制、归一化层等模型内部结构
- ❌ 直接操作张量的数学运算（除了简单的初始化/后处理）
- ❌ 定义模型类（模型类应该在 `diffsynth/models/` 中）

### 原则 3：数据链必须通过 Unit 实现

**除以下操作外，所有数据链上的处理必须设计为 PipelineUnit：**

| 允许在 `__call__` 中直接执行的操作 | 说明 |
|----------------------------------|------|
| `scheduler.set_timesteps(...)` | 调度器 timestep 计算 |
| `self.unit_runner(unit, ...)` | Unit 链执行 |
| `self.cfg_guided_model_fn(...)` | 去噪循环（CFG + model_fn） |
| `self.scheduler.step(...)` | Scheduler 步进 |
| `self.vae.decode(...)` / 解码器调用 | 最终解码输出 |
| `self.load_models_to_device(...)` | VRAM 管理 |
| 三字典的初始化和返回值组装 | 数据准备和最终返回 |

**禁止在 `__call__` 中直接编写数据处理逻辑。** 以下操作都**必须**设计为独立的 Unit：

| 必须在 Unit 中完成的操作 | 对应 Unit 类型 |
|----------------------|--------------|
| 文本/音频/图像编码 | PromptEmbedder / ConditionEmbedder |
| 尺寸计算/校验 | ShapeChecker |
| 噪声生成 | NoiseInitializer |
| 条件张量拼接/投影 | ConditionEmbedder |
| 输入图像/视频/音频的预处理+编码（训练用） | InputImageEmbedder / InputVideoEmbedder / InputAudioEmbedder |
| 图像编辑、局部重采样等功能输入 | EditImageEmbedder / InpaintEmbedder / LayerInputImageEmbedder 等功能专属 Embedder |
| 局部编辑 mask 计算 | RetakeEmbedder |
| 语言模型生成 | LMGenerator |
| 其他任何中间数据处理 | 自定义 Unit |

**判断标准**：如果一个操作对数据做了变换（张量计算、编码、拼接、投影等），它就应该是一个 Unit。`__call__` 只负责串联：初始化三字典 → 跑 Unit 链 → 去噪循环 → 解码返回。

### 原则 4：最小化新增代码，最大化复用

在 `__init__`、`from_pretrained`、`__call__` 三个函数中，**除非必要，不要创建任何之前不存在的操作**。

| 场景 | 正确做法 | 错误做法 |
|------|---------|---------|
| 数据处理/张量变换 | 放到 PipelineUnit 中 | 在 `__call__` 中直接写张量操作 |
| 模型加载逻辑 | 复用 `download_and_load_models` + `fetch_model` | 自己写新的加载函数 |
| 尺寸校验 | 复用 `check_resize_height_width` | 自己写新的校验逻辑 |
| VRAM 管理 | 复用 `check_vram_management_state` | 自己写显存检查 |
| Scheduler 特异逻辑 | 放到 `flow_match.py` 的 template 方法中 | 在 `__call__` 中直接改 scheduler 内部状态 |
| 条件编码 | 放到对应的 PipelineUnit 中 | 在 `__call__` 或 `__init__` 中直接调用 encoder |

**判断标准**：在编写代码前，先问自己——这个操作在已有的 Pipeline（如 `z_image.py`、`flux2_image.py`）或 `BasePipeline` 中是否存在类似实现？如果存在，直接复用；如果不存在，考虑是否应该做成 Unit，而不是塞进 `__init__`/`from_pretrained`/`__call__`。

### 原则 5：version_upgrade 以复用为先，不为数值对齐重写 Pipeline

`version_upgrade` 的 Pipeline 文件已经存在，且正在为同系列的其他变体服务——那些变体已经验证通过。新版本只是该系列的一个新变体，接入方式是在已有 Pipeline 上做**最小增量**，不是重写。

| ✅ 允许 | ❌ 禁止 |
|--------|--------|
| 为新变体新增 `__call__` 参数（带默认值） | 为对齐 E2E 数值而改写已有 Unit 的 `process` 逻辑 |
| 为新变体新增专属 Unit | 为对齐数值而改写 `model_fn` 的已有分支 |
| 在 `model_fn` 中新增参数以透传新变体的条件 | 为对齐数值而改动 scheduler template 的已有公式 |
| 复用已有的 Unit / scheduler / denoise loop | 为对齐数值而改动 denoise loop 结构 |
| — | 改变已有变体的参数默认值或行为 |

**⚠️ 严禁为了强求 E2E 数值完全一致而对 Pipeline 做大幅修改。** 已验证的变体和新变体共用同一个 Pipeline 文件，大幅修改会把已经跑通的变体一起拖入未知状态，代价远大于收益。

因此 version_upgrade 的验收标准与 new_series 不同：**只要求生成结果符合预期，不要求与原库数值一致**。数值对比照常运行并记录，但不作为通过/失败门槛。判定口径见 Step 6 与 [references/e2e-test.md](references/e2e-test.md)。

### 不可违背的设计原则

| 原则 | 说明 |
|------|------|
| **Batch Size 恒为 1** | 整个框架中 batch size 永远为 1，不暴露 batch_size 参数，不编写多 batch 推理逻辑 |
| **正向/负向分离 Forward** | 永远不要将正向（positive）和负向（negative）的 forward 放到一个 batch 中运行。正向和负向分别通过 `inputs_posi` / `inputs_nega` 独立传递，各自单独调用模型 forward |
| **Denoising Loop 最小修改** | 模板中的 denoising loop（`cfg_guided_model_fn` + `scheduler.step` 循环）是框架的核心逻辑，除非有充分理由（如目标库推理流程明确要求不同的采样策略），否则不得修改其结构和执行顺序。新增功能应通过添加 Unit 或扩展 `model_fn` 参数来实现，而不是改动循环本身 |
| **Loop 内数据操作优先放 model_fn** | denoising loop 内部某些特殊数据变换无法设计为 Unit（因为依赖 timestep 或循环中间状态）时，优先将逻辑放入 `model_fn` 中处理。只有在 `model_fn` 也不适合的情况下（如涉及 scheduler step 前后需要插入的操作），才考虑直接在 `__call__` 的 loop 内编写 |

## DiffSynth Pipeline 架构

Pipeline 架构的完整权威定义见 [pipeline-template.md](references/pipeline-template.md)。以下是核心要点：

| 部分 | 位置 | 作用 |
|------|------|------|
| Pipeline 文件 | `diffsynth/pipelines/{series}.py` | 定义 Pipeline 类（`__init__` + `from_pretrained` + `__call__` + PipelineUnit 链 + model_fn） |
| 推理示例 | `examples/{series}/model_inference/{feature}.py` | 每个功能一个脚本，展示基本用法 |
| 低显存推理 | `examples/{series}/model_inference_low_vram/{feature}.py` | 启用 VRAM 管理的推理脚本（由 `diffsynth-pipeline-lowvram` 生成） |
| 训练脚本 | `examples/{series}/model_training/train.py` | 整个系列共用一个训练脚本 |
| 训练配置 | `examples/{series}/model_training/full/` + `lora/` | 每个模型一个 .sh 脚本 |
| 验证脚本 | `examples/{series}/model_training/validate_full/` + `validate_lora/` | 每个模型一个 .py 脚本 |

**PipelineUnit 三种模式**（详见共享参考）：
- **普通模式**：从 `inputs_shared` 读取参数，结果写回 `inputs_shared`
- **seperate_cfg**：分别处理正向/负向提示词（`input_params_posi`/`input_params_nega`）
- **take_over**：接管整个函数

### 文件命名规范

**所有脚本命名必须严格遵循以下规范，不得自行发明格式。**

| 文件类型 | 目录 | 命名格式 | 示例 |
|---------|------|---------|------|
| 推理示例 | `model_inference/` | `{ModelName}-{Feature}.py` | `LTX-2-T2AV-TwoStage.py` |
| 低显存推理 | `model_inference_low_vram/` | `{ModelName}-{Feature}.py` | `LTX-2-T2AV-TwoStage.py` |
| 训练共用脚本 | `model_training/` | `train.py` | 固定文件名 |
| 全量训练配置 | `model_training/full/` | `{ModelName}-{Feature}.sh` | `LTX-2-T2AV-splited.sh` |
| LoRA 训练配置 | `model_training/lora/` | `{ModelName}-{Feature}.sh` | `LTX-2-T2AV-IC-LoRA-splited.sh` |
| 全量验证 | `model_training/validate_full/` | `{ModelName}-{Feature}.py` | `LTX-2-T2AV.py` |
| LoRA 验证 | `model_training/validate_lora/` | `{ModelName}-{Feature}.py` | `LTX-2-T2AV.py` |

**命名规则详解**：

1. **`{ModelName}`**：模型名称，使用 PascalCase + 版本号，连字符分隔。如 `LTX-2`、`LTX-2.3`、`Z-Image-Turbo`、`Qwen-Image`
2. **`{Feature}`**：功能描述，使用缩写，连字符分隔：
   - 任务类型：`T2I`（文本生成图像）、`T2V`（文本生成视频）、`I2V`（图像生成视频）、`T2AV`（文本生成音视频）、`I2AV`（图像生成音视频）、`I2I`（图像编辑）、`I2L`（图像生成图像）
   - 变体后缀：`TwoStage`、`OneStage`、`DistilledPipeline`、`8steps`、`splited`
   - 控制/LoRA 类型：`IC-LoRA-Union-Control`、`IC-LoRA-Detailer`、`Camera-Control-Static`
3. **所有脚本的 `{ModelName}-{Feature}` 前缀必须保持一致**，推理脚本、训练脚本、验证脚本使用相同的功能标识
4. **不要使用下划线、不要使用空格、不要使用驼峰式功能名**

**`__call__` 固定流程**：
1. 设置 scheduler timesteps
2. 准备三字典：`inputs_posi`、`inputs_nega`、`inputs_shared`
3. 运行 Unit 链：`self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)`
4. Denoise loop：`cfg_guided_model_fn` + `scheduler.step`
5. VAE 解码 → 返回输出

**典型 Pipeline 流程**（简化示意，详见 [pipeline-template.md](references/pipeline-template.md) 中的完整模板）：

```python
@torch.no_grad()
def __call__(self, prompt, negative_prompt="", cfg_scale=1.0, ...):
    # 1. Scheduler
    self.scheduler.set_timesteps(num_inference_steps, ...)
    
    # 2. 三字典输入
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {"cfg_scale": cfg_scale, ...所有共享参数...}
    
    # 3. Unit 链执行
    for unit in self.units:
        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
    
    # 4. Denoise loop（通过 cfg_guided_model_fn 自动处理 CFG）
    self.load_models_to_device(self.in_iteration_models)
    models = {name: getattr(self, name) for name in self.in_iteration_models}
    for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
        noise_pred = self.cfg_guided_model_fn(
            self.model_fn, cfg_scale,
            inputs_shared, inputs_posi, inputs_nega,
            **models, timestep=timestep, progress_id=progress_id
        )
        inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
    
    # 5. VAE 解码
    self.load_models_to_device(['vae'])
    image = self.vae.decode(inputs_shared["latents"])
    return self.vae_output_to_image(image)
```

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pipeline-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pipeline-report.md << EOF

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
| 基本信息表中的 `Conda 环境名称` | 测试环境 |
| 基本信息表中的 `目标库路径` | 代码参考、对比基准 |
| Pipeline 功能规划表 | 本次要接入的功能名称、原库推理脚本路径、输入输出、依赖组件 |
| Scheduler 信息 | 调度策略类型、template 名称、特异参数 |
| 接入类型 | new_series / version_upgrade，决定是否新建 Pipeline 文件 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
export EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_pipeline_{feature}"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "diffsynth-pipeline",
  "model_name": "{model-name}",
  "feature": "{feature}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "pipeline_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "user_checks": []
}
EOF

ln -sfn ${EXEC_LOG_DIR} packages/{model-name}/.sisyphus/execution-logs/latest
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在执行具体接入工作前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/pipeline-{feature}-plan.md`。** 基于蓝图信息和本次要接入的功能，明确 Pipeline 架构、Unit 链设计、model_fn 设计、Scheduler 方案、推理示例脚本和 E2E 验证方案。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始编写代码。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 3. 确定本次要接入的功能

> 📖 开始前：重读本步骤描述，确认流程与报告路径

从蓝图报告或用户指定中获取本次要接入的功能信息：
- 功能名称（如 Text-to-Image、Image-to-Video）
- 对应的原库推理脚本路径
- 该功能的输入、输出、特殊处理
- 依赖的模型组件

**每次只接入一个功能。**

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4. 创建或更新 Pipeline 文件

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**首次调用**：创建新的 Pipeline 文件。
**后续调用**：在已有 Pipeline 文件中追加该功能的 Unit 和参数。

文件位置：`{diffsynth_root}/diffsynth/pipelines/{series}.py`

**接入类型为 `version_upgrade` 时**：Pipeline 文件已存在，且正在服务同系列其他已验证通过的变体。只做最小增量（新增带默认值的参数、新增专属 Unit、`model_fn` 新增透传参数），**不得预先为了后续 E2E 数值对齐而改写已有 Unit、`model_fn`、scheduler template 或 denoise loop**，详见「原则 5」。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4.1 分析目标库推理流程 → 输出设计蓝图

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取目标库的推理脚本**，将其逐行拆解为以下四个逻辑阶段：

```
目标库推理脚本 → 拆解为：
├── 阶段 1: 条件准备（文本编码、音频编码、图像编码等）
├── 阶段 2: 噪声初始化（生成初始 latent）
├── 阶段 3: 去噪循环（scheduler + DiT forward + scheduler step）
└── 阶段 4: 解码输出（VAE decode、后处理）
```

**拆解方法**：逐行阅读目标库推理脚本，标注每行代码属于哪个阶段，记录涉及的中间变量。

**分析完成后，必须将设计蓝图保存到文件，并等待用户确认后才能继续。**

**保存路径**：`packages/{model-name}/.sisyphus/reports/blueprint-{feature}.md`

**输出格式**：

```markdown
## Pipeline 设计蓝图（等待用户确认）

**功能**: {功能名称，如 Text-to-Image}
**目标库脚本**: {文件路径}

### 推理流程拆解

| 步骤 | 目标库代码位置 | 对应阶段 | 中间变量 | 建议处理 |
|------|--------------|---------|---------|---------|
| 1 | inference.py:L10-15 | 条件准备 | prompt_embeds | Unit: PromptEmbedder |
| 2 | inference.py:L18-20 | 噪声初始化 | noise | Unit: NoiseInitializer |
| 3 | inference.py:L23-30 | 去噪循环 | noise_pred | model_fn + denoise loop |
| 4 | inference.py:L33-35 | 解码输出 | image | __call__ 直接调用 VAE |

### Unit 链设计

**4 个核心 Unit（顺序固定，不可颠倒）** + 功能专属 Unit：

| Unit 顺序 | Unit 名称 | 模式 | 输入参数 | 输出参数 | 目标库对应操作 |
|-----------|----------|------|---------|---------|--------------|
| 1 | {Series}Unit_ShapeChecker | 普通 | height, width | height, width | inference.py:L5 尺寸对齐 |
| 2 | {Series}Unit_PromptEmbedder | seperate | prompt | prompt_embeds, attn_mask | inference.py:L10-15 |
| 3 | {Series}Unit_NoiseInitializer | 普通 | seed, height, width | noise | inference.py:L18-20 |
| 4 | {Series}Unit_InputImageEmbedder | 普通 | input_image, noise | latents, input_latents | inference.py:L21-25 |
| 5+ | {Series}Unit_{Feature} | ... | ... | ... | 功能专属 Unit |

### model_fn 设计

- **参数**: dit, latents, timestep, prompt_embeds
- **timestep 处理**: {是否需要 /1000. 缩放}
- **目标库对应调用**: {文件路径 + 行号}

### Pipeline 参数

- **prompt**: str（功能核心输入）
- **negative_prompt**: str（默认 ""）
- **cfg_scale**: float（默认 1.0）
- **height/width**: int（默认 1024）
- **seed**: int（默认 None）
- **num_inference_steps**: int（默认 30）
- **{其他功能特有参数}**

### 调度器

- **类型**: {Flow Matching / DDPM / ...}
- **Template 名称**: "{Series}"
- **原库 scheduler 类名及路径**: {如 `diffusers.schedulers.scheduling_flow_matching.FlowMatchEulerDiscreteScheduler`}
- **特异参数**: {如有，列出名称、类型、默认值、用途}
- **特殊方案**: {如有，列出方案名称、special_case 值、固定 sigmas、step 数}
- **训练 timestep 权重策略**: {如有，描述；否则写"无"}
```

**⚠️ 必须等待用户确认设计蓝图后才能继续。** 用户可能调整 Unit 顺序、参数名或调度器策略。

```bash
# 将设计蓝图保存到文件
cat > packages/{model-name}/.sisyphus/reports/blueprint-{feature}.md << 'EOF'
# 设计蓝图内容（即上方输出的完整 Markdown）
EOF

# 向用户展示蓝图文件路径并等待确认
echo "设计蓝图已保存至 packages/{model-name}/.sisyphus/reports/blueprint-{feature}.md"
echo "请确认设计蓝图是否合理，输入 '是' 继续，输入 '否' 我将根据反馈调整。"
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4.2 根据设计蓝图编写 Pipeline 初稿

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**用户确认设计蓝图后**，按蓝图中的规划编写 Pipeline 代码。

**首次调用**时，创建完整的 Pipeline 文件结构：

```python
class {SeriesName}Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("{Series}")
        # 组件类型提示
        self.text_encoder: TextEncoderClass = None
        self.dit: DiTClass = None
        self.vae: VAEClass = None

        self.in_iteration_models = ("dit",)
        self.units = [
            # 根据设计蓝图中的 Unit 链顺序填写
        ]
        self.model_fn = model_fn_{series}
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = {SeriesName}Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("{series}_text_encoder")
        pipe.dit = model_pool.fetch_model("{series}_dit")
        pipe.vae = model_pool.fetch_model("{series}_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 30,
        progress_bar_cmd = tqdm,
        # 该功能特有的参数...
    ):
        # 1. Scheduler
        self.scheduler.set_timesteps(num_inference_steps, ...)

        # 2. 三字典
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            # 该功能特有的参数...
        }

        # 3. Unit 链
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # 5. 解码
        self.load_models_to_device(['vae'])
        output = self.vae.decode(inputs_shared["latents"])
        return self.vae_output_to_image(output)
```

**Pipeline 类编写完成。Unit 类和 model_fn 函数在后续步骤中编写（Step 2.4），放在 Pipeline 类之后。**

**关键约束**：
- model_fn 的参数必须与 denoise loop 中 `cfg_guided_model_fn` 传入的参数匹配
- `__call__` 中除允许在 `__call__` 中直接执行的操作外，**不得有任何数据处理逻辑**（详见「Pipeline 设计原则」章节）

**后续调用时**（追加新功能），只需：
1. 在 `__call__` 中添加该功能的新参数
2. 在 `inputs_shared` 中添加该功能的参数
3. 在 `units` 链中添加该功能对应的 Unit
4. 在 `model_fn` 中添加该功能需要的参数（如有）
5. 确保不改变已有功能的参数默认值和逻辑

详细模板和 PipelineUnit 设计原则见 [references/pipeline-template.md](references/pipeline-template.md)。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4.3 Scheduler 接入

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**Scheduler 是 Pipeline 与模型之间的桥梁。** 不同模型系列可能使用不同的 timestep 调度策略。接入规则详见 [references/scheduler-integration.md](references/scheduler-integration.md)。

**决策流程**：

1. **从蓝图报告读取调度策略信息**：蓝图必须在「Pipeline 功能规划」中说明目标模型的调度类型和特异参数
2. **判断接入方式**：
   - Flow Matching + 公式与已有 template 相同 → 复用已有 template
   - Flow Matching + 公式不同 → 在 `FlowMatchScheduler` 中新增 `set_timesteps_{series}` 方法
   - 有 distilled/turbo 等特殊方案 → 在 `set_timesteps_{series}` 中用 `special_case` 区分，返回固定 sigmas
   - 非 Flow Matching → 在 `FlowMatchScheduler` 中新增方法，按该采样器逻辑实现
3. **模型特异参数透传**：`__call__` 中暴露为有默认值的参数 → 透传到 `set_timesteps(**kwargs)`

**LTX-2 示例（Flow Matching + 模型特异参数 + 特殊方案）**：

```python
# __init__
self.scheduler = FlowMatchScheduler("LTX-2")

# __call__ 中暴露特异参数和特殊方案并透传
def __call__(
    self,
    num_inference_steps: int = 30,
    dynamic_shift_len: int = 4096,  # LTX-2 特异参数
    terminal: float = 0.1,           # LTX-2 特异参数
    special_case: str = None,        # 特殊方案：None / "stage2" / "distilled_stage1"
):
    self.scheduler.set_timesteps(
        num_inference_steps,
        denoising_strength=denoising_strength,
        dynamic_shift_len=dynamic_shift_len,
        terminal=terminal,
        special_case=special_case,
    )
```

**何时需要修改 `flow_match.py`**：

| 情况 | 动作 |
|------|------|
| 新系列使用 Flow Matching，公式与已有 template 相同 | 无需修改，Pipeline 初始化时传已有 template 名称 |
| 新系列使用 Flow Matching，公式不同 | 在 `flow_match.py` 新增 `set_timesteps_{series}` 方法 + 注册到 `__init__` 字典 |
| 新系列使用 Flow Matching，有特异参数 | 新增方法的签名中包含这些参数，Pipeline `__call__` 中透传 |
| 新系列有 distilled/turbo 等特殊方案 | 在 `set_timesteps_{series}` 中用 `special_case` 参数区分，返回固定 sigmas |
| 新系列不使用 Flow Matching | 在 `flow_match.py` 新增方法，按该采样器逻辑实现 |

**验证：运行目标库的完整 Scheduler 测试并对比 timesteps。**

1. 运行目标库的完整推理代码，**输出 `timesteps`（或 sigmas）的完整列表**
2. 运行新接入的 scheduler 代码，**输出同样参数下 `self.scheduler.timesteps` 的完整列表**
3. 逐项对比两者，**如果任何值不一致，必须向目标库对齐**

```python
# 目标库输出 timesteps
print("Target timesteps:", scheduler.timesteps)

# 新接入的 scheduler 输出
print("DiffSynth timesteps:", self.scheduler.timesteps)
```

这是 Scheduler 接入的**最终验证步骤**。timesteps 是去噪循环的核心骨架，任何偏差都会导致最终输出不一致。只有两者完全匹配后，才能进入 Step 2.4。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4.4 设计 Unit 链并逐个编写

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**Unit 设计的核心方法**：将目标库推理脚本中的每个操作步骤，转化为一个 PipelineUnit。

1. **列出目标库推理脚本中的所有操作步骤**（跳过 import 和变量初始化）
2. **将每个操作步骤映射为一个 Unit**：
   - 该步骤的输入 → `input_params`，输出 → `output_params`
   - 需要加载哪些模型到 GPU → `onload_model_names`
   - 是否区分正向/负向 → `seperate_cfg`
3. **按目标库的执行顺序排列 Unit**

**Unit 链结构：4 个必须 + N 个可选。** 每个 Pipeline 必须包含以下 4 个 Unit，按固定顺序排列，不可跳过或颠倒。根据功能需要，可在此基础上追加额外的功能专属 Unit：

```
ShapeChecker → PromptEmbedder → NoiseInitializer → InputEmbedder → [功能专属 Unit...]
```

**helper 函数和常量就近写在 Unit 内部，不要提升到模块级。** 某个 Unit 需要的辅助函数写成该 Unit 的 `@staticmethod`，需要的常量写成该 Unit 的类属性（`_UPPER_CASE` 命名）。

**跨 Unit 复用时用 `类名.成员` 直接访问**，不要为此把代码搬到模块级：

```python
image = FluxImageUnit_InsertAnything.postprocess(image, insert_anything_post)
crop_left = MiniMaxMusic3Unit_ChunkDenoiser.crop_left_latent * self.latent_hop_length
```

参考 `MiniMaxH3Unit_PackedSequenceBuilder`（`_INTERP`、`_T_GROUP`、`_SEQ_ALIGN` 等类属性 + 内部 `@staticmethod`）和 `LingBotVideoUnit_PromptEmbedder` 的 `_RUNTIME_KEYS`。

#### 4.4.1 ShapeChecker —— 尺寸校验

**职责**：检查并调整输入尺寸，确保满足模型下采样整除要求。

**核心约束**：
- **必须复用 `pipe.check_resize_height_width(height, width)`**，不得自行编写尺寸校验逻辑
- **必须放在 Unit 链最前面**，后续 Unit 依赖调整后的尺寸
- 输入输出参数均为 `("height", "width")`，视频模型额外处理 `num_frames`

```python
class {SeriesName}Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}
```

#### 4.4.2 PromptEmbedder —— 文本条件编码

**职责**：将文本 prompt 编码为 text embedding。

**核心约束**：
- 必须使用 `seperate_cfg=True`，分别处理正向和负向提示词
- **必须加载 text_encoder**（`onload_model_names=("text_encoder",)`）
- **从目标库映射**：找到目标库的 `encode_prompt` 方法，将其逻辑移植到 `encode_prompt` 辅助方法中

```python
class {SeriesName}Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe, prompt):
        # TODO: 从目标库的 encode_prompt 移植编码逻辑
        text_inputs = pipe.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))
        return prompt_embeds

    def process(self, pipe, prompt, negative_prompt=None):
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt_embeds = self.encode_prompt(pipe, prompt)
            return {"prompt_embeds": prompt_embeds}
        return {}
```

#### 4.4.3 NoiseInitializer —— 初始噪声生成

**职责**：生成去噪循环的初始噪声 latent。

**核心约束**：
- **必须调用 `pipe.generate_noise(...)`**，不得自行使用 `torch.randn`
- **输出 key 必须为 `"noise"`**，这是框架约定的固定名称
- 唯一例外：不需要目标库对应操作，只需保证相同 seed/shape 下与原库噪声一致

```python
class {SeriesName}Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: {SeriesName}Pipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise(
            (1, 16, height // 8, width // 8),
            seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise}
```

#### 4.4.4 InputEmbedder —— 输入模态编码（训练/推理共用）

**职责**：InputEmbedder **仅用于**推理阶段传递噪声和训练阶段计算输入图的 VAE 编码。**不要在其中混入任何特殊功能的输入处理**。任何特殊功能（图像编辑、局部重采样、retake、上下文等）必须编写对应的专属 Embedder（如 `EditImageEmbedder`、`InpaintEmbedder`、`RetakeEmbedder`），每个功能一个独立的 Embedder Unit。

**核心约束**：
- **推理模式**（`pipe.scheduler.training=False`）：对输入模态做 VAE 编码后调用 `pipe.scheduler.add_noise(...)` 加噪，返回 `{"latents": latents}`。**不返回 `input_latents`**。
- **训练模式**（`pipe.scheduler.training=True`）：返回 `{"latents": noise, "input_latents": vae_encode(数据)}`。
- **输入为 None 时**：返回 `{"latents": noise, "input_latents": None}`。
- `input_params` 必须包含 `"noise"`，`output_params` 必须包含 `"latents"` 和 `"input_latents"`。

**注意**：推理模式的逻辑在本 skill 阶段验证；训练模式（`pipe.scheduler.training=True` 分支）的正确性由后续 `diffsynth-pipeline-training` skill 负责验证和修正。本步骤只需按标准模板正确编写训练分支代码，无需在本阶段验证其输出。

**参考优先级**：
1. 优先参考同系列已有的同模态 Embedder
2. 其次参考其他采用相同 VAE 或相似功能的 Pipeline

```python
class {SeriesName}Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}

        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}
```

**功能专属 Embedder**：图像编辑、局部重采样、分层编辑等功能，**必须编写独立的 Embedder Unit**（如 `EditImageEmbedder`、`InpaintEmbedder`、`LayerInputImageEmbedder`），不要将编辑功能的输入混入 `InputImageEmbedder`。参考 Qwen-Image Pipeline 的设计模式。

#### 4.4.5 功能专属 Unit

在 4 个必须 Unit 之后，根据本次要接入的功能，追加对应的功能专属 Unit。例如：
- 图像编辑 → `EditImageEmbedder`
- 局部重采样 → `InpaintEmbedder`
- 分层编辑 → `LayerInputImageEmbedder`
- 上下文图像 → `ContextImageEmbedder`

每个功能专属 Embedder 处理该功能特有的输入数据预处理和编码逻辑。

#### 4.4.6 Unit 类型参考

| Unit 类型 | 对应目标库操作 | 输入 | 输出 | 用途 |
|-----------|--------------|------|------|------|
| **ShapeChecker** | 输入尺寸检查/调整 | height, width | 调整后的 height, width | 必须 |
| **PromptEmbedder** | 文本编码 | prompt | prompt_embeds | 必须 |
| **NoiseInitializer** | 初始噪声生成 | seed, height, width | noise | 必须 |
| **InputImageEmbedder** | 图像预处理 + VAE 编码 | input_image, noise | latents | 必须 |
| **ConditionEmbedder** | 条件编码（音频、timbre 等） | 条件输入 | condition_embeds | 可选 |
| **EditImageEmbedder** | 编辑图像编码 | edit_image | edit_latents | 编辑功能 |
| **InpaintEmbedder** | 局部重采样 mask 处理 | inpaint_mask | inpaint_mask | 编辑功能 |
| **LayerInputImageEmbedder** | 分层图像 latent 编码 | layer_input_image | layer_input_latents | 编辑功能 |
| **ContextImageEmbedder** | 上下文图像 latent 编码 | context_image | context_latents | 编辑功能 |
| **RetakeEmbedder** | 局部编辑 mask + latent | retake_input, regions | mask, latents | 编辑功能 |
| **LMGenerator** | 语言模型生成 | prompt, params | generated_text | 可选 |

**示例：从目标库推理脚本映射 Unit 链**

```python
# 目标库推理脚本
height, width = round_up_to_multiple(height, 16), round_up_to_multiple(width, 16)  # → ShapeChecker
prompt_embeds = text_encoder(tokenizer(prompt))                                      # → PromptEmbedder
noise = torch.randn((1, 16, height//8, width//8), generator=generator)              # → NoiseInitializer
for t in scheduler.timesteps:                                                        # → model_fn + denoise loop
    noise_pred = dit(latents, t, prompt_embeds)
image = vae.decode(latents)                                                          # → __call__ 直接调用
```

**逐个编写 Unit 并检查**：

每个 Unit 编写完成后确认：
- [ ] `input_params` / `output_params` 与数据流上下游匹配
- [ ] `process` 方法检查参数是否为 None（返回 `{}` 跳过）
- [ ] 每个 Unit（除 NoiseInitializer 外）都有明确的目标库对应操作
- [ ] Unit 的输出与目标库对应操作的输出一致（数值相同或 `allclose(atol=1e-5)`）

> `version_upgrade` 时，后两条只针对**本次新增或修改**的 Unit；已有 Unit 不重新验证、不改写。

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 4.5 设计 model_fn

> 📖 开始前：重读本步骤描述，确认流程与报告路径

`model_fn` 是 denoise loop 中调用 DiT 前向传播的函数。它应该是目标库中模型 `forward` 调用的直接映射。

**代码位置**：放在所有 Unit 类之后，文件末尾。

**代码文件的完整顺序**：
1. imports
2. Pipeline 类（Step 2.2 完成）
3. PipelineUnit 类（Step 2.4 编写）
4. model_fn 函数（本步骤编写，放在所有 Unit 之后，文件末尾）

**模板**：

```python
def model_fn_{series}(
    dit,
    latents,
    timestep,
    prompt_embeds,
    # 其他条件参数（根据功能需求添加）...
    **kwargs
):
    # timestep 处理（根据调度器类型，常见为 /1000.）
    timestep = timestep / 1000.

    # 调用 DiT（与目标库相同的调用方式）
    output = dit(
        latents,
        timestep,
        prompt_embeds,
        # 其他参数...
    )
    return output
```

**关键约束**：
- model_fn 的参数必须与 denoise loop 中 `cfg_guided_model_fn` 传入的参数完全匹配
- 不要将 model_fn 放在 Unit 类之前
- 每个新增功能如果需要在 DiT forward 中传递额外参数，必须在 model_fn 中添加对应参数

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 5. 创建推理示例脚本

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**详细步骤、vram_config 级别、低显存版本生成方法**见 [references/inference-scripts.md](references/inference-scripts.md)。

文件位置：`{diffsynth_root}/examples/{series}/model_inference/{feature}.py`

**脚本结构**（通常 15-25 行）：

```python
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="..."),
        # ... 所有需要的组件
    ],
)
# 调用 Pipeline，传入该功能需要的参数
output = pipe(prompt="...", seed=42)
output.save("output.jpg")
```

**注意**：模型路径通过 `ModelConfig` 指定，模型首次运行时会自动从 ModelScope 下载并创建软链接。

**同时创建低显存版本**：
文件位置：`{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py`

低显存版本在标准版本基础上添加三处修改（详见 [inference-scripts.md](references/inference-scripts.md)）：
1. 定义 `vram_config` 字典（offload_dtype、offload_device 等 8 个键）
2. 在每个组件 `ModelConfig` 中传入 `**vram_config`
3. 在 `from_pretrained` 中传入 `vram_limit`

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 6. 一致性验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**⚠️ 开始测试前，必须先完整阅读以下文档：**
- **[references/e2e-test.md](references/e2e-test.md)** — E2E 测试的完整流程、对比方法、失败判定标准和日志规范

**不得在未阅读的情况下自行简化或偏离流程。**

**输出类型适配**：根据 Pipeline 的输出类型，采用不同的数值化方式：

| 输出类型 | 数值化方式 | 对比形状 |
|---------|-----------|---------|
| **图像** | `np.array(PIL.Image)` | `(H, W, C)` uint8 |
| **音频** | `np.ndarray` | `(channels, samples)` 或 `(samples,)` float32 |
| **视频帧** | `np.stack([np.array(f) for f in frames])` | `(T, H, W, C)` uint8 |
| **多模态** | 每种模态分别数值化 | 分别对比 |

**验收标准按接入类型分流**（接入类型在 Step 0 从蓝图读取）：

| 接入类型 | 结构门槛（跑通 / 类型 / shape / dtype / 无 NaN-Inf） | 数值一致（`allclose(atol=1e-5)`） | 最终验收 |
|---------|:--:|:--:|:--|
| `new_series` | 必须通过 | **必须通过**，不一致则逐 Unit 定位修复 | 用户确认输出质量 |
| `version_upgrade` | 必须通过 | **不作为门槛**：照常运行、记录 `max_diff` / MSE，不判定失败 | **用户确认生成结果符合预期即通过** |

**version_upgrade 的数值对比照跑，但不因数值差异判失败，也不得为缩小差异而修改已有 Pipeline 逻辑**（见「原则 5」）。

#### 6a. 阶段一：确保 Pipeline 能跑通

**⚠️ 不要在 Pipeline 还没跑通时就急于对比输出。** 如果脚本中途报错，先修复错误，确保完整运行后再进入对比阶段。

```bash
cd {diffsynth_root}
python examples/{series}/model_inference/{feature}.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/step_05a_diffsynth_run.log

# 检查退出码
if [ $? -eq 0 ]; then
    echo "✅ DiffSynth Pipeline 跑通，无报错"
else
    echo "❌ DiffSynth Pipeline 运行失败，先修复错误再进入对比阶段"
    tail -50 ${EXEC_LOG_DIR}/outputs/step_05a_diffsynth_run.log
    # 修复后重新运行，直到跑通为止
fi
```

#### 6b. 阶段二：E2E 端到端对比

确认 Pipeline 能跑通后，按照 [references/e2e-test.md](references/e2e-test.md) 的完整流程执行：

1. **在原库中运行**，保存 golden reference（最终输出 + 推理参数 JSON）
2. **在 DiffSynth 中运行 Pipeline**，使用相同参数
3. **验证输出一致性**：检查输出类型、shape、dtype 一致，无 NaN/Inf
4. **保存执行留痕**：测试脚本、命令日志、对比结果

**通过标准**（来自 e2e-test.md）：

| 检查项 | 判定 |
|--------|------|
| 输出存在且非空 | ✅ 通过 |
| 输出类型一致 | ✅ 通过 |
| 输出形状一致 | ✅ 通过 |
| dtype 一致 | ✅ 通过 |
| 无 NaN/Inf | ✅ 通过 |

**数值判定按接入类型分流**：
- `new_series`：**数值必须完全一致**（数值相同或 `allclose(atol=1e-5)`），不一致则逐 Unit 定位修复
- `version_upgrade`：只运行对比并记录 `max_diff` / MSE，**不作为通过/失败门槛**

两种类型在结构门槛通过后，都由用户检查输出文件确认质量。

**读取 E2E 结果**：
```bash
E2E_STATUS=$(python -c "import json; print(json.load(open('${EXEC_LOG_DIR}/checkpoints/e2e_comparison_result.json'))['status'])")
```

#### 6c. 根据 E2E 结果处理

**E2E 通过 → 输出报告 → 用户确认 → 完成**

⚠️ **E2E 通过不代表自动完成。必须向用户输出报告并获得明确确认后，才能标记该功能接入完成。**

```bash
if [ "$E2E_STATUS" = "PASSED" ]; then
    echo "E2E 测试通过，生成接入报告供用户确认"

    # Code Review 检查清单
    cat > ${EXEC_LOG_DIR}/checkpoints/code_review_checklist.md << EOF
# Pipeline Code Review Checklist
- [ ] Pipeline 类继承 BasePipeline
- [ ] __init__ 中正确设置 scheduler、units 链、model_fn
- [ ] from_pretrained 正确加载所有组件
- [ ] __call__ 遵循固定流程：scheduler → 三字典 → units → denoise loop → VAE decode
- [ ] 每个 Unit（除 NoiseInitializer）都有对应的目标库代码参考记录
- [ ] 每个 Unit 的输出与目标库对应操作的输出一致
- [ ] model_fn 有对应的目标库模型前向调用参考
- [ ] 每个 Unit 的 input_params/output_params 定义正确
- [ ] Unit 的 process 方法检查参数是否为 None
- [ ] 推理示例脚本可直接运行
- [ ] （version_upgrade）未为对齐数值而改写已有 Unit / model_fn / scheduler template / denoise loop
- [ ] （version_upgrade）已有变体的参数默认值与行为未改变
EOF

    # Agent 逐项检查上述清单
    echo "Code review 完成"

    # ⚠️ 向用户输出报告并等待确认
    cat << REPORT
## ✅ E2E 测试通过 — 请确认

**功能**: {feature}
**接入类型**: {new_series / version_upgrade}
**MSE**: {mse}
**Max Diff**: {max_diff}

### Code Review 结果
\`cat ${EXEC_LOG_DIR}/checkpoints/code_review_checklist.md\`

### 查看输出文件
- 原库: \`${EXEC_LOG_DIR}/checkpoints/output_original.*\`
- DiffSynth: \`${EXEC_LOG_DIR}/checkpoints/output_diffsynth.*\`

请确认：E2E 测试结果和 Code Review 是否通过？
- 输入 **是** → 标记该功能接入完成
- 输入 **否** → 说明具体问题，我将修复后重新运行 E2E
REPORT

    # ⏸️ 等待用户明确回复
    # 如果用户指出问题 → 修复后重新运行 E2E 直到通过
    # 如果用户确认通过 → 标记完成
fi
```

**version_upgrade 的确认重点**：报告中需额外说明「数值差异仅供参考，不作为门槛；请重点确认生成结果是否符合预期」。用户确认结果符合预期即标记完成，**不要**提议为缩小 `max_diff` 而修改 Pipeline。

**E2E 失败 → 修复问题后重新运行**

E2E 失败意味着结构性问题（输出为空、类型/形状不匹配、数值异常等）。检查 Unit 对应关系、参数传递、数据类型等，修复后重新运行 E2E 测试，直到通过。

**version_upgrade 数值不一致 ≠ E2E 失败。** `max_diff` 超过 `atol` 时，正确动作是：

1. 确认结构门槛全部通过（跑通、类型、shape、dtype、无 NaN/Inf）
2. 确认组件级一致性已在 `diffsynth-model-code` 阶段验证通过（新变体的模型 forward 与原库一致）
3. 把 `max_diff` / MSE 与可能来源（原库版本差异、scheduler 特异参数、采样策略变化等）写进报告
4. 输出文件交用户确认生成质量

**❌ 禁止**：为了让 `max_diff` 降下来而改写已有 Unit、`model_fn`、scheduler template 或 denoise loop。

#### 6d. 用户检查清单

```bash
# 首次生成：创建文件头
cat > ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - diffsynth-pipeline ({feature})

执行时间: $(date -Iseconds)
模型: {model-name}

---
EOF

# 追加检查点
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - diffsynth-pipeline ({feature})

## [CHECK-001] 端到端输出一致性
- **严重程度**: critical
- **描述**: 原库和 DiffSynth 的最终输出是否一致
- **查看对比结果**: \`cat ${EXEC_LOG_DIR}/checkpoints/e2e_comparison_result.json\`
- **通过标准**（按接入类型分流，详见 e2e-test.md）:
  - \`new_series\`: 输出数值完全一致（数值相同或 \`allclose(atol=1e-5)\`）
  - \`version_upgrade\`: 结构门槛通过（跑通 / 类型 / shape / dtype / 无 NaN-Inf）+ 生成结果符合预期；\`max_diff\` 仅记录，不作为门槛
- **状态**: ⬜ 待确认

## [CHECK-002] Pipeline 初始化
- **严重程度**: critical
- **描述**: Pipeline 是否能正常初始化和加载模型
- **查看日志**: \`${EXEC_LOG_DIR}/outputs/step_05a_diffsynth_run.log\`
- **通过标准**: 无 ImportError、模型加载成功、无 CUDA OOM
- **状态**: ⬜ 待确认

## [CHECK-003] 推理示例脚本
- **严重程度**: info
- **描述**: 生成的推理示例脚本是否正确
- **查看文件**: \`{diffsynth_root}/examples/{series}/model_inference/{feature}.py\`
- **检查点**:
  - [ ] 脚本可直接运行
  - [ ] 输出路径正确
  - [ ] 参数设置合理
- **状态**: ⬜ 待确认

## [CHECK-004] 低显存版本（如创建）
- **严重程度**: info
- **描述**: 低显存版本是否正常工作
- **查看文件**: \`{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py\`
- **测试命令**: \`python examples/{series}/model_inference_low_vram/{feature}.py\`
- **状态**: ⬜ 待确认

## [CHECK-005] Unit 与目标库对应关系（关键）
- **严重程度**: critical
- **描述**: 每个 Unit 和 model_fn 是否都能在目标库中找到对应的操作参考
- **检查清单**:
  - [ ] 每个 Unit 的 `process` 方法都标注了对应的目标库代码位置
  - [ ] 每个 Unit 的输出与目标库对应操作的输出一致
  - [ ] model_fn 能在目标库中找到对应的模型前向调用代码
  - [ ] NoiseInitializer 除外（只需验证相同 seed 下噪声一致）
  - [ ] （version_upgrade）仅新增该变体所需的 Unit / 参数，未改写已有变体的逻辑
- **状态**: ⬜ 待确认

## [CHECK-006] 数据链 Unit 化审查（关键）
- **严重程度**: critical
- **描述**: 除去噪循环和 VAE 解码外，所有数据处理是否都设计为独立的 PipelineUnit
- **检查清单**:
  - [ ] `__call__` 中没有直接的数据处理逻辑（张量计算、编码、拼接、投影等）
  - [ ] `__call__` 仅包含：三字典初始化 → Unit 链执行 → 去噪循环 → 解码返回
  - [ ] 每个数据处理步骤（编码、尺寸计算、噪声生成、条件拼接等）都有对应的 Unit
  - [ ] 允许在 `__call__` 中的操作仅限：`scheduler.set_timesteps`、`unit_runner`、`cfg_guided_model_fn`、`scheduler.step`、`vae.decode`、`load_models_to_device`、三字典初始化和返回值组装
- **状态**: ⬜ 待确认
EOF
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

### 7. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在 Pipeline 接入完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/pipeline-{feature}-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/pipeline-{feature}-report.md" \
  "packages/{model-name}/.sisyphus/user_report/pipeline-{feature}-report.md"; do
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

> 📝 完成后：更新渐进式报告 → `skill_work_report/pipeline-report.md`

## 目录结构

本 skill 读取和写入以下路径：
- **蓝图报告**: `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- **中间结果**: `packages/{model-name}/.sisyphus/intermediate/e2e/` - 端到端测试输出
- **执行日志**: `packages/{model-name}/.sisyphus/execution-logs/$(date)_pipeline_{feature}/`

## 本次调用需要的信息

- **功能名称**：如 Text-to-Image、Image-to-Video
- **对应的原库推理脚本路径**：用于 E2E 测试
- **该功能的输入、输出、特殊处理**：用于设计 Unit 和 `__call__` 参数
- **依赖的模型组件**：用于 `from_pretrained` 中的 model_configs
- **是否是首次调用**：决定是创建新 Pipeline 文件还是更新已有文件

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的所有测试脚本（端到端测试、对比脚本）
- **输出目录**: `outputs/` - 命令执行日志
- **检查点目录**: `checkpoints/` - 输出文件、对比结果
- **用户检查清单**: `user-checks.md` - 需要人工确认的项目

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/pipeline-{feature}-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Pipeline ({feature}) 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-pipeline |
| 功能 | {feature} |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取 Pipeline 接入所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、接入类型、Pipeline 功能规划表、本次接入功能
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_pipeline_{feature}/` 目录及子目录
- 创建 `manifest.json` 和 `latest` 软链接

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示 Pipeline 接入规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/pipeline-{feature}-plan.md`
- 向用户展示 Pipeline 架构设计、Unit 链、对比验证方案
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/pipeline-{feature}-plan.md`

---

### Step 3: 确定本次接入的功能

**目标**：从蓝图报告中确认本次要接入的 Pipeline 功能。

**执行内容**：
- 从蓝图的 Pipeline 功能规划表中找到本次要接入的功能
- 如果是首次调用，确认系列名称、Pipeline 类名
- 如果是后续调用，确认已有 Pipeline 文件路径和需要追加的功能
- 如果是 version_upgrade，确认与已有版本的差异

**产出物**：确认后的功能清单

**注意事项**：
- 每次只接入一个功能，按从简单到难的顺序

---

### Step 4: 创建/更新 Pipeline 文件

**目标**：编写 Pipeline 类、Unit 链和 model_fn。

**执行内容**：
- 首次调用：创建新的 Pipeline 文件 `{diffsynth_root}/diffsynth/pipelines/{series}.py`
- 后续调用：在已有 Pipeline 文件中追加新功能
- 按顺序编写：Pipeline 类 → Unit 链 → model_fn
- Unit 链设计：ShapeChecker → PromptEmbedder → NoiseInitializer → InputEmbedder → 功能专属 Unit

**产出物**：
- Pipeline 文件

**注意事项**：
- Pipeline 按系列组织，不按功能拆分
- 每个 Unit 的 input_params/output_params 必须与上下游匹配
- `version_upgrade`：只做最小增量，不得为对齐 E2E 数值而改写已有 Unit / model_fn / scheduler / denoise loop

---

### Step 5: 创建推理示例脚本

**目标**：为接入的功能创建可运行的推理示例脚本。

**执行内容**：
- 创建 `{diffsynth_root}/examples/{series}/model_inference/{feature}.py`
- 脚本包含：Pipeline 初始化、参数设置、调用执行、结果保存

**产出物**：
- 推理示例脚本

---

### Step 6: 一致性验证

**目标**：验证 DiffSynth Pipeline 输出可用且符合预期；`new_series` 还要求与原库数值一致。

**执行内容**：
- 使用相同输入分别运行目标库和 DiffSynth Pipeline
- 结构门槛（两种接入类型都必须通过）：跑通、输出类型 / shape / dtype 一致、无 NaN/Inf
- 数值对比按接入类型分流：
  - `new_series`：要求完全一致（数值相同或 `allclose(atol=1e-5)`），不一致则逐 Unit 定位修复
  - `version_upgrade`：只运行并记录 `max_diff` / MSE，不作为通过/失败门槛
- `version_upgrade`：确认未为对齐数值而改写已有 Unit / model_fn / scheduler / denoise loop

**产出物**：
- 一致性验证报告（含 `max_diff` / MSE）
- 对比结果（通过/失败）

**注意事项**：
- `version_upgrade` 下数值不一致不算失败，以用户确认生成结果符合预期为准

---

### Step 7: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## Pipeline 架构设计

- **系列名称**: {series}
- **Pipeline 类名**: {SeriesName}Pipeline
- **Unit 链**: ShapeChecker → PromptEmbedder → NoiseInitializer → {功能专属 Unit}
- **model_fn 设计**: {CFG 策略、scheduler 类型}
- **推理示例输入**: {prompt 格式}
- **对比方案**: 原库 golden reference 运行方式

## 验证方案
- **单元测试**: Pipeline 初始化成功、Unit 链正确
- **E2E 测试**: 生成图像与原库对比
- **一致性测试**: `new_series` 要求最终输出数值完全一致；`version_upgrade` 只要求结构门槛通过 + 生成结果符合预期（`max_diff` / MSE 仅记录，不作为门槛）
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/pipeline-{feature}-report.md`（每个功能一个报告）

**步骤划分**（与上方「工作流程」章节的 Step 0-7 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 确定本次接入的功能 | Step 3 |
| 4 | 创建/更新 Pipeline 文件 | Step 4 |
| 5 | 创建推理示例脚本 | Step 5 |
| 6 | 一致性验证 | Step 6 |
| 7 | 最终验证 | Step 7 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/pipeline-{feature}-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 生成的代码文件

- **Pipeline 文件**: `{diffsynth_root}/diffsynth/pipelines/{series}.py`
- **推理示例**: `{diffsynth_root}/examples/{series}/model_inference/{feature}.py`
- **低显存版本**: `{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py`

### 向用户报告

Pipeline 接入完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/pipeline-{feature}-report.md << 'OUTER_EOF'
## ✅ Pipeline 接入完成 ({feature})

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 📋 人工检查清单

请查看并确认以下检查项：
`cat packages/{model-name}/.sisyphus/execution-logs/latest/user-checks.md`

关键检查项：
1. [CHECK-001] 端到端输出一致性 - **必须确认**
2. [CHECK-002] Pipeline 初始化 - **必须确认**
3. [CHECK-003] 推理示例脚本
4. [CHECK-004] 低显存版本（如创建）
5. [CHECK-005] Unit 与目标库对应关系 - **关键，必须确认**
6. [CHECK-006] 数据链 Unit 化审查 - **关键，必须确认**

### 📁 生成的文件

- Pipeline: `diffsynth/pipelines/{series}.py`
- 推理示例: `examples/{series}/model_inference/{feature}.py`
- 低显存版本: `examples/{series}/model_inference_low_vram/{feature}.py`
- E2E 测试脚本: `scripts/test_e2e_original.py`, `scripts/test_e2e_diffsynth.py`, `scripts/compare_outputs.py`
- 对比结果: `checkpoints/e2e_comparison_result.json`

### 🚀 下一步

确认所有检查项后：
- 如果还有其他功能 → 再次调用 `diffsynth-pipeline`
- 如果所有功能完成 → `diffsynth-testing` - 全量测试
- 如果需要训练 → 调用 `diffsynth-pipeline-training` - 训练模块接入
OUTER_EOF
```
