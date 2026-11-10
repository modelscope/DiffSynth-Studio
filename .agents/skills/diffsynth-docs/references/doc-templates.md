# DiffSynth-Studio 文档模板结构

所有 Model_Details 文档遵循以下固定章节顺序和模板。生成新模型文档时，按实际接入的功能在对应位置填入内容。

---

## 中文文档结构（zh）

```markdown
# {Model-Name}

{一句话模型描述}

## 安装

{标准安装命令，固定内容}

## 快速开始

{带显存管理 vram_config 的基础推理脚本完整代码}

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
{每行对应 model_inference/ 下的一个推理脚本，从脚本中提取 model_id 作为第一列}
{同一模型 ID 有多个脚本时，格式为：[模型 ID: 变体名](模型链接)。变体名来自脚本文件名，去掉 .py 后缀和前缀重复的系列名（如 Ernie-Image-T2I.py → T2I，Ernie-Image-T2I-PE.py → T2I-PE）}
{某功能没有训练脚本时对应列填 -}
{代码链接使用完整 GitHub URL：`https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/...` }

{可选：特殊训练脚本列表}
{可选：DeepSpeed ZeRO 3 训练说明}

## 模型推理

模型通过 `{PipelineClass}.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`{PipelineClass}` 推理的输入参数包括：

* `param1`: {说明}
* `param2`: {说明}
...

{可选：显存不足提示 + 显存管理链接}
{可选：多卡并行加速章节}
{可选：其他推理相关章节}

## 模型训练

{model_name} 系列模型统一通过 `examples/{model}/model_training/train.py` 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* {Model-Name} 专有参数
    * `--tokenizer_path`: {说明}
    * ...

{样例数据集下载命令}

{训练脚本参考链接}
{可选：训练技巧}
```

---

## English Documentation Structure (en)

```markdown
# {Model-Name}

{One-sentence model description}

## Installation

{Standard installation commands, fixed content}

## Quick Start

{Complete code example with VRAM management vram_config from the base inference script}

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
{Each row corresponds to one inference script under model_inference/, model_id extracted from the script}
{When the same model ID has multiple scripts, format is: [Model ID: Variant](link). Variant name comes from script filename, removing .py extension and repeated series prefix (e.g. Ernie-Image-T2I.py → T2I, Ernie-Image-T2I-PE.py → T2I-PE)}
{Use - for columns where training scripts are not available}
{Code links use full GitHub URL: `https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/...` }

{Optional: Special training scripts list}
{Optional: DeepSpeed ZeRO 3 training notes}

## Model Inference

The model is loaded via `{PipelineClass}.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `{PipelineClass}` inference include:

* `param1`: {description}
* `param2`: {description}
...

{Optional: VRAM warning + memory management link}
{Optional: Multi-GPU acceleration section}
{Optional: Other inference-related sections}

## Model Training

Models in the {model_name} series are trained uniformly via `examples/{model}/model_training/train.py`. The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
        * `--extra_inputs`: Additional input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models to load in FP8 format, currently only supported for models whose parameters are not updated by gradients.
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. `bitsandbytes_nf4`), and `exclude_modules` optionally lists layers kept in full precision.
    * Basic Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether unused parameters exist in DDP training.
        * `--weight_decay`: Weight decay magnitude.
        * `--task`: Training task, defaults to `sft`.
    * Output Configuration
        * `--output_path`: Path to save the model.
        * `--remove_prefix_in_ckpt`: Remove prefix in the model's state dict.
        * `--save_steps`: Interval in training steps to save the model.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path to LoRA checkpoint.
        * `--preset_lora_path`: Path to preset LoRA checkpoint for LoRA differential training.
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image/video. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image/video. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for video (video generation models only).
* {Model-Name} Specific Parameters
    * `--tokenizer_path`: {description}
    * ...

{Example dataset download command}

{Training script reference link}
{Optional: Training tips}
```

---

## 固定内容模板（中英文共用）

### 安装章节 — 中文

```markdown
## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。
```

### Installation Section — English

```markdown
## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).
```

### 样例数据集 — 中文

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

### Example Dataset — English

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

### 训练结尾 — 中文

```markdown
我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
```

### Training Footer — English

```markdown
We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
```

### GitHub 链接前缀（固定）— 仅用于模型文档

```
https://github.com/modelscope/DiffSynth-Studio/blob/main/
```

**链接使用规则 — 务必区分两种文件：**

| 文件类型 | 代码链接格式 | 示例 |
|---------|-------------|------|
| **模型文档**（`docs/zh/Model_Details/*.md`、`docs/en/Model_Details/*.md`） | 完整 GitHub URL | `[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_inference/xxx.py)` |
| **README**（`README.md`、`README_zh.md`） | 本地相对路径 | `[code](/examples/ernie_image/model_inference/xxx.py)` |

### 快速开始代码上方说明 — 中文

```markdown
运行以下代码可以快速加载 [{model_id}]({model_url}) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 {peak_vram}G 显存即可运行。
```

**内部指令（不输出到文档）**：`{peak_vram}` 从 `packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md` 的「最高显存占用」字段获取，向上取整到整数。如 lowvram 测试未执行或无数据，默认 24G。只提取数值，不输出此说明。

### Quick Start Preamble — English

```markdown
Running the following code will load the [{model_id}]({model_url}) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of {peak_vram}GB VRAM.
```

**Internal directive (do not output to doc)**: `{peak_vram}` is obtained from the "最高显存占用" (peak VRAM usage) field in `packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md`, rounded up to an integer. If lowvram testing was not executed or data is unavailable, default to 24G. Only extract the number, do not output this note.

---

## README 更新模板

根 README 需要改三处：**更新历史**（new_series 和 version_upgrade 都要）、**全局模型表**（new_series 和 version_upgrade 都要）、**基础框架分类列表**（仅 new_series）。

> **不要**生成 per-model 的 `<details>` 区块，也不要在 README 中重复快速开始代码。所有模型统一登记在文件末尾的全局表里。

### 模型类型判断

用于更新历史条目的措辞，以及基础框架分类列表的归类：

- **图像模型**：Pipeline 文件名含 `image`（如 `ernie_image`、`flux`、`qwen_image`、`z_image`、`joyai_image`），功能为 text-to-image、image editing、image-to-image 等
- **视频模型**：Pipeline 文件名含 `video`（如 `wan_video`、`ltx2_audio_video`、`step_video`），功能为 text-to-video、video editing、image-to-video 等
- **音频模型**：Pipeline 文件名含 `audio` / `music`（如 `ace_step`、`minimax_music3`），功能为 text-to-music、text-to-audio 等
- **音视频联合生成**归入视频类（上游 MiniMax-H3、LTX-2 均在视频分类下）
- **不确定时**：参考蓝图报告中的 Pipeline 功能规划表，或向用户确认

### 更新历史条目（Update History）

**在 "## Update History"（英文）或 "## 更新历史"（中文）章节中，在最新一条更新记录的上方插入一条新的更新条目。** 格式参考已有条目，使用日期加粗 + 简要描述 + 相关链接。

**README.md（英文）— 更新历史条目：**

- **new_series** 模板：

```markdown
- **{Month Day, Year}** {Model-Series-Name} open-sourced, welcome a new member to the {image/video/audio} model family! Support includes text-to-{image/video/music} generation, low VRAM inference, and training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/{Model-Series-Name}.md) and [example code](/examples/{model}/).
```

- **version_upgrade** 模板（根据实际新增功能调整描述）：

```markdown
- **{Month Day, Year}** We add support for {new feature/variant} in {Model-Series-Name}. For details, please refer to the [documentation](/docs/en/Model_Details/{Model-Series-Name}.md) and [example code](/examples/{model}/).
```

**README_zh.md（中文）— 更新历史条目：**

- **new_series** 模板：

```markdown
- **{Year}年{Month}月{Day}日** {Model-Series-Name} 开源，欢迎加入{图像/视频/音频}生成模型家族！支持文生{图/视频/音乐}推理、低显存推理和训练能力。详情请参考[文档](/docs/zh/Model_Details/{Model-Series-Name}.md)和[示例代码](/examples/{model}/)。
```

- **version_upgrade** 模板（根据实际新增功能调整描述）：

```markdown
- **{Year}年{Month}月{Day}日** 我们为 {Model-Series-Name} 新增了{新增功能/变体描述}。详情请参考[文档](/docs/zh/Model_Details/{Model-Series-Name}.md)和[示例代码](/examples/{model}/)。
```

### 全局模型表

位置：`README.md` 的 `## All Supported Models`、`README_zh.md` 的 `## 全部支持的模型`，均在文件末尾。

表格 **8 列**，比文档页的模型总览表在最前多一个「基础架构」列：

```markdown
| 基础架构 | 模型 ID | 推理 | 低显存推理 | 全量训练 | 全量训练后验证 | LoRA 训练 | LoRA 训练后验证 |
|-|-|-|-|-|-|-|-|
```

```markdown
| Architecture | Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
|-|-|-|-|-|-|-|-|
```

**分块与插入位置**：

- 表格按**基础架构**分块，同一架构的行必须连续，块内顺序与文档页模型总览表一致
- 架构块按接入时间**由新到旧**排列
- **架构已在表中** → 把新行插入该架构已有的连续块内
- **全新架构** → 在表头分隔行下方另起新块，置于**表格最前面**
- version_upgrade 同理：插入该架构已有的块内，**不要**追加到表格末尾

**行内容**：

1. 第 1 列填**基础架构名**，不是模型系列名。上游把 JoyAI-Echo 归 `LTX-2`、MOVA 与 LongCat-Video 归 `Wan`。同一架构多行重复填同一个值
2. 第 2 列起与该模型文档页 `## 模型总览` 的对应行一致，**链接指向的脚本必须相同**
3. 链接使用根相对路径 `/examples/{model}/...`
4. 不支持的功能填 `-`

### 基础框架分类列表（仅 new_series）

位置：`README.md` 的 `## Basic Framework` / `README_zh.md` 的 `## 基础框架`，在 "Model overview:" / "模型一览：" 之后，按类别分组。

类别：`Image generation` / `图像生成`、`Video generation` / `视频生成`、`Audio generation` / `音频生成`，各类别内**新的在前**。`Evaluation models` / `评测模型` 是扁平单行条目，不适用于普通模型。

> 该列表不追求覆盖全部模型——上游 MoVA 在全局表有行但不在此列表中。

英文条目：

```markdown
    - {Model-Series-Name}: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/{Model-Name}.html), [Example code](/examples/{model}/)
```

中文条目：

```markdown
    - {Model-Series-Name}：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/{Model-Name}.html)、[样例代码](/examples/{model}/)
```

差异点（容易写错）：

1. 文档链接是 **ReadTheDocs HTML URL**（`.html` 结尾），不是仓库内 `.md` 路径——与更新历史条目的链接形式不同
2. locale 段：英文 `/en/latest/`，中文 `/zh-cn/latest/`
3. 中文用全角冒号 `：` 和顿号 `、`；英文用 `:` 和 `,`
4. 缩进 4 个空格，嵌在类别项之下

**该章节顶部的精选快速开始表格不要动**："快速开始，体验热门和最新模型" / "Quick start: experience popular and the latest models" 下面那张 8 列表格是人工精选的子集（当前只有 MiniMax-H3、ACE-Step、Z-Image、Krea-2），并非每个模型都在内——最新的 MiniMax-Music3 就不在。是否收录由维护者决定，不自动添加。

### README 更新要点

1. **更新历史条目**：无论 `new_series` 还是 `version_upgrade`，都在 "## Update History" / "## 更新历史" 中插入新条目，位于最新一条记录的**上方**
2. **模型类型判断**：用于更新历史措辞和基础框架分类列表的归类，不决定表格位置
3. **全局表插入位置**：new_series 插到表格最前；version_upgrade 插入该系列已有的连续块内
4. **行一致性**：全局表的行与文档页模型总览表的行必须链接到相同的脚本
5. **不要求表头一致**：全局表英文列头是 `Validation After Full Training` / `Validation After LoRA Training`，文档页是 `Full Training Validation` / `LoRA Training Validation`；且全局表 8 列、文档页 7 列。一致性只针对行内容，不针对列头措辞和列数
6. **路径格式 — 区分文件类型**：
   - **README.md / README_zh.md** 中的代码链接：使用**根相对路径**（如 `/examples/{model}/model_inference/xxx.py`）
   - **模型文档**（`docs/zh/Model_Details/*.md`、`docs/en/Model_Details/*.md`）中的代码链接：使用**完整 GitHub URL**（如 `https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/{model}/model_inference/xxx.py`）

---

## 章节名称映射

中英文文档章节一一对应关系：

| 中文章节 | 英文对应标题 |
|---------|-------------|
| 安装 | Installation |
| 快速开始 | Quick Start |
| 模型总览 | Model Overview |
| 模型推理 | Model Inference |
| 模型训练 | Model Training |
| 模型 ID | Model ID |
| 推理 | Inference |
| 低显存推理 | Low VRAM Inference |
| 全量训练 | Full Training |
| 全量训练后验证 | Full Training Validation |
| LoRA 训练 | LoRA Training |
| LoRA 训练后验证 | LoRA Training Validation |
| 特殊训练脚本 | Special Training Scripts |
| 样例数据集 | Example Dataset |
| {Model-Name} 专有参数 | {Model-Name} Specific Parameters |
| 通用训练参数 | General Training Parameters |
| 模型下载 | Model Download |

---

## 关键规则

1. **模型总览表格**：**new_series 严格 7 列**（模型 ID / 推理 / 低显存推理 / 全量训练 / 全量训练后验证 / LoRA 训练 / LoRA 训练后验证），不增不减，不支持的功能填 `-`，**不要加"额外参数"列**。version_upgrade 时沿用该页已有的列数——上游 `FLUX.md`、`LTX-2.md`、`Wan.md` 是 8 列（第 2 列为「额外参数」）的历史写法，追加行时对齐已有表头即可。
2. **模型 ID 来源**：从每个推理脚本的 `ModelConfig(model_id=...)` 中提取，作为表格第一列。
3. **每行一个脚本**：`model_inference/` 下的每个推理脚本都应该在模型总览表格中占一行（包括 PE 变体等）。如果某脚本没有对应的训练脚本，对应列填 `-`。
3.1 **同一模型多脚本处理**：当同一模型 ID 对应多个推理脚本时，模型 ID 列格式为 `[模型 ID: 变体名](模型链接)`。变体名从脚本文件名提取：去掉 `.py` 后缀，去掉文件名前缀中重复的系列名部分。例如 `Ernie-Image-T2I.py` → `T2I`，`Ernie-Image-T2I-PE.py` → `T2I-PE`。参考 LTX-2.md 的表格格式（如 `LTX-2.3: OneStagePipeline-I2AV`）。
4. **快速开始代码**：使用**带显存管理 vram_config** 的基础推理脚本（如文生图），完整复制不简化或改写，包含所有 import、vram_config、from_pretrained、推理调用、输出保存。显存大小从 `packages/{model-name}/.sisyphus/skill_work_report/lowvram-report.md` 的「最高显存占用」字段获取，向上取整到整数；如 lowvram 测试未执行或无数据，默认写 24G。**只提取数值，不需要任何解释。**
5. **安装章节**：只有安装命令，**不要添加"模型下载"章节**。
6. **训练参数**：通用参数使用上方固定模板，只从代码提取模型专有参数。
7. **`{model}` 路径名**：从 `examples/{model}/model_inference/xxx.py` 推断，常见映射如 `Qwen-Image` → `qwen_image`，`Wan` → `wanvideo`，`FLUX` → `flux`。
