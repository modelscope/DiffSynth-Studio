# 训练数据集指南

**适用场景**：为训练模块准备数据集——下载、复用已有数据、或新增样例数据集。

## 数据集下载

所有样例数据集统一托管在 ModelScope 的 `DiffSynth-Studio/diffsynth_example_dataset` 仓库中。

**下载命令**：

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --local_dir ./data/diffsynth_example_dataset
```

**参数说明**：
- `--local_dir` 指定本地存放路径

**下载完成后数据集结构**：

```
data/diffsynth_example_dataset/
└── {series}/
    └── {dataset_name}/
        ├── metadata.csv          # 数据索引文件
        ├── images/               # 图像文件（图像任务）
        ├── videos/               # 视频文件（视频任务）
        └── ...                   # 其他数据文件
```

## 已有数据集选择指南

在为新模型接入训练流程前，先确认是否有可用的样例数据集：

| 任务类型 | 参考数据集 | 下载路径 | metadata 列 |
|---------|-----------|---------|------------|
| 基础图像生成（T2I） | `qwen_image/Qwen-Image` | `--include "qwen_image/Qwen-Image/*"` | `image`, `prompt` |
| 视频生成（T2V） | `wanvideo/Wan2.1-T2V-14B` | `--include "wanvideo/Wan2.1-T2V-14B/*"` | `video`, `prompt` |
| 音视频联合生成（T2AV） | `ltx2/LTX-2.3-T2AV-splited` | `--include "ltx2/LTX-2.3-T2AV-splited/*"` | `video`, `prompt`, `input_audio`, `frame_rate` |

**选择原则**：
- 如果新模型的任务类型与已有数据集相同（如都是基础图像生成），则**在 `data/diffsynth_example_dataset/{series}/` 下以训练脚本同名创建新文件夹**，将已有数据集的数据复制过来使用
- 如果推理 Pipeline 已完成 E2E 测试，优先使用 Pipeline 推理输出创建样例数据集（见下方「从 Pipeline 推理输出创建」章节）
- 如果任务类型完全新颖且无推理输出可用，需要新增数据集——请用户提供数据后再继续

## 复用数据集方案

当新模型与已有模型任务类型相同时，**不需要重新下载数据**，直接在样例数据集中新建对应路径即可。

**命名规则**：`data/diffsynth_example_dataset/{series}/{ModelName}/` 中的 `{ModelName}` 与训练脚本文件名（去掉 `.sh`）一一对应。例如：

| 训练脚本 | 数据集路径 |
|---------|-----------|
| `full/Qwen-Image.sh` | `qwen_image/Qwen-Image/` |
| `full/Qwen-Image-Edit.sh` | `qwen_image/Qwen-Image-Edit/` |
| `full/Wan2.1-T2V-14B.sh` | `wanvideo/Wan2.1-T2V-14B/` |

**操作流程**：

```bash
# 示例：为新模型 {ModelName}（基础图像生成任务）复用 Qwen-Image 数据集
# 1. 在样例数据集的 series 目录下，以 {ModelName} 为名创建新文件夹
mkdir -p {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}

# 2. 复制已有数据集内容
cp -r {diffsynth_root}/data/diffsynth_example_dataset/qwen_image/Qwen-Image/* \
       {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/

# 3. 确认 metadata.csv 和数据文件都已就位
ls {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
```

**训练脚本中的路径引用**：

```bash
# 数据集路径中的 {ModelName} 与本训练脚本文件名一致
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "{series}/{ModelName}/*" \
    --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/{series}/model_training/train.py \
    --dataset_base_path "data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    ...
```

**各任务类型可复用的源数据集**：

| 新模型任务 | 复用源数据集 | 复制内容 |
|-----------|-------------|---------|
| 基础 T2I | `qwen_image/Qwen-Image/` 或 `z_image/Z-Image/` | `metadata.csv` + 图像文件 |
| 图像编辑 | `qwen_image/Qwen-Image-Edit/` | `metadata.csv` + 图像文件 |
| T2V 视频 | `wanvideo/Wan2.1-T2V-14B/` | `metadata.csv` + `video*.mp4` |
| T2AV 音视频 | `ltx2/LTX-2.3-T2AV-splited/` | `metadata.csv` + `video.mp4` |

## 从 Pipeline 推理输出创建样例数据集

当没有可直接复用的样例数据集时，**优先使用已完成推理 Pipeline 的输出作为训练样例数据**。由于推理 Pipeline 已经完成 E2E 测试，推理生成的图像/视频及其对应的 prompt 可以直接用作训练数据，保证训练数据与模型能力的一致性。

### 定位推理输出

推理输出位于 Pipeline 执行日志目录：

```bash
# 查找最新的推理输出
PIPELINE_EXEC_LOG="packages/{model-name}/.sisyphus/execution-logs/latest/outputs/"
ls ${PIPELINE_EXEC_LOG}/

# 或搜索所有推理输出
find packages/{model-name}/.sisyphus/execution-logs/ -name "*.jpg" -o -name "*.png" -o -name "*.mp4" | head -20
```

### 获取推理 Prompt

从推理脚本或执行日志中提取使用的 prompt：

```bash
# 查看推理脚本中的 prompt
grep -r "prompt" packages/{model-name}/.sisyphus/execution-logs/*/scripts/*.py 2>/dev/null

# 查看执行日志中的 prompt 记录
grep -i "prompt" packages/{model-name}/.sisyphus/execution-logs/*/outputs/*.log 2>/dev/null
```

### 创建数据集

1. **创建目录结构**：
   ```bash
   mkdir -p {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/{images|videos}/
   ```

2. **复制推理输出**：
   ```bash
   # 图像任务
   n=0
   for img in ${PIPELINE_EXEC_LOG}/*.jpg ${PIPELINE_EXEC_LOG}/*.png; do
     [ -f "$img" ] && cp "$img" "{diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/images/$(printf '%03d' $((++n))).$(echo $img | rev | cut -d. -f1 | rev)"
   done

   # 视频任务
   n=0
   for vid in ${PIPELINE_EXEC_LOG}/*.mp4; do
     [ -f "$vid" ] && cp "$vid" "{diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/videos/$(printf '%03d' $((++n))).mp4"
   done
   ```

3. **编写 metadata.csv**：
   ```bash
   cd {diffsynth_root}/data/diffsynth_example_dataset/{series}/{ModelName}/
   # 根据实际推理输出数量和对应的 prompt 编写 metadata.csv
   # 列名必须与 Pipeline Unit input_params 一致
   ```

### 注意事项

- 推理输出数量通常较少（2-5 条），但用于训练验证已足够
- `metadata.csv` 中的 prompt 必须使用推理时实际使用的 prompt
- 数据文件命名使用编号格式（`001.jpg`, `002.jpg`），方便索引
- 如果推理输出不存在或不足，回退到「新增样例数据集规范」，请用户提供数据集

## 新增样例数据集规范

当没有可直接复用的样例数据集，且推理输出也不可用时，需要新增。步骤如下：

### Step 1：确定任务类型和数据字段

根据任务类型确定 `metadata.csv` 的列：

| 任务 | 必需列 | 可选列 |
|------|--------|--------|
| 基础图像生成 | `image`, `prompt` | - |
| 多图像输入 | `image`（JSON 数组）, `prompt` | - |
| 视频生成 | `video`, `prompt` | `frame_rate` |
| 音视频联合 | `video`, `prompt`, `input_audio` | `frame_rate` |
| 图像编辑 | `image`, `edit_image`, `prompt` | - |

### Step 2：准备数据文件

```
data/diffsynth_example_dataset/{series}/{dataset_name}/
├── metadata.csv
└── images/                    # 或 videos/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

图像文件相对于 `base_path` 存放。视频任务中，视频文件可以是目录（包含帧图像序列）或视频文件。

### Step 3：编写 `metadata.csv`

**基础图像生成**：
```csv
image,prompt
images/001.jpg,a beautiful sunset
images/002.jpg,a mountain landscape
```

**多图像输入**（`image` 列为 JSON 数组）：
```csv
image,prompt
"[\"images/001_a.jpg\", \"images/001_b.jpg\"]",a layered scene
```

**视频任务**：
```csv
video,prompt,frame_rate
videos/001,a person walking,24
videos/002,a cat running,24
```

### Step 4：提交到 ModelScope

数据集需要推送到 `DiffSynth-Studio/diffsynth_example_dataset` 仓库，目录结构为 `{series}/{dataset_name}/`。

**提交流程**（agent 不可自行推送）：
1. Agent 创建好 commit 后，向用户展示 commit 内容和数据集文件清单
2. **用户确认**数据集内容无误
3. 用户确认后，agent 执行 `git push` 将数据集提交到远程仓库

**如果暂时无法提交**：可以先在本地使用数据集进行训练验证，确认训练流程正确后再提交到远程仓库。

### Step 5：接入训练流程

数据集准备就绪后，继续按照本指南的后续步骤完成训练模块接入。

## 训练 Key 命名约定

**核心原则**：训练数据集的列名必须与 Pipeline Unit 的 `input_params` 声明一致。DiffSynth 遵循训推一致原则，**能推理就能训练**，不存在中间映射层。

### 数据集列名 = Unit input_params

训练数据流中，数据集列名直接传递给 Unit 参数：

```
metadata.csv 列名 "edit_image" → parse_extra_inputs 直接注入 → Unit process(self, pipe, edit_image, ...)
```

**如果列名不一致，修改数据集文件的列名，不修改代码做适配。**

### 为什么需要保持一致

以图像编辑任务为例，Pipeline 中的 EditImageEmbedder Unit 声明了参数名：

```python
class EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize"),
            ...
        )
```

那么数据集的 `metadata.csv` 必须包含 `edit_image` 列——**列名与 `input_params` 中的名字逐字一致**，这样 `parse_extra_inputs` 才能将 `data["edit_image"]` 直接注入到 `inputs_shared["edit_image"]`，供 Unit 使用。

### 新模型接入时的检查步骤

1. **查看 Pipeline Units 顺序和声明**：打开 `pipeline.py`，找到每个 Unit 的 `input_params` 声明
2. **识别非标准字段**：除了 `prompt`、`image`、`input_image` 等通用字段外，列出所有 Unit 需要的特殊字段及其确切名称
3. **对照数据集列名**：检查数据集 `metadata.csv` 的列名是否与 Unit `input_params` 一致
4. **修改数据集列名**：对于不一致的列，修改 `metadata.csv` 的列名使其匹配
5. **设置 `--extra_inputs`**：确保训练脚本中的 `--extra_inputs` 使用数据集列名（与 Unit `input_params` 一致）

### 常见任务的 dataset 列名

| 任务类型 | 数据集列名 | 说明 |
|---------|-----------|------|
| 基础图像生成 | `image`, `prompt` | `image` 会被映射到 `input_image` 参数 |
| 图像编辑 | `image`, `edit_image`, `prompt` | `edit_image` 与 Unit `input_params` 一致 |
| 视频生成 | `video`, `prompt` | `video` 会被映射到 `input_video` 参数 |
| 音视频联合 | `video`, `prompt`, `input_audio` | 列名与 Unit 参数一致 |

**检查清单**：
- [ ] 已阅读 `pipeline.py` 中所有 Unit 的 `input_params`
- [ ] 已确认数据集 `metadata.csv` 列名与 Unit `input_params` 完全一致
- [ ] 复用数据集时，如有列名不一致，已修改 `metadata.csv` 使其匹配
- [ ] `--data_file_keys` 使用数据集列名
- [ ] `--extra_inputs` 使用数据集列名
