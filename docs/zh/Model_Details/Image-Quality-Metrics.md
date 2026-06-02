# 图像质量评估指标

DiffSynth-Studio 在 `diffsynth.metrics` 中提供了一组图像质量评估指标和奖励模型，用于评估生成图像的文本对齐、审美质量、人类偏好和图像分布质量。这些指标的示例代码位于 [`examples/image_quality_metric/`](../../../examples/image_quality_metric/)。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 PickScore，并对一张图像和一段提示词进行评分。默认模型会从 ModelScope 下载到 `./models`。

```python
from diffsynth.metrics import PickScoreMetric, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)
image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "a dog"
metric = PickScoreMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="PickScore/model.safetensors"),
    device="cuda"
)
score = metric.compute(prompt, image)[0]
print(f"PickScore score:: {score:.3f}")
```

## 指标总览

|指标|输入|输出|示例代码|
|-|-|-|-|
|PickScore|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/pickscore.py)|
|ImageReward|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/image_reward.py)|
|HPSv2|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/hpsv2.py)|
|HPSv3|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/hpsv3.py)|
|CLIP Score|prompt + PIL 图像|图文匹配度|[code](../../../examples/image_quality_metric/clipscore.py)|
|UnifiedReward 2.0|prompt + PIL 图像|多维度分数|[code](../../../examples/image_quality_metric/unified_reward_2.py)|
|Qwen-Image-Bench|prompt + PIL 图像|多级维度分数|[code](../../../examples/image_quality_metric/qwen_image_bench.py)|
|UnifiedReward Edit|编辑指令 + 源图 + 编辑图|图像编辑质量分数|[code](../../../examples/image_quality_metric/unified_reward_edit.py)|
|Aesthetic|PIL 图像|美学分数|[code](../../../examples/image_quality_metric/aesthetic.py)|
|FID|reference 图像目录 + generated 图像目录|分布距离|[code](../../../examples/image_quality_metric/fid.py)|

### 文本-图像对齐与偏好评估

适用指标： **PickScore**，**ImageReward**，**HPSv2**，**HPSv3**，**CLIP Score**，**UnifiedReward 2.0**，**Qwen-Image-Bench**

这类模型用于评估图像是否遵循提示词以及是否符合人类视觉偏好。它们必须同时接收 `prompt` 和 `image`。

**基础打分**
```python
score = metric.compute(prompt, image)[0]
```

**批量打分**
如果需要评估多张图像，可以直接传入列表：

```python
scores = metric.compute("a cute cat", [image1, image2, image3])

scores = metric.compute(["a cat", "a dog"], [image_cat, image_dog])
```

其中 prompt 为单个字符串时，会对每张图像使用同一个 prompt。prompt 为字符串列表时，prompt 数量需要和图像数量一致。

### 多维度图像质量评估

适用指标： **UnifiedReward 2.0**，**Qwen-Image-Bench**

这两个指标同样接收 `prompt` 和 `image`，但除了主分数外，还会通过 `evaluate()` 返回更细的评估维度，适合需要分析图文对齐、画面一致性、风格或多级质量维度的场景。

**Qwen-Image-Bench**

```python
from diffsynth.metrics import ModelConfig, QwenImageBenchMetric

metric = QwenImageBenchMetric.from_pretrained(
    model_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Bench",
        origin_file_pattern="model-*.safetensors",
    ),
    processor_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Bench",
        origin_file_pattern="",
    ),
    device="cuda",
)
details = metric.evaluate(prompt, image)[0]
score = details["total_score"]
print(details["level1_scores"])
print(details["level2_scores"])
```

如果只需要主分数，也可以调用 `metric.compute(prompt, image)`。

### 图像编辑质量评估

适用指标： **UnifiedReward Edit**

UnifiedReward Edit 用于评估编辑结果是否遵循编辑指令，并衡量是否存在过度编辑。输入通常包括编辑指令、源图和编辑图。它支持三种任务：

* `edit_pointwise_score`：对单个编辑结果打分，输入为 `[source_image, edited_image]`。
* `edit_pairwise_rank`：比较两个编辑结果并返回胜者，输入为 `[source_image, edited_image_1, edited_image_2]`。
* `edit_pairwise_score`：分别返回两个编辑结果的分数，输入为 `[source_image, edited_image_1, edited_image_2]`。

```python
from diffsynth.metrics import ModelConfig, UnifiedRewardEditMetric

metric = UnifiedRewardEditMetric.from_pretrained(
    model_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors",
    ),
    processor_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/",
    ),
    device="cuda",
)

details = metric.evaluate(
    instruction,
    [source_image, edited_image],
    task="edit_pointwise_score",
)[0]
print(details["score"], details["editing_success"], details["overediting"])
```

### 纯图像美学评估

适用指标： **Aesthetic**

该模型仅评估图像本身的构图、色彩、清晰度等美学特征，不需要提示词介入。


```python
from diffsynth.metrics import AestheticMetric

metric = AestheticMetric.from_pretrained(device="cuda")
score = metric.compute(image)[0]
```

### 数据集分布评估
适用指标： **FID** (Fréchet Inception Distance)

FID 不对单张图片打分，而是比较真实参考图像集与生成图像集的整体特征分布距离。分数越低，说明生成分布越接近真实分布。

```python
from diffsynth.metrics import FIDMetric

reference_dir = "path/to/real_reference_images"
generated_dir = "path/to/model_generated_images"

metric = FIDMetric.from_pretrained(device="cuda", batch_size=16)
fid_score = metric.compute(reference_dir, generated_dir)
print(f"FID: {fid_score:.3f}")
```

FID 的基准不是固定唯一的。对于通用图像生成，常使用 COCO Validation；如果是特定领域（如医学图像、电商商品），应提供该领域真实数据构成的 `reference_dir`。


## 注意事项

* PickScore、ImageReward、HPSv2、HPSv3、CLIPScore、UnifiedReward 2.0、Qwen-Image-Bench、UnifiedReward Edit、Aesthetic 的分数适合做同一指标内部的相对比较，不建议直接把不同指标的数值大小相互比较。
* HPSv3、UnifiedReward 2.0、UnifiedReward Edit 和 Qwen-Image-Bench 基于多模态大模型，显存需求明显高于 CLIP 类指标。
* FID 对 reference 选择、样本量和 generated 样本量较敏感。
