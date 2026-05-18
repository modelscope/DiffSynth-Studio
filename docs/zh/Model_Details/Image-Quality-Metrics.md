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
from PIL import Image
from diffsynth.metrics import PickScoreMetric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = PickScoreMetric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/PickScore_v1"),
    processor_config=ModelConfig(model_id="AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    device=device,
)
score = metric.calc_scores(prompt, image)[0]
print("PickScore:", score)
```

## 指标总览

|指标|默认模型|输入|输出|示例代码|
|-|-|-|-|-|
|PickScore|[AI-ModelScope/PickScore_v1](https://www.modelscope.cn/models/AI-ModelScope/PickScore_v1)|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/pickscore.py)|
|ImageReward|[ZhipuAI/ImageReward](https://www.modelscope.cn/models/ZhipuAI/ImageReward)|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/image_reward.py)|
|HPSv2|[AI-ModelScope/HPSv2](https://www.modelscope.cn/models/AI-ModelScope/HPSv2)|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/hpsv2.py)|
|HPSv3|[MizzenAI/HPSv3](https://www.modelscope.cn/models/MizzenAI/HPSv3)|prompt + PIL 图像|偏好分数|[code](../../../examples/image_quality_metric/hpsv3.py)|
|CLIP Score|[AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K](https://www.modelscope.cn/models/AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K)|prompt + PIL 图像|图文相似度|[code](../../../examples/image_quality_metric/clipscore.py)|
|Aesthetic|[AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE](https://www.modelscope.cn/models/AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE)|PIL 图像|美学分数|[code](../../../examples/image_quality_metric/aesthetic.py)|
|FID|[diffusionTry/weights-inception-2015-12-05-6726825d](https://www.modelscope.cn/models/diffusionTry/weights-inception-2015-12-05-6726825d)|reference 图像目录 + generated 图像目录|分布距离|[code](../../../examples/image_quality_metric/fid.py)|

## 单图奖励模型

**PickScore**、**ImageReward**、**HPSv2**、**HPSv3** 和 **CLIP Score** 的输入形式相同：一段文本提示词和一张已经打开的 `PIL.Image.Image`。示例：

```python
from PIL import Image
from diffsynth.metrics import CLIPMetric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = CLIPMetric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    device=device,
)
scores = metric.calc_scores(prompt, image)[0]
```

如果要评估多张图像，可以传入 PIL 图像列表：

```python
scores = metric.calc_scores(prompt, [image1, image2, image3])
```

其中 prompt 为单个字符串时，会对每张图像使用同一个 prompt。prompt 为字符串列表时，prompt 数量需要和图像数量一致。

## Aesthetic

Aesthetic 只评估图像审美质量，不使用 prompt。

```python
from PIL import Image
from diffsynth.metrics import AestheticMetric

path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
metric = AestheticMetric.from_pretrained(device="cuda")
score = metric.calc_scores(image)[0]
```

## FID

FID 用于比较两组图像的特征分布。它不是单图打分，也不使用 prompt。典型用法是比较真实参考图像目录和生成结果目录：

```python
from diffsynth.metrics import FIDMetric

reference_dir = FIDMetric.default_reference_dir(
    local_dir="data/examples/ImageQualityMetric/reference/coco_2014_caption_validation",
    max_images=10000,
)
generated_dir = ""

metric = FIDMetric.from_pretrained(device="cuda", batch_size=16)
score = metric.compute(reference_dir, generated_dir)
print("FID:", score)
```

FID 的 reference 不是固定唯一的官方答案。对于通用文生图质量评估，COCO validation 是一个方便的默认选择；对于人像、商品图、医学等垂直任务，应传入该领域真实数据构成的 `reference_dir`。


## 注意事项

* PickScore、ImageReward、HPSv2、HPSv3、CLIPScore、Aesthetic 的分数适合做同一指标内部的相对比较，不建议直接把不同指标的数值大小相互比较。
* HPSv3 基于 Qwen2-VL，模型较大，显存需求明显高于 CLIP 类指标。
* FID 对 reference 选择、样本量和 generated 样本量较敏感。
