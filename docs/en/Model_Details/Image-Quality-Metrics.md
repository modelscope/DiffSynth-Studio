# Image Quality Evaluation Metrics

DiffSynth-Studio provides a suite of image quality evaluation metrics and reward models in `diffsynth.metrics` to assess text alignment, aesthetic quality, human preference, and image distribution quality of generated images. Example code for these metrics can be found in [`examples/image_quality_metric/`](../../../examples/image_quality_metric/).

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Install Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load PickScore and score an image against a prompt. The default models will be downloaded from ModelScope to `./models`.

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

## Metrics Overview

| Metric | Default Model | Input | Output | Example Code |
| --- | --- | --- | --- | --- |
| PickScore | [AI-ModelScope/PickScore_v1](https://www.modelscope.cn/models/AI-ModelScope/PickScore_v1) | prompt + PIL Image | Preference Score | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/pickscore.py) |
| ImageReward | [ZhipuAI/ImageReward](https://www.modelscope.cn/models/ZhipuAI/ImageReward) | prompt + PIL Image | Preference Score | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/image_reward.py) |
| HPSv2 | [AI-ModelScope/HPSv2](https://www.modelscope.cn/models/AI-ModelScope/HPSv2) | prompt + PIL Image | Preference Score | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/hpsv2.py) |
| HPSv3 | [MizzenAI/HPSv3](https://www.modelscope.cn/models/MizzenAI/HPSv3) | prompt + PIL Image | Preference Score | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/hpsv3.py) |
| CLIP Score | [AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K](https://www.modelscope.cn/models/AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K) | prompt + PIL Image | Text-Image Similarity | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/clipscore.py) |
| Aesthetic | [AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE](https://www.modelscope.cn/models/AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE) | PIL Image | Aesthetic Score | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/aesthetic.py) |
| FID | [diffusionTry/weights-inception-2015-12-05-6726825d](https://www.modelscope.cn/models/diffusionTry/weights-inception-2015-12-05-6726825d) | reference image directory + generated image directory | Distribution Distance | [code](https://www.google.com/search?q=../../../examples/image_quality_metric/fid.py) |

## Single-Image Reward Models

**PickScore**, **ImageReward**, **HPSv2**, **HPSv3**, and **CLIP Score** share the same input format: a text prompt and an opened `PIL.Image.Image`. Example:

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

If you want to evaluate multiple images, you can pass a list of PIL images:

```python
scores = metric.calc_scores(prompt, [image1, image2, image3])
```

When the prompt is a single string, the same prompt will be used for every image. When the prompt is a list of strings, the number of prompts must match the number of images.

## Aesthetic

Aesthetic only evaluates the aesthetic quality of the image and does not use a prompt.

```python
from PIL import Image
from diffsynth.metrics import AestheticMetric

path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
metric = AestheticMetric.from_pretrained(device="cuda")
score = metric.calc_scores(image)[0]
```

## FID

FID is used to compare the feature distributions of two sets of images. It does not score single images, nor does it use a prompt. A typical use case is comparing a directory of real reference images against a directory of generated results:

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

The reference for FID is not a single, fixed official answer. For general text-to-image quality evaluation, the COCO validation set is a convenient default choice; for vertical tasks such as portraits, product images, or medical images, a `reference_dir` consisting of real data from that specific domain should be provided.

## Important Notes

* The scores from PickScore, ImageReward, HPSv2, HPSv3, CLIPScore, and Aesthetic are suitable for relative comparison within the same metric. It is not recommended to directly compare the numerical values across different metrics.
* HPSv3 is based on Qwen2-VL and is a larger model, requiring significantly more VRAM than CLIP-based metrics.
* FID is sensitive to the choice of reference, the reference sample size, and the generated sample size.