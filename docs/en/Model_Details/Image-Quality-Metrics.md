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
from diffsynth.metrics import PickScoreMetric, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)

image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
device = "cuda"

metric = PickScoreMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="PickScore/model.safetensors"),
    device=device
    )

score = metric.compute(prompt, image)[0]
print(f"PickScore score:: {score:.3f}")
```

## Metrics Overview

| Metric | Input | Output | Example Code |
| --- | --- | --- | --- |
| PickScore | prompt + PIL Image | Preference Score | [code](../../../examples/image_quality_metric/pickscore.py) |
| ImageReward | prompt + PIL Image | Preference Score | [code](../../../examples/image_quality_metric/image_reward.py) |
| HPSv2 | prompt + PIL Image | Preference Score | [code](../../../examples/image_quality_metric/hpsv2.py) |
| HPSv3 | prompt + PIL Image | Preference Score | [code](../../../examples/image_quality_metric/hpsv3.py) |
| CLIP Score | prompt + PIL Image | Text-Image Similarity | [code](../../../examples/image_quality_metric/clipscore.py) |
| Aesthetic | PIL Image | Aesthetic Score | [code](../../../examples/image_quality_metric/aesthetic.py) |
| FID | reference image directory + generated image directory | Distribution Distance | [code](../../../examples/image_quality_metric/fid.py) |

### Text-Image Alignment and Preference Evaluation

Applicable metrics: **PickScore**, **ImageReward**, **HPSv2**, **HPSv3**, **CLIP Score**

These models are used to evaluate whether an image follows the prompt and aligns with human visual preferences. They must receive both the `prompt` and the `image` simultaneously.

**Basic Scoring**

```python
score = metric.compute(prompt, image)[0]
```

**Batch Scoring**

If you need to evaluate multiple images, you can directly pass a list:

```python
scores = metric.compute("a cute cat", [image1, image2, image3])

scores = metric.compute(["a cat", "a dog"], [image_cat, image_dog])
```

When prompt is a single string, the same prompt will be applied to every image. When prompt is a list of strings, the number of prompts must exactly match the number of images.

### Pure Image Aesthetics Evaluation

Applicable metric: **Aesthetic**

This model solely evaluates aesthetic features such as the composition, color, and clarity of the image itself. It does not require a prompt.

```python
from diffsynth.metrics import AestheticMetric

metric = AestheticMetric.from_pretrained(device="cuda")
score = metric.compute(image)[0]
```

### Dataset Distribution Evaluation

Applicable metric: **FID** (Fréchet Inception Distance)

FID does not score individual images; instead, it compares the overall feature distribution distance between a real reference image set and a generated image set. A lower score indicates that the generated distribution is closer to the real distribution.

```python
from diffsynth.metrics import FIDMetric

reference_dir = "path/to/real_reference_images"
generated_dir = "path/to/model_generated_images"

metric = FIDMetric.from_pretrained(device="cuda", batch_size=16)
fid_score = metric.compute(reference_dir, generated_dir)
print(f"FID: {fid_score:.3f}")
```

The baseline for FID is not fixed or unique. For general image generation, COCO Validation is commonly used; for specific domains (such as medical images or e-commerce products), a `reference_dir` composed of real data from that specific domain should be provided.

## Important Notes

* The scores from PickScore, ImageReward, HPSv2, HPSv3, CLIPScore, and Aesthetic are suitable for relative comparison within the same metric. It is not recommended to directly compare the numerical values across different metrics.
* HPSv3 is based on Qwen2-VL and is a larger model, requiring significantly more VRAM than CLIP-based metrics.
* FID is sensitive to the choice of reference, the reference sample size, and the generated sample size.
