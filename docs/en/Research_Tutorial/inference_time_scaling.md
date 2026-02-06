# Inference-Time Scaling

DiffSynth-Studio supports Inference-time Scaling technology, specifically implementing the **Spectral Evolution Search (SES)** algorithm. This technology allows users to improve generated image quality during the inference stage by increasing computational cost, without retraining the model.

## 1. Basic Principle

The traditional text-to-image inference process starts from random Gaussian noise and generates an image through a fixed number of denoising steps. The quality generated in this way highly depends on the randomness of the initial noise.

**SES (Spectral Evolution Search)** transforms the inference process into a **search optimization problem** targeting the initial noise:

1. **Search Space**: Search for the lowest frequency part of the initial noise in the frequency domain of wavelet transform.
2. **Evolutionary Strategy**: Use the Cross-Entropy Method to iteratively sample noise populations.
3. **Reward Feedback**: Use reward models such as PickScore to score generated low-step preview images.
4. **Result Output**: Find the noise with the highest score and perform complete high-quality denoising.

This method essentially **trades inference computation time for generation quality**.

For more technical details on this method, please refer to the paper: **[Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation](https://arxiv.org/abs/2602.03208)**.

## 2. Quick Start

In DiffSynth-Studio, SES has been integrated into the pipelines of mainstream text-to-image models. You only need to set `enable_ses=True` when calling `pipe()` to enable it.

The following is [quick start code](https://www.google.com/search?q=../../../examples/z_image/model_inference/Z-Image-Turbo-SES.py) using **Z-Image-Turbo** as an example:

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

prompt = "Chinese woman in red Hanfu holding a fan, with a bright yellow neon lightning bolt floating above her palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."

image = pipe(
    prompt=prompt, 
    seed=42, 
    rand_device="cuda",
    enable_ses=True,
    ses_reward_model="pick",
    ses_eval_budget=50,
    ses_inference_steps=8
)
image.save("image_Z-Image-Turbo_ses.jpg")
```

## 3. Supported Models and Parameters

### 3.1 Core Parameters Detailed

In the `pipe()` call, the following parameters control the behavior of SES:

| Parameter Name | Type | Default Value | Description |
| --- | --- | --- | --- |
| `enable_ses` | `bool` | `False` | Whether to enable SES optimization. |
| `ses_reward_model` | `str` | `"pick"` | Reward model selection. Supports `"pick"` (PickScore), `"hps"` (HPSv2), `"clip"`. |
| `ses_eval_budget` | `int` | `50` | Total search budget (total number of evaluated samples). Higher values mean a higher quality ceiling but longer time consumption. |
| `ses_inference_steps` | `int` | `10` | The number of steps used to generate preview images during the search phase. Higher values provide more accurate quality assessment for candidate noise but take longer; 8-15 is recommended. |

### 3.2 Supported Model List

Currently, the following text-to-image models support SES:

* **[Qwen-Image](../../../examples/qwen_image/model_inference/Qwen-Image-SES.py)**
* **[FLUX.1-dev](../../../examples/flux/model_inference/FLUX.1-dev-SES.py)**
* **[FLUX.2-dev](../../../examples/flux2/model_inference/FLUX.2-dev-SES.py)**
* **[Z-Image](../../../examples/z_image/model_inference/Z-Image-SES.py) / [Z-Image-Turbo](../../../examples/z_image/model_inference/Z-Image-Turbo-SES.py)**

## 4. Effect Demonstration

As the search budget (`ses_eval_budget`) increases, SES can stably improve image quality. The following shows the quality changes brought by different computational budgets under the same random seed.

**Scenario 1: Qwen-Image**

* **Prompt**: *"Springtime in the style of Paul Delvaux"*
* **Reward Model**: PickScore

| **Budget = 0** | **Budget = 10** | **Budget = 30** | **Budget = 50** |
| --- | --- | --- | --- |
|  |  |  |  |
| <img width="220" alt="Image" src="https://github.com/user-attachments/assets/250a8c18-d086-49ed-98dc-5eebc5234231" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/3d4b9ecf-36bc-4f63-81fe-e0be9526f103" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/1ed69cca-bd76-43da-940b-b8da49b5a693" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/bd887d83-fc78-4a1c-abb9-be814aefa8f9" /> |

**Scenario 2: FLUX.1-dev**

* **Prompt**: *"A masterful painting of a young woman in the style of Diego Velázquez."*
* **Reward Model**: HPSv2

| **Budget = 0** | **Budget = 10** | **Budget = 30** | **Budget = 50** |
| --- | --- | --- | --- |
|  |  |  |  |
| <img width="220" alt="Image" src="https://github.com/user-attachments/assets/c34a89b8-2f5a-420d-ad23-844a2befd074" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/ba9cb54a-1b50-4ada-b636-237e71383617" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/e3327a88-a903-4c30-8ee7-9e35fb117cc4" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/45e5faa2-e69d-4ffc-9400-e80c0a78f1bd" /> |