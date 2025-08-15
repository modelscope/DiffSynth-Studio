# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/) 

[切换到中文](./README_zh.md)

## Introduction

Welcome to the magic world of Diffusion models! DiffSynth-Studio is an open-source Diffusion model engine developed and maintained by [ModelScope](https://www.modelscope.cn/) team. We aim to foster technical innovation through framework development, bring together the power of the open-source community, and explore the limits of generative models!

DiffSynth currently includes two open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Focused on aggressive technical exploration, for academia, providing support for more cutting-edge model capabilities.
* [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine): Focused on stable model deployment, for industry, offering higher computing performance and more stable features.

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) are the core projects behind ModelScope [AIGC zone](https://modelscope.cn/aigc/home), offering powerful AI content generation abilities. Come and try our carefully designed features and start your AI creation journey!

## Installation

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

<details>
<summary>Other installation methods</summary>

Install from PyPI (version updates may be delayed; for latest features, install from source)

```
pip install diffsynth
```

If you meet problems during installation, they might be caused by upstream dependencies. Please check the docs of these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

</details>

## Basic Framework

DiffSynth-Studio redesigns the inference and training pipelines for mainstream Diffusion models (including FLUX, Wan, etc.), enabling efficient memory management and flexible model training.

### Qwen-Image Series (🔥New Model)

Details: [./examples/qwen_image/](./examples/qwen_image/)

![Image](https://github.com/user-attachments/assets/738078d8-8749-4a53-a046-571861541924)

<details>

<summary>Quick Start</summary>

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "A detailed portrait of a girl underwater, wearing a blue flowing dress, hair gently floating, clear light and shadow, surrounded by bubbles, calm expression, fine details, dreamy and beautiful."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

</details>

<details>

<summary>Model Overview</summary>

|Model ID|Inference|Low VRAM Inference|Full Training|Validation after Full Training|LoRA Training|Validation after LoRA Training|
|-|-|-|-|-|-|-|
|[Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image)|[code](./examples/qwen_image/model_inference/Qwen-Image.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image.py)|[code](./examples/qwen_image/model_training/full/Qwen-Image.sh)|[code](./examples/qwen_image/model_training/validate_full/Qwen-Image.py)|[code](./examples/qwen_image/model_training/lora/Qwen-Image.sh)|[code](./examples/qwen_image/model_training/validate_lora/Qwen-Image.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)|[code](./examples/qwen_image/model_inference/Qwen-Image-Distill-Full.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-Full.py)|[code](./examples/qwen_image/model_training/full/Qwen-Image-Distill-Full.sh)|[code](./examples/qwen_image/model_training/validate_full/Qwen-Image-Distill-Full.py)|[code](./examples/qwen_image/model_training/lora/Qwen-Image-Distill-Full.sh)|[code](./examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-Full.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)|[code](./examples/qwen_image/model_inference/Qwen-Image-Distill-LoRA.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-LoRA.py)|-|-|-|-|
|[DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen)|[code](./examples/qwen_image/model_inference/Qwen-Image-EliGen.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen.py)|-|-|[code](./examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh)|[code](./examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny)|[code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py)|-|-|-|-|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth)|[code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py)|-|-|-|-|

</details>

### FLUX Series

Detail page: [./examples/flux/](./examples/flux/)

![Image](https://github.com/user-attachments/assets/c01258e2-f251-441a-aa1e-ebb22f02594d)

<details>

<summary>Quick Start</summary>

```python
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

image = pipe(prompt="a cat", seed=0)
image.save("image.jpg")
```

</details>

<details>

<summary>Model Overview</summary>

| Model ID | Extra Parameters | Inference | Low VRAM Inference | Full Training | Validate After Full Training | LoRA Training | Validate After LoRA Training |
|-|-|-|-|-|-|-|-|
|[FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)||[code](./examples/flux/model_inference/FLUX.1-dev.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev.py)|[code](./examples/flux/model_training/full/FLUX.1-dev.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev.py)|
|[FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev)||[code](./examples/flux/model_inference/FLUX.1-Krea-dev.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-Krea-dev.py)|[code](./examples/flux/model_training/full/FLUX.1-Krea-dev.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-Krea-dev.py)|[code](./examples/flux/model_training/lora/FLUX.1-Krea-dev.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-Krea-dev.py)|
|[FLUX.1-Kontext-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev)|`kontext_images`|[code](./examples/flux/model_inference/FLUX.1-Kontext-dev.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-Kontext-dev.py)|[code](./examples/flux/model_training/full/FLUX.1-Kontext-dev.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-Kontext-dev.py)|[code](./examples/flux/model_training/lora/FLUX.1-Kontext-dev.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-Kontext-dev.py)|
|[FLUX.1-dev-Controlnet-Inpainting-Beta](https://www.modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)|`controlnet_inputs`|[code](./examples/flux/model_inference/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-Controlnet-Inpainting-Beta.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Inpainting-Beta.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|
|[FLUX.1-dev-Controlnet-Union-alpha](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha)|`controlnet_inputs`|[code](./examples/flux/model_inference/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-Controlnet-Union-alpha.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Union-alpha.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Union-alpha.py)|
|[FLUX.1-dev-Controlnet-Upscaler](https://www.modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler)|`controlnet_inputs`|[code](./examples/flux/model_inference/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-Controlnet-Upscaler.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Upscaler.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Upscaler.py)|
|[FLUX.1-dev-IP-Adapter](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-IP-Adapter)|`ipadapter_images`, `ipadapter_scale`|[code](./examples/flux/model_inference/FLUX.1-dev-IP-Adapter.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-IP-Adapter.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-IP-Adapter.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-IP-Adapter.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev-IP-Adapter.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-IP-Adapter.py)|
|[FLUX.1-dev-InfiniteYou](https://www.modelscope.cn/models/ByteDance/InfiniteYou)|`infinityou_id_image`, `infinityou_guidance`, `controlnet_inputs`|[code](./examples/flux/model_inference/FLUX.1-dev-InfiniteYou.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-InfiniteYou.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-InfiniteYou.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-InfiniteYou.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev-InfiniteYou.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-InfiniteYou.py)|
|[FLUX.1-dev-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen)|`eligen_entity_prompts`, `eligen_entity_masks`, `eligen_enable_on_negative`, `eligen_enable_inpaint`|[code](./examples/flux/model_inference/FLUX.1-dev-EliGen.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-EliGen.py)|-|-|[code](./examples/flux/model_training/lora/FLUX.1-dev-EliGen.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev-EliGen.py)|
|[FLUX.1-dev-LoRA-Encoder](https://www.modelscope.cn/models/DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev)|`lora_encoder_inputs`, `lora_encoder_scale`|[code](./examples/flux/model_inference/FLUX.1-dev-LoRA-Encoder.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev-LoRA-Encoder.py)|[code](./examples/flux/model_training/full/FLUX.1-dev-LoRA-Encoder.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev-LoRA-Encoder.py)|-|-|
|[FLUX.1-dev-LoRA-Fusion-Preview](https://modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev)||[code](./examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py)|-|-|-|-|-|
|[Step1X-Edit](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit)|`step1x_reference_image`|[code](./examples/flux/model_inference/Step1X-Edit.py)|[code](./examples/flux/model_inference_low_vram/Step1X-Edit.py)|[code](./examples/flux/model_training/full/Step1X-Edit.sh)|[code](./examples/flux/model_training/validate_full/Step1X-Edit.py)|[code](./examples/flux/model_training/lora/Step1X-Edit.sh)|[code](./examples/flux/model_training/validate_lora/Step1X-Edit.py)|
|[FLEX.2-preview](https://www.modelscope.cn/models/ostris/Flex.2-preview)|`flex_inpaint_image`, `flex_inpaint_mask`, `flex_control_image`, `flex_control_strength`, `flex_control_stop`|[code](./examples/flux/model_inference/FLEX.2-preview.py)|[code](./examples/flux/model_inference_low_vram/FLEX.2-preview.py)|[code](./examples/flux/model_training/full/FLEX.2-preview.sh)|[code](./examples/flux/model_training/validate_full/FLEX.2-preview.py)|[code](./examples/flux/model_training/lora/FLEX.2-preview.sh)|[code](./examples/flux/model_training/validate_lora/FLEX.2-preview.py)|
|[Nexus-Gen](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2)|`nexus_gen_reference_image`|[code](./examples/flux/model_inference/Nexus-Gen-Editing.py)|[code](./examples/flux/model_inference_low_vram/Nexus-Gen-Editing.py)|[code](./examples/flux/model_training/full/Nexus-Gen.sh)|[code](./examples/flux/model_training/validate_full/Nexus-Gen.py)|[code](./examples/flux/model_training/lora/Nexus-Gen.sh)|[code](./examples/flux/model_training/validate_lora/Nexus-Gen.py)|

</details>



### Wan Series

Detail page: [./examples/wanvideo/](./examples/wanvideo/)

https://github.com/user-attachments/assets/1d66ae74-3b02-40a9-acc3-ea95fc039314

<details>

<summary>Quick Start</summary>

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

video = pipe(
    prompt="A documentary photography style scene: a lively puppy rapidly running on green grass. The puppy has brown-yellow fur, upright ears, and looks focused and joyful. Sunlight shines on its body, making the fur appear soft and shiny. The background is an open field with occasional wildflowers, and faint blue sky and clouds in the distance. Strong sense of perspective captures the motion of the puppy and the vitality of the surrounding grass. Mid-shot side-moving view.",
    negative_prompt="Bright colors, overexposed, static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)
```

</details>

<details>

<summary>Model Overview</summary>

| Model ID | Extra Parameters | Inference | Full Training | Validate After Full Training | LoRA Training | Validate After LoRA Training |
|-|-|-|-|-|-|-|
|[Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.2-I2V-A14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-I2V-A14B.py)|
|[Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)||[code](./examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-T2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-T2V-A14B.py)|
|[Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py)|[code](./examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.2-TI2V-5B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py)|
|[Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)||[code](./examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-T2V-1.3B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B.py)|
|[Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)||[code](./examples/wanvideo/model_inference/Wan2.1-T2V-14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-T2V-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-T2V-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-14B.py)|
|[Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-I2V-14B-480P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-I2V-14B-480P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-480P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-480P.py)|
|[Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)|`input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-720P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-720P.py)|
|[Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-FLF2V-14B-720P.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-FLF2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-FLF2V-14B-720P.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-FLF2V-14B-720P.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py)|
|[PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)|`control_video`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-14B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-InP.py)|
|[PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)|`control_video`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-14B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)|`control_video`, `reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control)|`control_video`, `reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./examples/wanvideo/examples/wanmodel_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP)|`input_image`, `end_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera)|`control_camera_video`, `input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)|`control_camera_video`, `input_image`|[code](./examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|
|[iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-1.3B-Preview.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B-Preview.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B-Preview.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-Preview.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B-Preview.py)|
|[Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py)|
|[Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)|`vace_control_video`, `vace_reference_image`|[code](./examples/wanvideo/model_inference/Wan2.1-VACE-14B.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-VACE-14B.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-VACE-14B.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-14B.py)|
|[DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)|`motion_bucket_id`|[code](./examples/wanvideo/model_inference/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./examples/wanvideo/model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./examples/wanvideo/model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./examples/wanvideo/model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./examples/wanvideo/model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py)|

</details>

### More Models



<details>
<summary>Image Generation Models</summary>

Detail page: [./examples/image_synthesis/](./examples/image_synthesis/)

|FLUX|Stable Diffusion 3|
|-|-|
|![image_1024_cfg](https://github.com/user-attachments/assets/984561e9-553d-4952-9443-79ce144f379f)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|

|Kolors|Hunyuan-DiT|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5)|

|Stable Diffusion|Stable Diffusion XL|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|

</details>



<details>
<summary>Video Generation Models</summary>

- HunyuanVideo: [./examples/HunyuanVideo/](./examples/HunyuanVideo/)

https://github.com/user-attachments/assets/48dd24bb-0cc6-40d2-88c3-10feed3267e9  

- StepVideo: [./examples/stepvideo/](./examples/stepvideo/)

https://github.com/user-attachments/assets/5954fdaa-a3cf-45a3-bd35-886e3cc4581b  

- CogVideoX: [./examples/CogVideoX/](./examples/CogVideoX/)

https://github.com/user-attachments/assets/26b044c1-4a60-44a4-842f-627ff289d006  

</details>



<details>
<summary>Image Quality Assessment Models</summary>

We have integrated a series of image quality assessment models. These models can be used for evaluating image generation models, alignment training, and similar tasks.

Detail page: [./examples/image_quality_metric/](./examples/image_quality_metric/)

* [ImageReward](https://github.com/THUDM/ImageReward)
* [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* [PickScore](https://github.com/yuvalkirstain/pickscore)
* [CLIP](https://github.com/openai/CLIP)
* [HPSv2](https://github.com/tgxs002/HPSv2)
* [HPSv2.1](https://github.com/tgxs002/HPSv2)
* [MPS](https://github.com/Kwai-Kolors/MPS)

</details>



## Innovative Achievements

DiffSynth-Studio is not just an engineering model framework, but also a platform for incubating innovative results.

<details>
<summary>Nexus-Gen: Unified Architecture for Image Understanding, Generation, and Editing</summary>

- Detail page: https://github.com/modelscope/Nexus-Gen  
- Paper: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
- Dataset: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
- Online Demo: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

![](https://github.com/modelscope/Nexus-Gen/raw/main/assets/illustrations/gen_edit.jpg)

</details>

<details>
<summary>ArtAug: Aesthetic Enhancement for Image Generation Models</summary>

- Detail page: [./examples/ArtAug/](./examples/ArtAug/)
- Paper: [ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction](https://arxiv.org/abs/2412.12888)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
- Online Demo: [ModelScope AIGC Tab](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0)

|FLUX.1-dev|FLUX.1-dev + ArtAug LoRA|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/e1d5c505-b423-45fe-be01-25c2758f5417)|![image_1_enhance](https://github.com/user-attachments/assets/335908e3-d0bd-41c2-9d99-d10528a2d719)|

</details>

<details>
<summary>EliGen: Precise Image Region Control</summary>

- Detail page: [./examples/EntityControl/](./examples/EntityControl/)
- Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
- Online Demo: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
- Dataset: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

|Entity Control Mask|Generated Image|
|-|-|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|

</details>

<details>
<summary>ExVideo: Extended Training for Video Generation Models</summary>

- Project Page: [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- Paper: [ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- Code Example: [./examples/ExVideo/](./examples/ExVideo/)
- Model: [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>

<details>
<summary>Diffutoon: High-Resolution Anime-Style Video Rendering</summary>

- Project Page: [Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- Paper: [Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models](https://arxiv.org/abs/2401.16224)
- Code Example: [./examples/Diffutoon/](./examples/Diffutoon/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

</details>

<details>
<summary>DiffSynth: The Initial Version of This Project</summary>

- Project Page: [Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/)
- Paper: [DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463)
- Code Example: [./examples/diffsynth/](./examples/diffsynth/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

</details>



## Update History
- **August 15, 2025** We open-sourced the [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset). This is an image dataset generated using the Qwen-Image model, with a total of 160,000 `1024 x 1024` images. It includes the general, English text rendering, and Chinese text rendering subsets. We provide caption, entity and control images annotations for each image. Developers can use this dataset to train models such as ControlNet and EliGen for the Qwen-Image model. We aim to promote technological development through open-source contributions!

- **August 13, 2025** We trained and open-sourced the ControlNet model for Qwen-Image, [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth), which adopts a lightweight architectural design. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py).

- **August 12, 2025** We trained and open-sourced the ControlNet model for Qwen-Image, [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny), which adopts a lightweight architectural design. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py).

- **August 11, 2025** We released another distilled acceleration model for Qwen-Image, [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA). It uses the same training process as [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full), but the model structure is changed to LoRA. This makes it work better with other open-source models.

- **August 7, 2025** We open-sourced the entity control LoRA of Qwen-Image, [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen). Qwen-Image-EliGen is able to achieve entity-level controlled text-to-image generation. See the [paper](https://arxiv.org/abs/2501.01097) for technical details. Training dataset: [EliGenTrainSet](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet).

- **August 5, 2025** We open-sourced the distilled acceleration model of Qwen-Image, [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full), achieving approximately 5x speedup.

- **August 4, 2025** 🔥 Qwen-Image is now open source. Welcome the new member to the image generation model family!

- **August 1, 2025** [FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) with a focus on aesthetic photography is comprehensively supported, including low-GPU-memory layer-by-layer offload, LoRA training and full training. See [./examples/flux/](./examples/flux/).

- **July 28, 2025** With the open-sourcing of Wan 2.2, we immediately provided comprehensive support, including low-GPU-memory layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, full training. See [./examples/wanvideo/](./examples/wanvideo/).

- **July 11, 2025** We propose Nexus-Gen, a unified model that synergizes the language reasoning capabilities of LLMs with the image synthesis power of diffusion models. This framework enables seamless image understanding, generation, and editing tasks.
  - Paper: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
  - Github Repo: https://github.com/modelscope/Nexus-Gen
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
  - Training Dataset: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
  - Online Demo: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

<details>
<summary>More</summary>

- **June 15, 2025** ModelScope's official evaluation framework, [EvalScope](https://github.com/modelscope/evalscope), now supports text-to-image generation evaluation. Try it with the [Best Practices](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/t2i_eval.html) guide.

- **March 25, 2025** Our new open-source project, [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine), is now open-sourced! Focused on stable model deployment. Geared towards industry. Offers better engineering support, higher computational performance, and more stable functionality.

- **March 31, 2025** We support InfiniteYou, an identity preserving method for FLUX. Please refer to [./examples/InfiniteYou/](./examples/InfiniteYou/) for more details.

- **March 13, 2025** We support HunyuanVideo-I2V, the image-to-video generation version of HunyuanVideo open-sourced by Tencent. Please refer to [./examples/HunyuanVideo/](./examples/HunyuanVideo/) for more details.

- **February 25, 2025** We support Wan-Video, a collection of SOTA video synthesis models open-sourced by Alibaba. See [./examples/wanvideo/](./examples/wanvideo/).

- **February 17, 2025** We support [StepVideo](https://modelscope.cn/models/stepfun-ai/stepvideo-t2v/summary)! State-of-the-art video synthesis model! See [./examples/stepvideo](./examples/stepvideo/).

- **December 31, 2024** We propose EliGen, a novel framework for precise entity-level controlled text-to-image generation, complemented by an inpainting fusion pipeline to extend its capabilities to image inpainting tasks. EliGen seamlessly integrates with existing community models, such as IP-Adapter and In-Context LoRA, enhancing its versatility. For more details, see [./examples/EntityControl](./examples/EntityControl/).
  - Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
  - Online Demo: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
  - Training Dataset: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

- **December 19, 2024** We implement advanced VRAM management for HunyuanVideo, making it possible to generate videos at a resolution of 129x720x1280 using 24GB of VRAM, or at 129x512x384 resolution with just 6GB of VRAM. Please refer to [./examples/HunyuanVideo/](./examples/HunyuanVideo/) for more details.

- **December 18, 2024** We propose ArtAug, an approach designed to improve text-to-image synthesis models through synthesis-understanding interactions. We have trained an ArtAug enhancement module for FLUX.1-dev in the format of LoRA. This model integrates the aesthetic understanding of Qwen2-VL-72B into FLUX.1-dev, leading to an improvement in the quality of generated images.
  - Paper: https://arxiv.org/abs/2412.12888
  - Examples: https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/ArtAug
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
  - Demo: [ModelScope](https://modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0), HuggingFace (Coming soon)

- **October 25, 2024** We provide extensive FLUX ControlNet support. This project supports many different ControlNet models that can be freely combined, even if their structures differ. Additionally, ControlNet models are compatible with high-resolution refinement and partition control techniques, enabling very powerful controllable image generation. See [`./examples/ControlNet/`](./examples/ControlNet/).

- **October 8, 2024.** We release the extended LoRA based on CogVideoX-5B and ExVideo. You can download this model from [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) or [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1).

- **August 22, 2024.** CogVideoX-5B is supported in this project. See [here](/examples/video_synthesis/). We provide several interesting features for this text-to-video model, including
  - Text to video
  - Video editing
  - Self-upscaling
  - Video interpolation

- **August 22, 2024.** We have implemented an interesting painter that supports all text-to-image models. Now you can create stunning images using the painter, with assistance from AI!
  - Use it in our [WebUI](#usage-in-webui).

- **August 21, 2024.** FLUX is supported in DiffSynth-Studio.
  - Enable CFG and highres-fix to improve visual quality. See [here](/examples/image_synthesis/README.md)
  - LoRA, ControlNet, and additional models will be available soon.

- **June 21, 2024.** We propose ExVideo, a post-tuning technique aimed at enhancing the capability of video generation models. We have extended Stable Video Diffusion to achieve the generation of long videos up to 128 frames.
  - [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - Source code is released in this repo. See [`examples/ExVideo`](./examples/ExVideo/).
  - Models are released on [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) and [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1).
  - Technical report is released on [arXiv](https://arxiv.org/abs/2406.14130).
  - You can try ExVideo in this [Demo](https://huggingface.co/spaces/modelscope/ExVideo-SVD-128f-v1)!

- **June 13, 2024.** DiffSynth Studio is transferred to ModelScope. The developers have transitioned from "I" to "we". Of course, I will still participate in development and maintenance.

- **Jan 29, 2024.** We propose Diffutoon, a fantastic solution for toon shading.
  - [Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
  - The source codes are released in this project.
  - The technical report (IJCAI 2024) is released on [arXiv](https://arxiv.org/abs/2401.16224).

- **Dec 8, 2023.** We decide to develop a new Project, aiming to release the potential of diffusion models, especially in video synthesis. The development of this project is started.

- **Nov 15, 2023.** We propose FastBlend, a powerful video deflickering algorithm.
  - The sd-webui extension is released on [GitHub](https://github.com/Artiprocher/sd-webui-fastblend).
  - Demo videos are shown on Bilibili, including three tasks.
    - [Video deflickering](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [Video interpolation](https://www.bilibili.com/video/BV1Lw411m71p)
    - [Image-driven video rendering](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - The technical report is released on [arXiv](https://arxiv.org/abs/2311.09265).
  - An unofficial ComfyUI extension developed by other users is released on [GitHub](https://github.com/AInseven/ComfyUI-fastblend).

- **Oct 1, 2023.** We release an early version of this project, namely FastSDXL. A try for building a diffusion engine.
  - The source codes are released on [GitHub](https://github.com/Artiprocher/FastSDXL).
  - FastSDXL includes a trainable OLSS scheduler for efficiency improvement.
    - The original repo of OLSS is [here](https://github.com/alibaba/EasyNLP/tree/master/diffusion/olss_scheduler).
    - The technical report (CIKM 2023) is released on [arXiv](https://arxiv.org/abs/2305.14677).
    - A demo video is shown on [Bilibili](https://www.bilibili.com/video/BV1w8411y7uj).
    - Since OLSS requires additional training, we don't implement it in this project.

- **Aug 29, 2023.** We propose DiffSynth, a video synthesis framework.
  - [Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/).
  - The source codes are released in [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth).
  - The technical report (ECML PKDD 2024) is released on [arXiv](https://arxiv.org/abs/2308.03463).

</details>