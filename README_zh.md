# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/) 

[Switch to English](./README.md)

## 简介

欢迎来到 Diffusion 模型的魔法世界！DiffSynth-Studio 是由[魔搭社区](https://www.modelscope.cn/)团队开发和维护的开源 Diffusion 模型引擎。我们期望以框架建设孵化技术创新，凝聚开源社区的力量，探索生成式模型技术的边界！

DiffSynth 目前包括两个开源项目：
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): 聚焦于激进的技术探索，面向学术界，提供更前沿的模型能力支持。
* [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine): 聚焦于稳定的模型部署，面向工业界，提供更高的计算性能与更稳定的功能。

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 与 [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) 作为魔搭社区 [AIGC 专区](https://modelscope.cn/aigc/home) 的核心技术支撑，提供了强大的AI生成内容能力。欢迎体验我们精心打造的产品化功能，开启您的AI创作之旅！

## 安装

从源码安装（推荐）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

<details>
<summary>其他安装方式</summary>

从 pypi 安装（存在版本更新延迟，如需使用最新功能，请从源码安装）

```
pip install diffsynth
```

如果在安装过程中遇到问题，可能是由上游依赖包导致的，请参考这些包的文档：

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

</details>



## 基础框架

DiffSynth-Studio 为主流 Diffusion 模型（包括 FLUX、Wan 等）重新设计了推理和训练流水线，能够实现高效的显存管理、灵活的模型训练。

### FLUX 系列

详细页面：[./examples/flux/](./examples/flux/)

![Image](https://github.com/user-attachments/assets/c01258e2-f251-441a-aa1e-ebb22f02594d)

<details>

<summary>快速开始</summary>

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

<summary>模型总览</summary>

|模型 ID|额外参数|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)||[code](./examples/flux/model_inference/FLUX.1-dev.py)|[code](./examples/flux/model_inference_low_vram/FLUX.1-dev.py)|[code](./examples/flux/model_training/full/FLUX.1-dev.sh)|[code](./examples/flux/model_training/validate_full/FLUX.1-dev.py)|[code](./examples/flux/model_training/lora/FLUX.1-dev.sh)|[code](./examples/flux/model_training/validate_lora/FLUX.1-dev.py)|
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
|[Nexus-Gen-Edit](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2)|`nexus_gen_reference_image`|[code](./examples/flux/model_inference/Nexus-Gen-Editing.py)|[code](./examples/flux/model_inference_low_vram/Nexus-Gen-Editing.py)|[code](./examples/flux/model_training/full/FLUX.1-NexusGen-Edit.sh)|[code](./examples/flux/model_training/validate_full/Nexus-Gen-Editing.py)|[code](./examples/flux/model_training/lora/FLUX.1-NexusGen-Edit.sh)|[code](./examples/flux/model_training/validate_lora/Nexus-Gen-Editing.py)|
</details>

### Wan 系列

详细页面：[./examples/wanvideo/](./examples/wanvideo/)

https://github.com/user-attachments/assets/1d66ae74-3b02-40a9-acc3-ea95fc039314

<details>

<summary>快速开始</summary>

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
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)
```

</details>

<details>

<summary>模型总览</summary>

|模型 ID|额外参数|推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
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



### 更多模型



<details>
<summary>图像生成模型</summary>

详细页面：[./examples/image_synthesis/](./examples/image_synthesis/)

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
<summary>视频生成模型</summary>

- HunyuanVideo：[./examples/HunyuanVideo/](./examples/HunyuanVideo/)

https://github.com/user-attachments/assets/48dd24bb-0cc6-40d2-88c3-10feed3267e9

- StepVideo：[./examples/stepvideo/](./examples/stepvideo/)

https://github.com/user-attachments/assets/5954fdaa-a3cf-45a3-bd35-886e3cc4581b

- CogVideoX：[./examples/CogVideoX/](./examples/CogVideoX/)

https://github.com/user-attachments/assets/26b044c1-4a60-44a4-842f-627ff289d006

</details>



<details>
<summary>图像质量评估模型</summary>

我们集成了一系列图像质量评估模型，这些模型可以用于图像生成模型的评测、对齐训练等场景中。

详细页面：[./examples/image_quality_metric/](./examples/image_quality_metric/)

* [ImageReward](https://github.com/THUDM/ImageReward)
* [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor)
* [PickScore](https://github.com/yuvalkirstain/pickscore)
* [CLIP](https://github.com/openai/CLIP)
* [HPSv2](https://github.com/tgxs002/HPSv2)
* [HPSv2.1](https://github.com/tgxs002/HPSv2)
* [MPS](https://github.com/Kwai-Kolors/MPS)

</details>



## 创新成果

DiffSynth-Studio 不仅仅是一个工程化的模型框架，更是创新成果的孵化器。

<details>
<summary>Nexus-Gen: 统一架构的图像理解、生成、编辑</summary>

- 详细页面：https://github.com/modelscope/Nexus-Gen
- 论文：[Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
- 数据集：[ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
- 在线体验：[ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

![](https://github.com/modelscope/Nexus-Gen/raw/main/assets/illustrations/gen_edit.jpg)

</details>



<details>
<summary>ArtAug: 图像生成模型的美学提升</summary>

- 详细页面：[./examples/ArtAug/](./examples/ArtAug/)
- 论文：[ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction](https://arxiv.org/abs/2412.12888)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
- 在线体验：[ModelScope AIGC Tab](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0)

|FLUX.1-dev|FLUX.1-dev + ArtAug LoRA|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/e1d5c505-b423-45fe-be01-25c2758f5417)|![image_1_enhance](https://github.com/user-attachments/assets/335908e3-d0bd-41c2-9d99-d10528a2d719)|

</details>



<details>

<summary>EliGen: 精准的图像分区控制</summary>

- 详细页面：[./examples/EntityControl/](./examples/EntityControl/)
- 论文：[EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
- 在线体验：[ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
- 数据集：[EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

|实体控制区域|生成图像|
|-|-|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|

</details>



<details>

<summary>ExVideo: 视频生成模型的扩展训练</summary>

- 项目页面：[Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- 论文：[ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- 代码样例：[./examples/ExVideo/](./examples/ExVideo/)
- 模型：[ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>



<details>

<summary>Diffutoon: 高分辨率动漫风格视频渲染</summary>

- 项目页面：[Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- 论文：[Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models](https://arxiv.org/abs/2401.16224)
- 代码样例：[./examples/Diffutoon/](./examples/Diffutoon/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

</details>



<details>

<summary>DiffSynth: 本项目的初代版本</summary>

- 项目页面：[Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/)
- 论文：[DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463)
- 代码样例：[./examples/diffsynth/](./examples/diffsynth/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

</details>



## 更新历史

- **2025年7月11日** 🔥🔥🔥 我们提出 Nexus-Gen，一个将大语言模型（LLM）的语言推理能力与扩散模型的图像生成能力相结合的统一框架。该框架支持无缝的图像理解、生成和编辑任务。
  - 论文: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
  - Github 仓库: https://github.com/modelscope/Nexus-Gen
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
  - 训练数据集: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
  - 在线体验: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

- **2025年6月15日** ModelScope 官方评测框架 [EvalScope](https://github.com/modelscope/evalscope) 现已支持文生图生成评测。请参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/t2i_eval.html)指南进行尝试。

- **2025年3月25日** 我们的新开源项目 [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) 现已开源！专注于稳定的模型部署，面向工业界，提供更好的工程支持、更高的计算性能和更稳定的功能。

<details>
<summary>更多</summary>

- **2025年3月31日** 我们支持 InfiniteYou，一种用于 FLUX 的人脸特征保留方法。更多细节请参考 [./examples/InfiniteYou/](./examples/InfiniteYou/)。

- **2025年3月13日** 我们支持 HunyuanVideo-I2V，即腾讯开源的 HunyuanVideo 的图像到视频生成版本。更多细节请参考 [./examples/HunyuanVideo/](./examples/HunyuanVideo/)。

- **2025年2月25日** 我们支持 Wan-Video，这是阿里巴巴开源的一系列最先进的视频合成模型。详见 [./examples/wanvideo/](./examples/wanvideo/)。

- **2025年2月17日** 我们支持 [StepVideo](https://modelscope.cn/models/stepfun-ai/stepvideo-t2v/summary)！先进的视频合成模型！详见 [./examples/stepvideo](./examples/stepvideo/)。

- **2024年12月31日** 我们提出 EliGen，一种用于精确实体级别控制的文本到图像生成的新框架，并辅以修复融合管道，将其能力扩展到图像修复任务。EliGen 可以无缝集成现有的社区模型，如 IP-Adapter 和 In-Context LoRA，提升其通用性。更多详情，请见 [./examples/EntityControl](./examples/EntityControl/)。
  - 论文: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
  - 在线体验: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
  - 训练数据集: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

- **2024年12月19日** 我们为 HunyuanVideo 实现了高级显存管理，使得在 24GB 显存下可以生成分辨率为 129x720x1280 的视频，或在仅 6GB 显存下生成分辨率为 129x512x384 的视频。更多细节请参考 [./examples/HunyuanVideo/](./examples/HunyuanVideo/)。

- **2024年12月18日** 我们提出 ArtAug，一种通过合成-理解交互来改进文生图模型的方法。我们以 LoRA 格式为 FLUX.1-dev 训练了一个 ArtAug 增强模块。该模型将 Qwen2-VL-72B 的美学理解融入 FLUX.1-dev，从而提升了生成图像的质量。
  - 论文: https://arxiv.org/abs/2412.12888
  - 示例: https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/ArtAug
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
  - 演示: [ModelScope](https://modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0), HuggingFace (即将上线)

- **2024年10月25日** 我们提供了广泛的 FLUX ControlNet 支持。该项目支持许多不同的 ControlNet 模型，并且可以自由组合，即使它们的结构不同。此外，ControlNet 模型兼容高分辨率优化和分区控制技术，能够实现非常强大的可控图像生成。详见 [`./examples/ControlNet/`](./examples/ControlNet/)。

- **2024年10月8日** 我们发布了基于 CogVideoX-5B 和 ExVideo 的扩展 LoRA。您可以从 [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) 或 [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) 下载此模型。

- **2024年8月22日** 本项目现已支持 CogVideoX-5B。详见 [此处](/examples/video_synthesis/)。我们为这个文生视频模型提供了几个有趣的功能，包括：
  - 文本到视频
  - 视频编辑
  - 自我超分
  - 视频插帧

- **2024年8月22日** 我们实现了一个有趣的画笔功能，支持所有文生图模型。现在，您可以在 AI 的辅助下使用画笔创作惊艳的图像了！
  - 在我们的 [WebUI](#usage-in-webui) 中使用它。

- **2024年8月21日** DiffSynth-Studio 现已支持 FLUX。
  - 启用 CFG 和高分辨率修复以提升视觉质量。详见 [此处](/examples/image_synthesis/README.md)
  - LoRA、ControlNet 和其他附加模型将很快推出。

- **2024年6月21日** 我们提出 ExVideo，一种旨在增强视频生成模型能力的后训练微调技术。我们将 Stable Video Diffusion 进行了扩展，实现了长达 128 帧的长视频生成。
  - [项目页面](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - 源代码已在此仓库中发布。详见 [`examples/ExVideo`](./examples/ExVideo/)。
  - 模型已发布于 [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) 和 [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1)。
  - 技术报告已发布于 [arXiv](https://arxiv.org/abs/2406.14130)。
  - 您可以在此 [演示](https://huggingface.co/spaces/modelscope/ExVideo-SVD-128f-v1) 中试用 ExVideo！

- **2024年6月13日** DiffSynth Studio 已迁移至 ModelScope。开发团队也从“我”转变为“我们”。当然，我仍会参与后续的开发和维护工作。

- **2024年1月29日** 我们提出 Diffutoon，这是一个出色的卡通着色解决方案。
  - [项目页面](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
  - 源代码已在此项目中发布。
  - 技术报告（IJCAI 2024）已发布于 [arXiv](https://arxiv.org/abs/2401.16224)。

- **2023年12月8日** 我们决定启动一个新项目，旨在释放扩散模型的潜力，尤其是在视频合成方面。该项目的开发工作正式开始。

- **2023年11月15日** 我们提出 FastBlend，一种强大的视频去闪烁算法。
  - sd-webui 扩展已发布于 [GitHub](https://github.com/Artiprocher/sd-webui-fastblend)。
  - 演示视频已在 Bilibili 上展示，包含三个任务：
    - [视频去闪烁](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [视频插帧](https://www.bilibili.com/video/BV1Lw411m71p)
    - [图像驱动的视频渲染](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - 技术报告已发布于 [arXiv](https://arxiv.org/abs/2311.09265)。
  - 其他用户开发的非官方 ComfyUI 扩展已发布于 [GitHub](https://github.com/AInseven/ComfyUI-fastblend)。

- **2023年10月1日** 我们发布了该项目的早期版本，名为 FastSDXL。这是构建一个扩散引擎的初步尝试。
  - 源代码已发布于 [GitHub](https://github.com/Artiprocher/FastSDXL)。
  - FastSDXL 包含一个可训练的 OLSS 调度器，以提高效率。
    - OLSS 的原始仓库位于 [此处](https://github.com/alibaba/EasyNLP/tree/master/diffusion/olss_scheduler)。
    - 技术报告（CIKM 2023）已发布于 [arXiv](https://arxiv.org/abs/2305.14677)。
    - 演示视频已发布于 [Bilibili](https://www.bilibili.com/video/BV1w8411y7uj)。
    - 由于 OLSS 需要额外训练，我们未在本项目中实现它。

- **2023年8月29日** 我们提出 DiffSynth，一个视频合成框架。
  - [项目页面](https://ecnu-cilab.github.io/DiffSynth.github.io/)。
  - 源代码已发布在 [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth)。
  - 技术报告（ECML PKDD 2024）已发布于 [arXiv](https://arxiv.org/abs/2308.03463)。

</details>
