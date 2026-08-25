# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/)
[![Discord](https://badgen.net//discord/members/Mm9suEeUDc)](https://discord.gg/Mm9suEeUDc)

[切换到中文版](./README_zh.md)

## Introduction

Welcome to the magical world of Diffusion models! DiffSynth-Studio is an open-source Diffusion model engine developed and maintained by the [ModelScope Community](https://www.modelscope.cn/) team. We hope to foster technological innovation through framework construction, aggregate the power of the open-source community, and explore the rich capabilities of generative model technology!

Framework features:

* [Model Support](#all-supported-models): The framework integrates mainstream open-source Diffusion models, covering image generation, video generation, audio generation, and image quality metrics models.
* [VRAM Management](https://diffsynth-studio-doc.readthedocs.io/en/latest/Pipeline_Usage/VRAM_management.html): Dynamically schedules model parameters across disk, memory, and VRAM, allowing consumer-grade GPUs with low VRAM to run inference with large models.
* [Parameter Quantization](https://diffsynth-studio-doc.readthedocs.io/en/latest/Pipeline_Usage/Quantization.html): Converts model parameters to quantized precisions such as NF4 and INT8, significantly reducing the VRAM requirements of model inference and LoRA training.
* [Arbitrary Training](https://diffsynth-studio-doc.readthedocs.io/en/latest/Pipeline_Usage/Model_Training.html): Almost every model that supports inference also supports training, whether base models, LoRAs, or any Adapter models with additional inputs.
* [Split Training](https://diffsynth-studio-doc.readthedocs.io/en/latest/Training/Split_Training.html): Uses a computational graph inference engine to track every variable in the Pipeline, splitting the training process into two stages for efficient training.

References:

* Developer documentation (for humans): [中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/README.html)、[English version](https://diffsynth-studio-doc.readthedocs.io/en/latest/README.html)
* Agent Skills (for AI): [DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills)

See also:

* [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI): A lightweight LoRA training tool built on DiffSynth-Studio, enabling LoRA training of models on consumer-grade GPUs.
* [ModelScope AIGC Zone (for Chinese users)](https://modelscope.cn/aigc/home): Productized features powered by DiffSynth-Studio as the core inference and training engine; experience the rich potential of the open-source model ecosystem.
* [ModelScope Civision (for global users)](https://modelscope.ai/civision/home): Unlock the vast potential of the open-source model ecosystem through productized capabilities powered by DiffSynth-Studio.

## Update History

> DiffSynth-Studio has undergone major version updates, and some old features are no longer maintained. If you need to use old features, please switch to the [last historical version](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3) before the major version update.

> Currently, the development personnel of this project are limited, with most of the work handled by [Artiprocher](https://github.com/Artiprocher) and [mi804](https://github.com/mi804). Therefore, the progress of new feature development will be relatively slow, and the speed of responding to and resolving issues is limited. We apologize for this and ask developers to understand.

- **August 25, 2026** We have open-sourced [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI), enabling one-click private deployment of LoRA training services. Combined with the model quantization feature, you can train large models even with consumer-grade GPUs.

- **August 19, 2026** We have released the model quantization feature. It provides a unified `QuantizeConfig` entry point supporting multiple quantization backends including bitsandbytes, torchao, and comfy-kitchen, with capabilities such as online quantization, loading pre-quantized weights, mixed quantization, saving quantized models, and quantization + LoRA training. For details, please refer to the [documentation](/docs/en/Pipeline_Usage/Quantization.md).

- **August 17, 2026** MiniMax-Music3 open-sourced, welcome a new member to the audio model family! Support includes text-to-music generation and low VRAM inference. For details, please refer to the [documentation](/docs/en/Model_Details/MiniMax-Music3.md) and [example code](/examples/minimax_music3/).

- **August 7, 2026** We add support for Wan-Animate-2 in the Wan series. Given a reference image and a driving video, it makes the reference character perform the motions in the driving video, generating high-quality character animation, with both standard and distilled variants. For details, please refer to the [documentation](/docs/en/Model_Details/Wan.md) and [example code](/examples/wanvideo/).

- **August 3, 2026** MiniMax-H3 open-sourced, welcome a new member to the video model family! Support includes text-to-video-audio generation, keyframe-guided generation, reference-driven generation, low VRAM inference, and NF4-quantized inference. For details, please refer to the [documentation](/docs/en/Model_Details/MiniMax-H3.md) and [example code](/examples/minimax_h3/).

- **July 28, 2026** LingBot-Video open-sourced, welcome a new member to the video model family! This release includes two variants, Dense-1.3B and MoE-30B-A3B (30B total parameters, ~3B active per token), both supporting text-to-video, image-to-video and text-to-image generation, low VRAM inference, and LoRA / full training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/LingBot-Video.md) and [example code](/examples/lingbot_video/). Huge thanks to [NancyFyong](https://github.com/NancyFyong) for contributing the integration of this model!

- **July 21, 2026** We have open-sourced [DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills). This is a composable collection of Agent Skills that automates the entire workflow of integrating external diffusion models into DiffSynth-Studio, significantly improving the standardization and efficiency of model integration. Get started with the [example](https://www.modelscope.cn/skills/DiffSynth-Studio/diffsynth-integrator/file/view/master/example.md?status=1)!

<details>
<summary>More</summary>

- **June 29, 2026** Boogu-Image open-sourced. Support includes text-to-image generation, image editing, low VRAM inference, and training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/Boogu-Image.md) and [example code](/examples/boogu_image/).

- **June 24, 2026** Krea-2 is now open-source, and we have provided full support. For more details, please refer to the [documentation](/docs/en/Model_Details/Krea-2.md) and [example code](/examples/krea2/).

- **June 16, 2026**: We have added a new Template model for ACE-Step: [vocals2music](https://www.modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-vocals2music). For more details, please refer to the [documentation](/docs/en/Model_Details/ACE-Step.md) and [example code](/examples/ace_step/).

- **June 15, 2026** We have open-sourced Image-to-LoRA V2, compressing the hours-long training process for image style LoRAs into a single model inference step, thereby exploring a new paradigm for LoRA model training. The [technical report](https://arxiv.org/abs/2606.13809) has been released. This release includes three models:
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2): Adapted for the Z-Image model
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2): Adapted for the FLUX.2-klein-base-4B model
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2): Adapted for the Hidream-O1-Image model

- **June 5, 2026** Ideogram 4 open-sourced. Support includes text-to-image inference. For details, please refer to the [documentation](/docs/en/Model_Details/Ideogram-4.md) and [example code](/examples/ideogram4/).

- **May 21, 2026**: Added support for image quality metrics models, including FID, CLIP, Aesthetic, PickScore, ImageReward, HPSv2, and HPSv3. For details, refer to the [documentation](/docs/en/Model_Details/Image-Quality-Metrics.md) and [example code](/examples/image_quality_metric/).

- **May 18, 2026** Added **CPU Offload Training** support. By moving model weights layer-by-layer between CPU and GPU, it significantly reduces GPU VRAM usage during training, enabling LoRA training of large models even on consumer-grade GPUs, compatible with all models. Simply add `--enable_model_cpu_offload` to your training command to enable (currently supports single-GPU training only). For details, see the [documentation](/docs/en/Training/Offload_Training.md).

- **May 14, 2026** HiDream-O1-Image open-sourced, welcome a new member to the image model family! Support includes text-to-image generation, image editing, low VRAM inference, and training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/HiDream-O1-Image.md) and [example code](/examples/hidream_o1_image/).

- **April 28, 2026** We released Diffusion Templates, a plugin framework designed for Diffusion models that significantly lowers the barrier to training controllable generative models. Let's explore this cutting-edge technology together!
    * Open-source code: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
    * Technical report: [arXiv](https://arxiv.org/abs/2604.24351)
    * Project homepage: [GitHub](https://modelscope.github.io/diffusion-templates-web/)
    * Documentation: [English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html) | [Chinese Version](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
    * Online demo: [ModelScope](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
    * Model collections: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates) | [ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates) | [HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
    * Datasets: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2) | [ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/ImagePulseV2) | [HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

- **April 27, 2026** We support ACE-Step-1.5! Support includes text-to-music generation, low VRAM inference, and LoRA training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/ACE-Step.md) and [example code](/examples/ace_step/).

- **April 27, 2026**: We have reinstated support for the Stable Diffusion v1.5 and SDXL models, providing academic research support exclusively for these two model types.

- **April 14, 2026** JoyAI-Image open-sourced, welcome a new member to the image editing model family! Support includes instruction-guided image editing, low VRAM inference, and training capabilities. For details, please refer to the [documentation](/docs/en/Model_Details/JoyAI-Image.md) and [example code](/examples/joyai_image/).


- **March 19, 2026**: Added support for [openmoss/MOVA-720p](https://modelscope.cn/models/openmoss/MOVA-720p) and [openmoss/MOVA-360p](https://modelscope.cn/models/openmoss/MOVA-360p) models, including training and inference capabilities. [Documentation](/docs/en/Model_Details/Wan.md) and [example code](/examples/mova/) are now available.

- **March 12, 2026**: We have added support for the [LTX-2.3](https://modelscope.cn/models/Lightricks/LTX-2.3) audio-video generation model. The features includes text-to-audio/video, image-to-audio/video, IC-LoRA control, audio-to-video, and audio-video inpainting. We have supported the complete inference and training functionalities. For details, please refer to the [documentation](/docs/en/Model_Details/LTX-2.md) and [code](/examples/ltx2/).

- **March 3, 2026**: We released the [DiffSynth-Studio/Qwen-Image-Layered-Control-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control-V2) model, which is an updated version of Qwen-Image-Layered-Control. In addition to the originally supported text-guided functionality, it adds brush-controlled layer separation capabilities.

- **March 2, 2026** Added support for [Anima](https://modelscope.cn/models/circlestone-labs/Anima). For details, please refer to the [documentation](docs/en/Model_Details/Anima.md). This is an interesting anime-style image generation model. We look forward to its future updates.

- **February 26, 2026** Added full and lora training support for the LTX-2 audio-video generation model. See the [documentation](/docs/en/Model_Details/LTX-2.md) for details.

- **February 10, 2026** Added inference support for the LTX-2 audio-video generation model. See the [documentation](/docs/en/Model_Details/LTX-2.md) for details. Support for model training will be implemented in the future.

- **February 2, 2026** The first document of the Research Tutorial series is now available, guiding you through training a small 0.1B text-to-image model from scratch. For details, see the [documentation](/docs/en/Research_Tutorial/train_from_scratch.md) and [model](https://modelscope.cn/models/DiffSynth-Studio/AAAMyModel). We hope DiffSynth-Studio can evolve into a more powerful training framework for Diffusion models.

- **January 27, 2026**: [Z-Image](https://modelscope.cn/models/Tongyi-MAI/Z-Image) is released, and our [Z-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) model is released concurrently. You can use it in [ModelScope Studios](https://modelscope.cn/studios/DiffSynth-Studio/Z-Image-i2L). For details, see the [documentation](/docs/zh/Model_Details/Z-Image.md).

- **January 19, 2026**: Added support for [FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) and [FLUX.2-klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) models, including training and inference capabilities. [Documentation](/docs/en/Model_Details/FLUX2.md) and [example code](/examples/flux2/) are now available.

- **January 12, 2026**: We trained and open-sourced a text-guided image layer separation model ([Model Link](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control)). Given an input image and a textual description, the model isolates the image layer corresponding to the described content. For more details, please refer to our blog post ([Chinese version](https://modelscope.cn/learn/4938), [English version](https://huggingface.co/blog/kelseye/qwen-image-layered-control)).

- **December 24, 2025**: Based on Qwen-Image-Edit-2511, we trained an In-Context Editing LoRA model ([Model Link](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)). This model takes three images as input (Image A, Image B, and Image C), and automatically analyzes the transformation from Image A to Image B, then applies the same transformation to Image C to generate Image D. For more details, please refer to our blog post ([Chinese version](https://mp.weixin.qq.com/s/41aEiN3lXKGCJs1-we4Q2g), [English version](https://huggingface.co/blog/kelseye/qwen-image-edit-2511-icedit-lora)).

- **December 9, 2025** We release a wild model based on DiffSynth-Studio 2.0: [Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L) (Image-to-LoRA). This model takes an image as input and outputs a LoRA. Although this version still has significant room for improvement in terms of generalization, detail preservation, and other aspects, we are open-sourcing these models to inspire more innovative research. For more details, please refer to our [blog](https://huggingface.co/blog/kelseye/qwen-image-i2l).

- **December 4, 2025** DiffSynth-Studio 2.0 released! Many new features online
  - [Documentation](/docs/en/README.md) online: Our documentation is still continuously being optimized and updated
  - [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md) module upgraded, supporting layer-level disk offload, releasing both memory and VRAM simultaneously
  - New model support
    - Z-Image Turbo: [Model](https://www.modelscope.ai/models/Tongyi-MAI/Z-Image-Turbo), [Documentation](/docs/en/Model_Details/Z-Image.md), [Code](/examples/z_image/)
    - FLUX.2-dev: [Model](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev), [Documentation](/docs/en/Model_Details/FLUX2.md), [Code](/examples/flux2/)
  - Training framework upgrade
    - [Split Training](/docs/en/Training/Split_Training.md): Supports automatically splitting the training process into two stages: data processing and training (even for training ControlNet or any other model). Computations that do not require gradient backpropagation, such as text encoding and VAE encoding, are performed during the data processing stage, while other computations are handled during the training stage. Faster speed, less VRAM requirement.
    - [Differential LoRA Training](/docs/en/Training/Differential_LoRA.md): This is a training technique we used in [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), now available for LoRA training of any model.
    - [FP8 Training](/docs/en/Training/FP8_Precision.md): FP8 can be applied to any non-training model during training, i.e., models with gradients turned off or gradients that only affect LoRA weights.

- **November 4, 2025** Supported the [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) model, which is trained based on Wan 2.1 and supports generating corresponding actions based on reference videos.

- **October 30, 2025** Supported the [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) model, which supports text-to-video, image-to-video, and video continuation. This model uses the Wan framework for inference and training in this project.

- **October 27, 2025** Supported the [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) model, adding another member to the Wan model ecosystem.

- **September 23, 2025** [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) released! This model was jointly developed and open-sourced by us and Taobao Experience Design Team. Built upon Qwen-Image, the model is specifically designed for e-commerce poster scenarios, supporting precise partition layout control. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py).

- **September 9, 2025** Our training framework supports various training modes. Currently adapted for Qwen-Image, in addition to the standard SFT training mode, Direct Distill is now supported. Please refer to [our sample code](./examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh). This feature is experimental, and we will continue to improve it to support more comprehensive model training functions.

- **August 28, 2025** We support Wan2.2-S2V, an audio-driven cinematic video generation model. See [./examples/wanvideo/](./examples/wanvideo/).

- **August 21, 2025** [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) released! Compared to the V1 version, the training dataset has been changed to [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset), so the generated images better conform to Qwen-Image's own image distribution and style. Please refer to [our sample code](./examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py).

- **August 21, 2025** We open-sourced the [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) structural control LoRA model, adopting the In Context technical route, supporting multiple categories of structural control conditions, including canny, depth, lineart, softedge, normal, and openpose. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py).

- **August 20, 2025** We open-sourced the [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) model, improving the editing effect of Qwen-Image-Edit on low-resolution image inputs. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py)

- **August 19, 2025** Qwen-Image-Edit open-sourced, welcome a new member to the image editing model family!

- **August 18, 2025** We trained and open-sourced the Qwen-Image inpainting ControlNet model [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint). The model structure adopts a lightweight design. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py).

- **August 15, 2025** We open-sourced the [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset) dataset. This is an image dataset generated using the Qwen-Image model, containing 160,000 `1024 x 1024` images. It includes general, English text rendering, and Chinese text rendering subsets. We provide annotations for image descriptions, entities, and structural control images for each image. Developers can use this dataset to train Qwen-Image models' ControlNet and EliGen models. We aim to promote technological development through open-sourcing!

- **August 13, 2025** We trained and open-sourced the Qwen-Image ControlNet model [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth). The model structure adopts a lightweight design. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py).

- **August 12, 2025** We trained and open-sourced the Qwen-Image ControlNet model [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny). The model structure adopts a lightweight design. Please refer to [our sample code](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py).

- **August 11, 2025** We open-sourced the distilled acceleration model [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA) for Qwen-Image, following the same training process as [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full), but the model structure has been modified to LoRA, thus being better compatible with other open-source ecosystem models.

- **August 7, 2025** We open-sourced the entity control LoRA model [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen) for Qwen-Image. Qwen-Image-EliGen can achieve entity-level controlled text-to-image generation. Technical details can be found in [the paper](https://arxiv.org/abs/2501.01097). Training dataset: [EliGenTrainSet](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet).

- **August 5, 2025** We open-sourced the distilled acceleration model [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) for Qwen-Image, achieving approximately 5x acceleration.

- **August 4, 2025** Qwen-Image open-sourced, welcome a new member to the image generation model family!

- **August 1, 2025** [FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) open-sourced, a text-to-image model focused on aesthetic photography. We provided comprehensive support in a timely manner, including low VRAM layer-by-layer offload, LoRA training, and full training. For more details, please refer to [./examples/flux/](./examples/flux/).

- **July 28, 2025** Wan 2.2 open-sourced. We provided comprehensive support in a timely manner, including low VRAM layer-by-layer offload, FP8 quantization, sequence parallelism, LoRA training, and full training. For more details, please refer to [./examples/wanvideo/](./examples/wanvideo/).

- **July 11, 2025** We propose Nexus-Gen, a unified framework that combines the language reasoning capabilities of Large Language Models (LLMs) with the image generation capabilities of diffusion models. This framework supports seamless image understanding, generation, and editing tasks.
  - Paper: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
  - GitHub Repository: https://github.com/modelscope/Nexus-Gen
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
  - Training Dataset: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
  - Online Experience: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

- **June 15, 2025** ModelScope's official evaluation framework [EvalScope](https://github.com/modelscope/evalscope) now supports text-to-image generation evaluation. Please refer to the [best practices](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/t2i_eval.html) guide to try it out.

- **March 25, 2025** Our new open-source project [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) is now open-sourced! Focused on stable model deployment, targeting industry, providing better engineering support, higher computational performance, and more stable features.

- **March 31, 2025** We support InfiniteYou, a face feature preservation method for FLUX. More details can be found in [./examples/InfiniteYou/](./examples/InfiniteYou/).

- **March 13, 2025** We support HunyuanVideo-I2V, the image-to-video generation version of Tencent's open-source HunyuanVideo. More details can be found in [./examples/HunyuanVideo/](./examples/HunyuanVideo/).

- **February 25, 2025** We support Wan-Video, a series of state-of-the-art video synthesis models open-sourced by Alibaba. See [./examples/wanvideo/](./examples/wanvideo/).

- **February 17, 2025** We support [StepVideo](https://modelscope.cn/models/stepfun-ai/stepvideo-t2v/summary)! Advanced video synthesis model! See [./examples/stepvideo](./examples/stepvideo/).

- **December 31, 2024** We propose EliGen, a new framework for entity-level controlled text-to-image generation, supplemented with an inpainting fusion pipeline, extending its capabilities to image inpainting tasks. EliGen can seamlessly integrate existing community models such as IP-Adapter and In-Context LoRA, enhancing their versatility. For more details, see [./examples/EntityControl](./examples/EntityControl/).
  - Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
  - Online Experience: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
  - Training Dataset: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

- **December 19, 2024** We implemented advanced VRAM management for HunyuanVideo, enabling video generation with resolutions of 129x720x1280 on 24GB VRAM or 129x512x384 on just 6GB VRAM. More details can be found in [./examples/HunyuanVideo/](./examples/HunyuanVideo/).

- **December 18, 2024** We propose ArtAug, a method to improve text-to-image models through synthesis-understanding interaction. We trained an ArtAug enhancement module for FLUX.1-dev in LoRA format. This model incorporates the aesthetic understanding of Qwen2-VL-72B into FLUX.1-dev, thereby improving the quality of generated images.
  - Paper: https://arxiv.org/abs/2412.12888
  - Example: https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/ArtAug
  - Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
  - Demo: [ModelScope](https://modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0), HuggingFace (coming soon)

- **October 25, 2024** We provide extensive FLUX ControlNet support. This project supports many different ControlNet models and can be freely combined, even if their structures are different. Additionally, ControlNet models are compatible with high-resolution optimization and partition control technologies, enabling very powerful controllable image generation. See [`./examples/ControlNet/`](./examples/ControlNet/).

- **October 8, 2024** We released extended LoRAs based on CogVideoX-5B and ExVideo. You can download this model from [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1) or [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1).

- **August 22, 2024** This project now supports CogVideoX-5B. See [here](/examples/video_synthesis/). We provide several interesting features for this text-to-video model, including:
  - Text-to-video
  - Video editing
  - Self super-resolution
  - Video interpolation

- **August 22, 2024** We implemented an interesting brush feature that supports all text-to-image models. Now you can create stunning images with the assistance of AI using the brush!
  - Use it in our [WebUI](#usage-in-webui).

- **August 21, 2024** DiffSynth-Studio now supports FLUX.
  - Enable CFG and high-resolution inpainting to improve visual quality. See [here](/examples/image_synthesis/README.md)
  - LoRA, ControlNet, and other addon models will be released soon.

- **June 21, 2024** We propose ExVideo, a post-training fine-tuning technique aimed at enhancing the capabilities of video generation models. We extended Stable Video Diffusion to achieve long video generation of up to 128 frames.
  - [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - Source code has been released in this repository. See [`examples/ExVideo`](./examples/ExVideo/).
  - Model has been released at [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) and [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1).
  - Technical report has been released at [arXiv](https://arxiv.org/abs/2406.14130).
  - You can try ExVideo in this [demo](https://huggingface.co/spaces/modelscope/ExVideo-SVD-128f-v1)!

- **June 13, 2024** DiffSynth Studio has migrated to ModelScope. The development team has also transitioned from "me" to "us". Of course, I will still participate in subsequent development and maintenance work.

- **January 29, 2024** We propose Diffutoon, an excellent cartoon coloring solution.
  - [Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
  - Source code has been released in this project.
  - Technical report (IJCAI 2024) has been released at [arXiv](https://arxiv.org/abs/2401.16224).

- **December 8, 2023** We decided to initiate a new project aimed at unleashing the potential of diffusion models, especially in video synthesis. The development work of this project officially began.

- **November 15, 2023** We propose FastBlend, a powerful video deflickering algorithm.
  - sd-webui extension has been released at [GitHub](https://github.com/Artiprocher/sd-webui-fastblend).
  - Demonstration videos have been showcased on Bilibili, including three tasks:
    - [Video Deflickering](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [Video Interpolation](https://www.bilibili.com/video/BV1Lw411m71p)
    - [Image-Driven Video Rendering](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - Technical report has been released at [arXiv](https://arxiv.org/abs/2311.09265).
  - Unofficial ComfyUI extensions developed by other users have been released at [GitHub](https://github.com/AInseven/ComfyUI-fastblend).

- **October 1, 2023** We released an early version of the project named FastSDXL. This was an initial attempt to build a diffusion engine.
  - Source code has been released at [GitHub](https://github.com/Artiprocher/FastSDXL).
  - FastSDXL includes a trainable OLSS scheduler to improve efficiency.
    - The original repository of OLSS is located [here](https://github.com/alibaba/EasyNLP/tree/master/diffusion/olss_scheduler).
    - Technical report (CIKM 2023) has been released at [arXiv](https://arxiv.org/abs/2305.14677).
    - Demonstration video has been released at [Bilibili](https://www.bilibili.com/video/BV1w8411y7uj).
    - Since OLSS requires additional training, we did not implement it in this project.

- **August 29, 2023** We propose DiffSynth, a video synthesis framework.
  - [Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/).
  - Source code has been released at [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth).
  - Technical report (ECML PKDD 2024) has been released at [arXiv](https://arxiv.org/abs/2308.03463).

</details>

## Installation

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

For more installation methods and instructions for non-NVIDIA GPUs, please refer to the [Installation Guide](/docs/en/Pipeline_Usage/Setup.md).

<details>
<summary>Download Source Configuration</summary>

> Before model inference and training, you can configure the model download source and other options through [environment variables](/docs/en/Pipeline_Usage/Environment_Variables.md).
> 
> This project downloads models from [ModelScope](https://modelscope.cn/) by default. For users outside China, you can download models from the [ModelScope International](https://modelscope.ai) site with the following configuration:
> 
> ```shell
> export MODELSCOPE_ENDPOINT=https://modelscope.ai
> ```
> 
> To download models from [HuggingFace](https://huggingface.co/), please modify the [environment variables](/docs/en/Pipeline_Usage/Environment_Variables.md). Note that model IDs may differ across platforms:
> 
> ```shell
> export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"
> ```

</details>

## Basic Framework

DiffSynth-Studio redesigns the inference and training pipelines for mainstream Diffusion models (including FLUX, Wan, etc.), enabling efficient memory management and flexible model training.

Quick start: experience popular and the latest models:

| Architecture | Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
|-|-|-|-|-|-|-|-|
| MiniMax-H3 | [MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft) | [code](/examples/ace_step/model_inference/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py) |
| Z-Image | [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [code](/examples/z_image/model_inference/Z-Image-Turbo.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo.py) |
| Krea-2 | [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) | [code](/examples/krea2/model_inference/Krea-2-Raw.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Raw.py) | [code](/examples/krea2/model_training/full/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Raw.py) | [code](/examples/krea2/model_training/lora/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Raw.py) |
| Krea-2 | [krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo) | [code](/examples/krea2/model_inference/Krea-2-Turbo.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/full/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/lora/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py) |

Model overview:

- Image generation
    - Boogu-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Boogu-Image.html), [Example code](/examples/boogu_image/)
    - Krea-2: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Krea-2.html), [Example code](/examples/krea2/)
    - Ideogram 4: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Ideogram-4.html), [Example code](/examples/ideogram4/)
    - HiDream-O1-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/HiDream-O1-Image.html), [Example code](/examples/hidream_o1_image/)
    - JoyAI-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/JoyAI-Image.html), [Example code](/examples/joyai_image/)
    - ERNIE-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/ERNIE-Image.html), [Example code](/examples/ernie_image/)
    - FLUX.2: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/FLUX2.html), [Example code](/examples/flux2/)
    - Z-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Z-Image.html), [Example code](/examples/z_image/)
    - Anima: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Anima.html), [Example code](/examples/anima/)
    - Qwen-Image: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Qwen-Image.html), [Example code](/examples/qwen_image/)
    - FLUX.1: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/FLUX.html), [Example code](/examples/flux/)
    - Stable Diffusion XL: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Stable-Diffusion-XL.html), [Example code](/examples/stable_diffusion_xl/)
    - Stable Diffusion: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Stable-Diffusion.html), [Example code](/examples/stable_diffusion/)
- Video generation
    - MiniMax-H3: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/MiniMax-H3.html), [Example code](/examples/minimax_h3/)
    - LingBot-Video: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/LingBot-Video.html), [Example code](/examples/lingbot_video/)
    - LTX-2: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/LTX-2.html), [Example code](/examples/ltx2/)
    - Wan: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html), [Example code](/examples/wanvideo/)
- Audio generation
    - MiniMax-Music3: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/MiniMax-Music3.html), [Example code](/examples/minimax_music3/)
    - ACE-Step: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/ACE-Step.html), [Example code](/examples/ace_step/)
- Evaluation models: [Documentation](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Image-Quality-Metrics.html), [Example code](/examples/image_quality_metric/)

[View all supported models](#all-supported-models)

## Innovative Achievements

We believe that a well-developed open-source code framework can lower the threshold for technical exploration. We have achieved many interesting technologies based on this codebase. Perhaps you also have many wild ideas, and with DiffSynth-Studio, you can quickly realize these ideas.

<details>

<summary>TreeAdapter: A Model System Built from Structured LoRAs</summary>

> A model system composed of 10,000+ LoRAs challenges the precise generation of tens of thousands of rare species.

* Paper: [TreeAdapter: Hierarchical Taxonomy-Guided Adapter Composition for Fine-Grained Species Image Generation](https://arxiv.org/abs/2607.24215)
* Model: [ModelScope](https://modelscope.cn/models/DiffSynth-Studio/TreeAdapter-KleinBase4B)

![Image](https://github.com/user-attachments/assets/1b461e0f-60aa-4b38-a44d-d1646cbbbc75)

</details>


<details>

<summary>Image-to-LoRA: Compressing Model Training into Model Inference</summary>

> True Meta Learning: feed a dataset of images into one end of the model, and the trained LoRA model comes out the other end.

* Paper: [Compressing Image Style Training into a Single Model Forward](https://arxiv.org/abs/2606.13809)
* Model:
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2): Adapted for the Z-Image model
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2): Adapted for the FLUX.2-klein-base-4B model
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2): Adapted for the Hidream-O1-Image model

|Input example 1|Output example 1|Input example 2|Output example 2|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/0a1fd252-851f-414e-af24-4c656ab54277)|![Image](https://github.com/user-attachments/assets/96259993-e732-424d-bf07-9ca1ede27890)|![Image](https://github.com/user-attachments/assets/a78573a0-c2cf-4e33-ac21-276078e8cad3)|![Image](https://github.com/user-attachments/assets/8177e883-cfef-4e38-a528-cdef01a9f9b8)|

</details>


<details>

<summary>Diffusion-Templates: A Plugin-Based Controllable Generation Framework</summary>

> One framework that turns every controllable generation capability into a plugin, allowing multiple models to combine and emerge with rich generation capabilities.

* Paper: [Diffusion Templates: A Unified Plugin Framework for Controllable Diffusion](https://arxiv.org/abs/2604.24351)
* Project homepage: [GitHub](https://modelscope.github.io/diffusion-templates-web/)
* Documentation: [English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html) | [Chinese Version](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* Online demo: [ModelScope](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* Model collections: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates) | [ModelScope International](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates) | [HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* Datasets: [ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2) | [ModelScope International](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2) | [HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

|Reference image|Local editing|Style transfer|Sharpness enhancement|
|-|-|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_reference.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Brightness_Edit_Inpaint.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Controlnet_Edit_SoftRGB.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Upscaler_Sharpness.png)|

</details>



<details>

<summary>Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation</summary>

> Trade inference time for higher quality of generated content.

- Paper: [Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation
](https://arxiv.org/abs/2602.03208)
- Sample Code: [/docs/en/Research_Tutorial/inference_time_scaling.md](/docs/en/Research_Tutorial/inference_time_scaling.md)

|FLUX.1-dev|FLUX.1-dev + SES|Qwen-Image|Qwen-Image + SES|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/5be15dc6-2805-4822-b04c-2573fc0f45f0)|![Image](https://github.com/user-attachments/assets/e71b8c20-1629-41d9-b0ff-185805c1da4e)|![Image](https://github.com/user-attachments/assets/7a73c968-133a-4545-9aa2-205533861cd4)|![Image](https://github.com/user-attachments/assets/c8390b22-14fe-48a0-a6e6-d6556d31235e)|

</details>


<details>

<summary>VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers</summary>

> Transform image 3 into image 4 based on the change from image 1 to image 2 — the emergent capability of image editing models.

- Paper: [VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers
](https://arxiv.org/abs/2602.03210)
- Sample code: [/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)

|Example 1|Example 2|Query|Output|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/380d2670-47bf-41cd-b5c9-37110cc4a943)|![Image](https://github.com/user-attachments/assets/7ceaf345-0992-46e6-b38f-394c2065b165)|![Image](https://github.com/user-attachments/assets/f7c26c21-6894-4d9e-b570-f1d44ca7c1de)|![Image](https://github.com/user-attachments/assets/c2bebe3b-5984-41ba-94bf-9509f6a8a990)|

</details>


<details>

<summary>AttriCtrl: Attribute Intensity Control for Image Generation Models</summary>

> Numerical attributes can also precisely control image generation models.

- Paper: [AttriCtrl: Fine-Grained Control of Aesthetic Attribute Intensity in Diffusion Models](https://arxiv.org/abs/2508.02151)
- Sample Code: [/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py](/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/AttriCtrl-FLUX.1-Dev)

|brightness scale = 0.1|brightness scale = 0.3|brightness scale = 0.5|brightness scale = 0.7|brightness scale = 0.9|
|-|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/e74b32a5-5b2e-4c87-9df8-487c0f8366b7)|![Image](https://github.com/user-attachments/assets/bfe8bec2-9e55-493d-9a26-7e9cce28e03d)|![Image](https://github.com/user-attachments/assets/b099dfe3-ff1f-4b96-894c-d48bbe92db7a)|![Image](https://github.com/user-attachments/assets/0a6b2982-deab-4b0d-91ad-888782de01c9)|![Image](https://github.com/user-attachments/assets/fcecb755-7d03-4020-b83a-13ad2b38705c)|

</details>


<details>

<summary>AutoLoRA: Automated LoRA Retrieval and Fusion</summary>

> LoRA is a product that unifies needs and solutions — how can we make better use of these LoRAs?

- Paper: [AutoLoRA: Automatic LoRA Retrieval and Fine-Grained Gated Fusion for Text-to-Image Generation](https://arxiv.org/abs/2508.02107)
- Sample Code: [/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py](/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev)

||[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)|
|-|-|-|-|-|
|[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)                              |![Image](https://github.com/user-attachments/assets/01c54d5a-4f00-4c2e-982a-4ec0a4c6a6e3)|![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|
|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)                       |![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/43720a9f-aa27-4918-947d-545389375d46)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|
|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)  |![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/041a3f9a-c7b4-4311-8582-cb71a7226d80)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|
|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)                          |![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|![Image](https://github.com/user-attachments/assets/a640fd54-3192-49a0-9281-b43d9ba64f09)|

</details>


<details>

<summary>Nexus-Gen: Unified Architecture for Image Understanding, Generation, and Editing</summary>

> What happens when a single model combines image understanding, generation, and editing capabilities?

- Detailed Page: https://github.com/modelscope/Nexus-Gen
- Paper: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
- Dataset: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
- Online Experience: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

![](https://github.com/modelscope/Nexus-Gen/raw/main/assets/illustrations/gen_edit.jpg)

</details>


<details>

<summary>ArtAug: Aesthetic Enhancement for Image Generation Models</summary>

> A single LoRA that significantly enhances detail and aesthetics.

- Detailed Page: [./examples/ArtAug/](./examples/ArtAug/)
- Paper: [ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction](https://arxiv.org/abs/2412.12888)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ArtAug-lora-FLUX.1dev-v1)
- Online Experience: [ModelScope AIGC Tab](https://www.modelscope.cn/aigc/imageGeneration?tab=advanced&versionId=7228&modelType=LoRA&sdVersion=FLUX_1&modelUrl=modelscope%3A%2F%2FDiffSynth-Studio%2FArtAug-lora-FLUX.1dev-v1%3Frevision%3Dv1.0)

|FLUX.1-dev|FLUX.1-dev + ArtAug LoRA|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/e1d5c505-b423-45fe-be01-25c2758f5417)|![image_1_enhance](https://github.com/user-attachments/assets/335908e3-d0bd-41c2-9d99-d10528a2d719)|

</details>


<details>

<summary>EliGen: Precise Image Partition Control</summary>

> How can region-based layers control the position of content in an image?

- Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
- Sample Code: [/examples/flux/model_inference/FLUX.1-dev-EliGen.py](/examples/flux/model_inference/FLUX.1-dev-EliGen.py)
- Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
- Online Experience: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
- Dataset: [EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

|Entity Control Region|Generated Image|
|-|-|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|

</details>


<details>

<summary>ExVideo: Extended Training for Video Generation Models</summary>

> If a video generation model can only generate 25 frames, how can we make it generate longer videos?

- Project Page: [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- Paper: [ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- Sample Code: Please refer to the [older version](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/ExVideo)
- Model: [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>


<details>

<summary>Diffutoon: High-Resolution Anime-Style Video Rendering</summary>

> I don't care what you say, I just love anime!

- Project Page: [Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- Paper: [Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models](https://arxiv.org/abs/2401.16224)
- Sample Code: Please refer to the [older version](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/Diffutoon)

The Diffutoon examples are from the pre-2.0 codebase and are not included in the current `main` branch. Use the older source tree above with the corresponding older version of DiffSynth-Studio.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

</details>


<details>

<summary>DiffSynth: The Original Version of This Project</summary>

> In the era before video generation models, how could image generation models be used to process videos?

- Project Page: [Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/)
- Paper: [DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463)
- Sample Code: Please refer to the [older version](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/diffsynth)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

</details>

## Contact Us

|Discord：https://discord.gg/Mm9suEeUDc|
|-|
|<img width="160" height="160" alt="Image" src="https://github.com/user-attachments/assets/29bdc97b-e35d-4fea-88d6-32e35182e458" />|


## All Supported Models

| Architecture | Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
|-|-|-|-|-|-|-|-|
| MiniMax-Music3 | [MiniMax/MiniMax-Music3](https://www.modelscope.cn/models/MiniMax/MiniMax-Music3) | [code](/examples/minimax_music3/model_inference/MiniMax-Music3.py) | [code](/examples/minimax_music3/model_inference_low_vram/MiniMax-Music3.py) | — | — | — | — |
| MiniMax-H3 | [MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py) |
| MiniMax-H3 | [MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py) |
| MiniMax-H3 | [MiniMax/MiniMax-H3: Retake](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Retake.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Retake.py) | - | - | - | - |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Pruned-Ref2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned int8_convrot](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: FL2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-FL2VA.py) |
| MiniMax-H3 | [Comfy-Org/MiniMax-H3: Ref2VA pruned fp8](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FP8-Pruned-Ref2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FP8-Pruned-Ref2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FP8-Pruned-Ref2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FP8-Pruned-Ref2VA.py) |
| MiniMax-H3 | [lightx2v/Minimax-h3-Turbo: FL2VA 4steps](https://www.modelscope.cn/models/lightx2v/Minimax-h3-Turbo) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-Turbo.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA-Turbo.py) | - | - | - | - |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-Text-Embeddings](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Text-Embeddings) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-Text-Embeddings.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Text-Embeddings.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-Text-Embeddings.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Text-Embeddings.py) | - | - |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_t2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_t2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_ti2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_ti2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-dense-1.3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2i.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2i.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-moe-30b-a3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-moe-30b-a3b_t2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-moe-30b-a3b_t2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-moe-30b-a3b_t2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_training/full/lingbot-video-moe-30b-a3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_full/lingbot-video-moe-30b-a3b_ti2v.py) | [code](/examples/lingbot_video/model_training/lora/lingbot-video-moe-30b-a3b_ti2v.sh) | [code](/examples/lingbot_video/model_training/validate_lora/lingbot-video-moe-30b-a3b_ti2v.py) |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2i.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2i.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: T2V + Refinement](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_t2v_refiner.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_t2v_refiner.py) | - | - | - | - |
| LingBot-Video | [Robbyant/lingbot-video-moe-30b-a3b: TI2V + Refinement](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) | [code](/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b_ti2v_refiner.py) | [code](/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b_ti2v_refiner.py) | - | - | - | - |
| ACE-Step | [ACE-Step/Ace-Step1.5](https://www.modelscope.cn/models/ACE-Step/Ace-Step1.5) | [code](/examples/ace_step/model_inference/Ace-Step1.5.py) | [code](/examples/ace_step/model_inference_low_vram/Ace-Step1.5.py) | [code](/examples/ace_step/model_training/full/Ace-Step1.5.sh) | [code](/examples/ace_step/model_training/validate_full/Ace-Step1.5.py) | [code](/examples/ace_step/model_training/lora/Ace-Step1.5.sh) | [code](/examples/ace_step/model_training/validate_lora/Ace-Step1.5.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-shift1](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift1) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-shift1.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift1.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-shift1.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift1.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-shift3](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-shift3) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-shift3.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-shift3.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-shift3.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-shift3.py) |
| ACE-Step | [ACE-Step/acestep-v15-turbo-continuous](https://www.modelscope.cn/models/ACE-Step/acestep-v15-turbo-continuous) | [code](/examples/ace_step/model_inference/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_training/full/acestep-v15-turbo-continuous.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-turbo-continuous.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-turbo-continuous.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-turbo-continuous.py) |
| ACE-Step | [ACE-Step/acestep-v15-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base.py) | [code](/examples/ace_step/model_training/full/acestep-v15-base.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-base.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-base.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-base.py) |
| ACE-Step | [ACE-Step/acestep-v15-base: CoverTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base-CoverTask.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base-CoverTask.py) | — | — | — | — |
| ACE-Step | [ACE-Step/acestep-v15-base: RepaintTask](https://www.modelscope.cn/models/ACE-Step/acestep-v15-base) | [code](/examples/ace_step/model_inference/acestep-v15-base-RepaintTask.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-base-RepaintTask.py) | — | — | — | — |
| ACE-Step | [ACE-Step/acestep-v15-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-sft) | [code](/examples/ace_step/model_inference/acestep-v15-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-sft.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-base](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-base) | [code](/examples/ace_step/model_inference/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-base.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-base.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-base.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-base.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft) | [code](/examples/ace_step/model_inference/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-turbo](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-turbo) | [code](/examples/ace_step/model_inference/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-turbo.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-turbo.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-turbo.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-turbo.py) |
| ACE-Step | [DiffSynth-Studio/acestep15xlsft-lora-music](https://www.modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-lora-music) | [code](/examples/ace_step/model_inference/acestep15xlsft-vocals2music.py) | [code](/examples/ace_step/model_inference_low_vram/acestep15xlsft-vocals2music.py) | [code](/examples/ace_step/model_training/full/acestep15xlsft-vocals2music.sh) | [code](/examples/ace_step/model_training/validate_full/acestep15xlsft-vocals2music.py) | - | - |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Base.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Base.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Base.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Base.py) |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Turbo](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Turbo.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Turbo.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Turbo.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Turbo.py) |
| Boogu-Image | [Boogu/Boogu-Image-0.1-Edit](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit) | [code](/examples/boogu_image/model_inference/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_training/full/Boogu-Image-0.1-Edit.sh) | [code](/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Edit.py) | [code](/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Edit.sh) | [code](/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Edit.py) |
| Krea-2 | [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) | [code](/examples/krea2/model_inference/Krea-2-Raw.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Raw.py) | [code](/examples/krea2/model_training/full/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Raw.py) | [code](/examples/krea2/model_training/lora/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Raw.py) |
| Krea-2 | [krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo) | [code](/examples/krea2/model_inference/Krea-2-Turbo.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/full/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/lora/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py) |
| Ideogram 4 | [ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8) | [code](/examples/ideogram4/model_inference/ideogram-4-fp8.py) | - | - | - | - | - |
| Ideogram 4 | [DiffSynth-Studio/ideogram-4-bf16-repackage](https://www.modelscope.cn/models/DiffSynth-Studio/ideogram-4-bf16-repackage) | [code](/examples/ideogram4/model_inference/ideogram-4-bf16-repackage.py) | [code](/examples/ideogram4/model_inference_low_vram/ideogram-4-bf16-repackage.py) | [code](/examples/ideogram4/model_training/full/Ideogram-4-bf16-repackage.sh) | - | [code](/examples/ideogram4/model_training/lora/Ideogram-4-bf16-repackage.sh) | [code](/examples/ideogram4/model_training/validate_lora/Ideogram-4-bf16-repackage.py) |
| HiDream-O1-Image | [HiDream-ai/HiDream-O1-Image](https://modelscope.cn/models/HiDream-ai/HiDream-O1-Image) | [code](/examples/hidream_o1_image/model_inference/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_training/full/HiDream-O1-Image.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image.py) | [code](/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image.sh) | [code](/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image.py) |
| HiDream-O1-Image | [HiDream-ai/HiDream-O1-Image-Dev](https://modelscope.cn/models/HiDream-ai/HiDream-O1-Image-Dev) | [code](/examples/hidream_o1_image/model_inference/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_training/full/HiDream-O1-Image-Dev.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image-Dev.py) | [code](/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image-Dev.sh) | [code](/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image-Dev.py) |
| HiDream-O1-Image | [DiffSynth-Studio/HidreamO1-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2) | [code](/examples/hidream_o1_image/model_inference/HidreamO1-i2L-v2.py) | [code](/examples/hidream_o1_image/model_inference_low_vram/HidreamO1-i2L-v2.py) | [code](/examples/hidream_o1_image/model_training/full/HidreamO1-i2L-v2.sh) | [code](/examples/hidream_o1_image/model_training/validate_full/HidreamO1-i2L-v2.py) | - | - |
| JoyAI-Image | [jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit) | [code](/examples/joyai_image/model_inference/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_inference_low_vram/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_training/full/JoyAI-Image-Edit.sh) | [code](/examples/joyai_image/model_training/validate_full/JoyAI-Image-Edit.py) | [code](/examples/joyai_image/model_training/lora/JoyAI-Image-Edit.sh) | [code](/examples/joyai_image/model_training/validate_lora/JoyAI-Image-Edit.py) |
| ERNIE-Image | [PaddlePaddle/ERNIE-Image](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image) | [code](/examples/ernie_image/model_inference/ERNIE-Image.py) | [code](/examples/ernie_image/model_inference_low_vram/ERNIE-Image.py) | [code](/examples/ernie_image/model_training/full/ERNIE-Image.sh) | [code](/examples/ernie_image/model_training/validate_full/ERNIE-Image.py) | [code](/examples/ernie_image/model_training/lora/ERNIE-Image.sh) | [code](/examples/ernie_image/model_training/validate_lora/ERNIE-Image.py) |
| ERNIE-Image | [PaddlePaddle/ERNIE-Image-Turbo](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image-Turbo) | [code](/examples/ernie_image/model_inference/ERNIE-Image-Turbo.py) | [code](/examples/ernie_image/model_inference_low_vram/ERNIE-Image-Turbo.py) | — | — | — | — |
| LTX-2 | [jd-opensource/JoyAI-Echo](https://modelscope.cn/models/jd-opensource/JoyAI-Echo) | [code](/examples/ltx2/model_inference/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_inference_low_vram/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_training/full/JoyAI-Echo-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/JoyAI-Echo-T2AV.py) | [code](/examples/ltx2/model_training/lora/JoyAI-Echo-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/JoyAI-Echo-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2.3-I2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2.3-I2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2.3-I2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-I2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-I2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-I2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2.3-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2.3-T2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2.3: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: A2V](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-A2V-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-A2V-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3: Retake](https://www.modelscope.cn/models/Lightricks/LTX-2.3) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-TwoStage-Retake.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-TwoStage-Retake.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-IC-LoRA-Union-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-IC-LoRA-Union-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control](https://www.modelscope.cn/models/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) | [code](/examples/ltx2/model_inference/LTX-2.3-T2AV-IC-LoRA-Motion-Track-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2.3-T2AV-IC-LoRA-Motion-Track-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2.3-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2.3-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py) | [code](/examples/ltx2/model_training/full/LTX-2-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_full/LTX-2-T2AV.py) | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV.py) |
| LTX-2 | [Lightricks/LTX-2-19b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Union-Control) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Union-Control.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Union-Control.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Detailer) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Detailer.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Detailer.py) | - | - | [code](/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh) | [code](/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py) |
| LTX-2 | [Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2) | [code](/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-In.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-In.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Out.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Out.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Left.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Left.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Right.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Right.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Up.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Up.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Down.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Down.py) | - | - | - | - |
| LTX-2 | [Lightricks/LTX-2-19b-LoRA-Camera-Control-Static](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) | [code](/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Static.py) | [code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Static.py) | - | - | - | - |
| FLUX.2 | [black-forest-labs/FLUX.2-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev) | [code](/examples/flux2/model_inference/FLUX.2-dev.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-dev.py) | - | - | [code](/examples/flux2/model_training/lora/FLUX.2-dev.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-dev.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) | [code](/examples/flux2/model_inference/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-4B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-4B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-4B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-4B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-9B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) | [code](/examples/flux2/model_inference/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-9B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-9B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-9B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-9B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B) | [code](/examples/flux2/model_inference/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-base-4B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-base-4B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-base-4B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-base-4B.py) |
| FLUX.2 | [black-forest-labs/FLUX.2-klein-base-9B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-9B) | [code](/examples/flux2/model_inference/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_inference_low_vram/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_training/full/FLUX.2-klein-base-9B.sh) | [code](/examples/flux2/model_training/validate_full/FLUX.2-klein-base-9B.py) | [code](/examples/flux2/model_training/lora/FLUX.2-klein-base-9B.sh) | [code](/examples/flux2/model_training/validate_lora/FLUX.2-klein-base-9B.py) |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Aesthetic](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Aesthetic) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Aesthetic.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Aesthetic.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Aesthetic.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Aesthetic.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Brightness](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Brightness) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Brightness.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Brightness.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Brightness.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Brightness.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Age](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Age) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Age.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Age.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Age.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Age.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-ControlNet](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ControlNet) | [code](/examples/flux2/model_inference/Template-KleinBase4B-ControlNet.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-ControlNet.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-ControlNet.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-ControlNet.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Edit](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Edit) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Edit.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Edit.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Edit.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Edit.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Inpaint) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Inpaint.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Inpaint.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Inpaint.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Inpaint.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-PandaMeme](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-PandaMeme) | [code](/examples/flux2/model_inference/Template-KleinBase4B-PandaMeme.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-PandaMeme.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-PandaMeme.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-PandaMeme.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Sharpness](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Sharpness) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Sharpness.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Sharpness.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Sharpness.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Sharpness.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-SoftRGB](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-SoftRGB) | [code](/examples/flux2/model_inference/Template-KleinBase4B-SoftRGB.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-SoftRGB.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-SoftRGB.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-SoftRGB.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-Upscaler](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Upscaler) | [code](/examples/flux2/model_inference/Template-KleinBase4B-Upscaler.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-Upscaler.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-Upscaler.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-Upscaler.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/Template-KleinBase4B-ContentRef](https://www.modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ContentRef) | [code](/examples/flux2/model_inference/Template-KleinBase4B-ContentRef.py) | [code](/examples/flux2/model_inference_low_vram/Template-KleinBase4B-ContentRef.py) | [code](/examples/flux2/model_training/full/Template-KleinBase4B-ContentRef.sh) | [code](/examples/flux2/model_training/validate_full/Template-KleinBase4B-ContentRef.py) | - | - |
| FLUX.2 | [DiffSynth-Studio/KleinBase4B-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2) | [code](/examples/flux2/model_inference/KleinBase4B-i2L-v2.py) | [code](/examples/flux2/model_inference_low_vram/KleinBase4B-i2L-v2.py) | [code](/examples/flux2/model_training/full/KleinBase4B-i2L-v2.sh) | [code](/examples/flux2/model_training/validate_full/KleinBase4B-i2L-v2.py) | - | - |
| Z-Image | [Tongyi-MAI/Z-Image](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image) | [code](/examples/z_image/model_inference/Z-Image.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image.py) | [code](/examples/z_image/model_training/full/Z-Image.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image.py) | [code](/examples/z_image/model_training/lora/Z-Image.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image.py) |
| Z-Image | [DiffSynth-Studio/Z-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) | [code](/examples/z_image/model_inference/Z-Image-i2L.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-i2L.py) | - | - | - | - |
| Z-Image | [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [code](/examples/z_image/model_inference/Z-Image-Turbo.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Union-2.1.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.py) |
| Z-Image | [PAI/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps](https://www.modelscope.cn/models/PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | [code](/examples/z_image/model_inference/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.py) |
| Z-Image | [DiffSynth-Studio/ZImage-i2L-v2](https://www.modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2) | [code](/examples/z_image/model_inference/ZImage-i2L-v2.py) | [code](/examples/z_image/model_inference_low_vram/ZImage-i2L-v2.py) | [code](/examples/z_image/model_training/full/ZImage-i2L-v2.sh) | [code](/examples/z_image/model_training/validate_full/ZImage-i2L-v2.py) | - | - |
| Anima | [circlestone-labs/Anima](https://www.modelscope.cn/models/circlestone-labs/Anima) | [code](/examples/anima/model_inference/anima-preview.py) | [code](/examples/anima/model_inference_low_vram/anima-preview.py) | [code](/examples/anima/model_training/full/anima-preview.sh) | [code](/examples/anima/model_training/validate_full/anima-preview.py) | [code](/examples/anima/model_training/lora/anima-preview.sh) | [code](/examples/anima/model_training/validate_lora/anima-preview.py) |
| Qwen-Image | [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) | [code](/examples/qwen_image/model_inference/Qwen-Image.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image.py) |
| Qwen-Image | [Qwen/Qwen-Image-2512](https://www.modelscope.cn/models/Qwen/Qwen-Image-2512) | [code](/examples/qwen_image/model_inference/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-2512.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-2512.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-2512.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-2512.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py) |
| Qwen-Image | [Qwen/Qwen-Image-Edit-2511](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Edit-2511.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2511.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2511.py) |
| Qwen-Image | [FireRedTeam/FireRed-Image-Edit-1.0](https://www.modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.0) | [code](/examples/qwen_image/model_inference/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_inference_low_vram/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_training/full/FireRed-Image-Edit-1.0.sh) | [code](/examples/qwen_image/model_training/validate_full/FireRed-Image-Edit-1.0.py) | [code](/examples/qwen_image/model_training/lora/FireRed-Image-Edit-1.0.sh) | [code](/examples/qwen_image/model_training/validate_lora/FireRed-Image-Edit-1.0.py) |
| Qwen-Image | [FireRedTeam/FireRed-Image-Edit-1.1](https://www.modelscope.cn/models/FireRedTeam/FireRed-Image-Edit-1.1) | [code](/examples/qwen_image/model_inference/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_inference_low_vram/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_training/full/FireRed-Image-Edit-1.1.sh) | [code](/examples/qwen_image/model_training/validate_full/FireRed-Image-Edit-1.1.py) | [code](/examples/qwen_image/model_training/lora/FireRed-Image-Edit-1.1.sh) | [code](/examples/qwen_image/model_training/validate_lora/FireRed-Image-Edit-1.1.py) |
| Qwen-Image | [lightx2v/Qwen-Image-Edit-2511-Lightning](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-Lightning.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-2511-Lightning.py) | - | - | - | - |
| Qwen-Image | [Qwen/Qwen-Image-Layered](https://www.modelscope.cn/models/Qwen/Qwen-Image-Layered) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Layered.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Layered-Control](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Layered-Control.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Layered-Control.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered-Control.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered-Control.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Layered-Control-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control-V2) | [code](/examples/qwen_image/model_inference/Qwen-Image-Layered-Control-V2.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Layered-Control-V2.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Layered-Control-V2.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Layered-Control-V2.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-V2.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) | [code](/examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-Poster.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-EliGen-Poster.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-EliGen-Poster.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Distill-Full.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-Full.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-Full.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA) | [code](/examples/qwen_image/model_inference/Qwen-Image-Distill-LoRA.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Distill-LoRA.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Distill-LoRA.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Canny.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Canny.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Canny.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Depth.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Depth.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Depth.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint) | [code](/examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/full/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Inpaint.py) | [code](/examples/qwen_image/model_training/lora/Qwen-Image-Blockwise-ControlNet-Inpaint.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Inpaint.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) | [code](/examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-In-Context-Control-Union.py) | - | - | [code](/examples/qwen_image/model_training/lora/Qwen-Image-In-Context-Control-Union.sh) | [code](/examples/qwen_image/model_training/validate_lora/Qwen-Image-In-Context-Control-Union.py) |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) | [code](/examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-Edit-Lowres-Fix.py) | - | - | - | - |
| Qwen-Image | [DiffSynth-Studio/Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L) | [code](/examples/qwen_image/model_inference/Qwen-Image-i2L.py) | [code](/examples/qwen_image/model_inference_low_vram/Qwen-Image-i2L.py) | - | - | - | - |
| Wan | [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) | [code](/examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-T2V-1.3B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B.py) |
| Wan | [Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | [code](/examples/wanvideo/model_inference/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-T2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-T2V-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-T2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-14B.py) |
| Wan | [Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) | [code](/examples/wanvideo/model_inference/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-I2V-14B-480P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-480P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-480P.py) |
| Wan | [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) | [code](/examples/wanvideo/model_inference/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-I2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-I2V-14B-720P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-I2V-14B-720P.py) |
| Wan | [Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P) | [code](/examples/wanvideo/model_inference/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-FLF2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-FLF2V-14B-720P.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-FLF2V-14B-720P.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py) |
| Wan | [iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B-Preview.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B-Preview.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B-Preview.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B-Preview.py) |
| Wan | [Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-1.3B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-1.3B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py) |
| Wan | [Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) | [code](/examples/wanvideo/model_inference/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-VACE-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-VACE-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-VACE-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-14B.py) |
| Wan | [PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-1.3B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-14B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-InP.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py) |
| Wan | [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py) |
| Wan | [DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1) | [code](/examples/wanvideo/model_inference/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py) | [code](/examples/wanvideo/model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py) |
| Wan | [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) | [code](/examples/wanvideo/model_inference/krea-realtime-video.py) | [code](/examples/wanvideo/model_inference_low_vram/krea-realtime-video.py) | [code](/examples/wanvideo/model_training/full/krea-realtime-video.sh) | [code](/examples/wanvideo/model_training/validate_full/krea-realtime-video.py) | [code](/examples/wanvideo/model_training/lora/krea-realtime-video.sh) | [code](/examples/wanvideo/model_training/validate_lora/krea-realtime-video.py) |
| Wan | [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) | [code](/examples/wanvideo/model_inference/LongCat-Video.py) | [code](/examples/wanvideo/model_inference_low_vram/LongCat-Video.py) | [code](/examples/wanvideo/model_training/full/LongCat-Video.sh) | [code](/examples/wanvideo/model_training/validate_full/LongCat-Video.py) | [code](/examples/wanvideo/model_training/lora/LongCat-Video.sh) | [code](/examples/wanvideo/model_training/validate_lora/LongCat-Video.py) |
| Wan | [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) | [code](/examples/wanvideo/model_inference/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_training/full/Video-As-Prompt-Wan2.1-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Video-As-Prompt-Wan2.1-14B.py) | [code](/examples/wanvideo/model_training/lora/Video-As-Prompt-Wan2.1-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Video-As-Prompt-Wan2.1-14B.py) |
| Wan | [Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-T2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-T2V-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-T2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-T2V-A14B.py) |
| Wan | [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-I2V-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-I2V-A14B.py) |
| Wan | [Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) | [code](/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-TI2V-5B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-14B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-2-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-2-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-2-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-2-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-2-14B.py) |
| Wan | [Wan-AI/Wan2.2-Animate-2-14B: Distilled](https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-2-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Animate-2-14B-Distilled.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Animate-2-14B-Distilled.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Animate-2-14B-Distilled.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Animate-2-14B-Distilled.py) |
| Wan | [Wan-AI/Wan2.2-S2V-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-S2V-14B) | [code](/examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-S2V-14B_multi_clips.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-S2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-S2V-14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-S2V-14B.py) |
| Wan | [PAI/Wan2.2-VACE-Fun-A14B](https://www.modelscope.cn/models/PAI/Wan2.2-VACE-Fun-A14B) | [code](/examples/wanvideo/model_inference/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-VACE-Fun-A14B.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-VACE-Fun-A14B.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-VACE-Fun-A14B.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-VACE-Fun-A14B.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-InP](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-InP.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-InP.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-InP.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-Control](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control.py) |
| Wan | [PAI/Wan2.2-Fun-A14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera) | [code](/examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/full/Wan2.2-Fun-A14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan2.2-Fun-A14B-Control-Camera.py) | [code](/examples/wanvideo/model_training/lora/Wan2.2-Fun-A14B-Control-Camera.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan2.2-Fun-A14B-Control-Camera.py) |
| Wan | [openmoss/MOVA-360p](https://modelscope.cn/models/openmoss/MOVA-360p) | [code](/examples/mova/model_inference/MOVA-360p-I2AV.py) | [code](/examples/mova/model_inference_low_vram/MOVA-360p-I2AV.py) | [code](/examples/mova/model_training/full/MOVA-360P-I2AV.sh) | [code](/examples/mova/model_training/validate_full/MOVA-360p-I2AV.py) | [code](/examples/mova/model_training/lora/MOVA-360P-I2AV.sh) | [code](/examples/mova/model_training/validate_lora/MOVA-360p-I2AV.py) |
| Wan | [openmoss/MOVA-720p](https://modelscope.cn/models/openmoss/MOVA-720p) | [code](/examples/mova/model_inference/MOVA-720p-I2AV.py) | [code](/examples/mova/model_inference_low_vram/MOVA-720p-I2AV.py) | [code](/examples/mova/model_training/full/MOVA-720P-I2AV.sh) | [code](/examples/mova/model_training/validate_full/MOVA-720p-I2AV.py) | [code](/examples/mova/model_training/lora/MOVA-720P-I2AV.sh) | [code](/examples/mova/model_training/validate_lora/MOVA-720p-I2AV.py) |
| Wan | [Wan-AI/Wan-Dancer-14B (global model)](https://modelscope.cn/models/Wan-AI/Wan-Dancer-14B) | [code](/examples/wanvideo/model_inference/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_training/full/Wan-Dancer-14B-global.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan-Dancer-14B-global.py) | [code](/examples/wanvideo/model_training/lora/Wan-Dancer-14B-global.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan-Dancer-14B-global.py) |
| Wan | [Wan-AI/Wan-Dancer-14B (local model)](https://modelscope.cn/models/Wan-AI/Wan-Dancer-14B) | [code](/examples/wanvideo/model_inference/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_inference_low_vram/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_training/full/Wan-Dancer-14B-local.sh) | [code](/examples/wanvideo/model_training/validate_full/Wan-Dancer-14B-local.py) | [code](/examples/wanvideo/model_training/lora/Wan-Dancer-14B-local.sh) | [code](/examples/wanvideo/model_training/validate_lora/Wan-Dancer-14B-local.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev) | [code](/examples/flux/model_inference/FLUX.1-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) | [code](/examples/flux/model_inference/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Krea-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Krea-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Krea-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Krea-dev.py) |
| FLUX.1 | [black-forest-labs/FLUX.1-Kontext-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev) | [code](/examples/flux/model_inference/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_training/full/FLUX.1-Kontext-dev.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-Kontext-dev.py) | [code](/examples/flux/model_training/lora/FLUX.1-Kontext-dev.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-Kontext-dev.py) |
| FLUX.1 | [alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://www.modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Inpainting-Beta.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Inpainting-Beta.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Inpainting-Beta.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Inpainting-Beta.py) |
| FLUX.1 | [InstantX/FLUX.1-dev-Controlnet-Union-alpha](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Union-alpha.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Union-alpha.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Union-alpha.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Union-alpha.py) |
| FLUX.1 | [jasperai/Flux.1-dev-Controlnet-Upscaler](https://www.modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler) | [code](/examples/flux/model_inference/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-Controlnet-Upscaler.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-Controlnet-Upscaler.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-Controlnet-Upscaler.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-Controlnet-Upscaler.py) |
| FLUX.1 | [InstantX/FLUX.1-dev-IP-Adapter](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-IP-Adapter) | [code](/examples/flux/model_inference/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-IP-Adapter.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-IP-Adapter.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-IP-Adapter.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-IP-Adapter.py) |
| FLUX.1 | [ByteDance/InfiniteYou](https://www.modelscope.cn/models/ByteDance/InfiniteYou) | [code](/examples/flux/model_inference/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-InfiniteYou.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-InfiniteYou.py) | [code](/examples/flux/model_training/lora/FLUX.1-dev-InfiniteYou.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-InfiniteYou.py) |
| FLUX.1 | [DiffSynth-Studio/Eligen](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen) | [code](/examples/flux/model_inference/FLUX.1-dev-EliGen.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-EliGen.py) | - | - | [code](/examples/flux/model_training/lora/FLUX.1-dev-EliGen.sh) | [code](/examples/flux/model_training/validate_lora/FLUX.1-dev-EliGen.py) |
| FLUX.1 | [DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev](https://www.modelscope.cn/models/DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev) | [code](/examples/flux/model_inference/FLUX.1-dev-LoRA-Encoder.py) | [code](/examples/flux/model_inference_low_vram/FLUX.1-dev-LoRA-Encoder.py) | [code](/examples/flux/model_training/full/FLUX.1-dev-LoRA-Encoder.sh) | [code](/examples/flux/model_training/validate_full/FLUX.1-dev-LoRA-Encoder.py) | - | - |
| FLUX.1 | [DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev](https://modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev) | [code](/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py) | - | - | - | - | - |
| FLUX.1 | [stepfun-ai/Step1X-Edit](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit) | [code](/examples/flux/model_inference/Step1X-Edit.py) | [code](/examples/flux/model_inference_low_vram/Step1X-Edit.py) | [code](/examples/flux/model_training/full/Step1X-Edit.sh) | [code](/examples/flux/model_training/validate_full/Step1X-Edit.py) | [code](/examples/flux/model_training/lora/Step1X-Edit.sh) | [code](/examples/flux/model_training/validate_lora/Step1X-Edit.py) |
| FLUX.1 | [ostris/Flex.2-preview](https://www.modelscope.cn/models/ostris/Flex.2-preview) | [code](/examples/flux/model_inference/FLEX.2-preview.py) | [code](/examples/flux/model_inference_low_vram/FLEX.2-preview.py) | [code](/examples/flux/model_training/full/FLEX.2-preview.sh) | [code](/examples/flux/model_training/validate_full/FLEX.2-preview.py) | [code](/examples/flux/model_training/lora/FLEX.2-preview.sh) | [code](/examples/flux/model_training/validate_lora/FLEX.2-preview.py) |
| FLUX.1 | [DiffSynth-Studio/Nexus-GenV2](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2) | [code](/examples/flux/model_inference/Nexus-Gen-Editing.py) | [code](/examples/flux/model_inference_low_vram/Nexus-Gen-Editing.py) | [code](/examples/flux/model_training/full/Nexus-Gen.sh) | [code](/examples/flux/model_training/validate_full/Nexus-Gen.py) | [code](/examples/flux/model_training/lora/Nexus-Gen.sh) | [code](/examples/flux/model_training/validate_lora/Nexus-Gen.py) |
| Stable Diffusion XL | [stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0) | [code](/examples/stable_diffusion_xl/model_inference/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_inference_low_vram/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_training/full/stable-diffusion-xl-base-1.0.sh) | [code](/examples/stable_diffusion_xl/model_training/validate_full/stable-diffusion-xl-base-1.0.py) | [code](/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh) | [code](/examples/stable_diffusion_xl/model_training/validate_lora/stable-diffusion-xl-base-1.0.py) |
| Stable Diffusion | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5) | [code](/examples/stable_diffusion/model_inference/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_inference_low_vram/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_training/full/stable-diffusion-v1-5.sh) | [code](/examples/stable_diffusion/model_training/validate_full/stable-diffusion-v1-5.py) | [code](/examples/stable_diffusion/model_training/lora/stable-diffusion-v1-5.sh) | [code](/examples/stable_diffusion/model_training/validate_lora/stable-diffusion-v1-5.py) |
