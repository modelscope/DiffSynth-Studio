# DiffSynth-Studio

<a href="https://github.com/modelscope/DiffSynth-Studio"><img src=".github/workflows/logo.gif" title="Logo" style="max-width:100%;" width="55" /></a> <a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/)
[![Discord](https://badgen.net//discord/members/Mm9suEeUDc)](https://discord.gg/Mm9suEeUDc)

[Switch to English](./README.md)

## 简介

欢迎来到 Diffusion 模型的魔法世界！DiffSynth-Studio 是由[魔搭社区](https://www.modelscope.cn/)团队开发和维护的开源 Diffusion 模型引擎。我们期望以框架建设孵化技术创新，凝聚开源社区的力量，探索生成式模型技术的丰富能力！

框架功能：

* [模型支持](#全部支持的模型)：框架集成了主流的开源 Diffusion 模型，涵盖图像生成、视频生成、音频生成，以及图像指标模型。
* [显存管理](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/VRAM_management.html)：在硬盘、内存、显存之间动态调度模型参数，让低显存的消费级 GPU 也能运行大模型的推理。
* [参数量化](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/Quantization.html)：将模型参数转换为 NF4、INT8 等量化精度，大幅减少模型推理和 LoRA 模型训练的显存需求。
* [任意训练](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Pipeline_Usage/Model_Training.html)：几乎所有支持推理的模型都支持训练，无论是基础模型、LoRA，还是任何带有额外输入的 Adapter 模型。
* [拆分训练](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Training/Split_Training.html)：使用计算图推理引擎追踪 Pipeline 中每个变量，将训练过程拆分为两阶段，高效训练。

参考资料：

* 开发者文档 (面向人类)：[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/README.html)、[English version](https://diffsynth-studio-doc.readthedocs.io/en/latest/README.html)
* Agent Skills (面向 AI)：[DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills)

查看更多：

* [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI): 基于 DiffSynth-Studio 构建的轻量化 LoRA 训练工具，可在消费级 GPU 上训练 LoRA 模型。
* [魔搭社区 AIGC 专区 (面向中国用户)](https://modelscope.cn/aigc/home): 以 DiffSynth-Studio 为核心推理和训练引擎的产品化功能，体验开源模型生态的丰富潜力。
* [ModelScope Civision (for global users)](https://modelscope.ai/civision/home): Unlock the vast potential of the open-source model ecosystem through productized capabilities powered by DiffSynth-Studio.

## 更新历史

> DiffSynth-Studio 经历了大版本更新，部分旧功能已停止维护，如需使用旧版功能，请切换到大版本更新前的[最后一个历史版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3)。

> 目前本项目的开发人员有限，大部分工作由 [Artiprocher](https://github.com/Artiprocher) 和 [mi804](https://github.com/mi804) 负责，因此新功能的开发进展会比较缓慢，issue 的回复和解决速度有限，我们对此感到非常抱歉，请各位开发者理解。

- **2026年8月25日** 我们开源了 [DiffSynth-WebUI](https://github.com/modelscope/DiffSynth-WebUI)，一键私有化部署 LoRA 训练服务，与模型参数量化功能结合，用消费级 GPU 也能训练超大模型。

- **2026年8月19日** 我们发布了模型量化功能。它提供了统一的 `QuantizeConfig` 入口，支持 bitsandbytes、torchao、comfy-kitchen 等多种量化后端，具备在线量化、加载预量化权重、混合量化、保存量化模型、量化 + LoRA 训练等能力。详情请参考[文档](/docs/zh/Pipeline_Usage/Quantization.md)。

- **2026年8月17日** MiniMax-Music3 开源，欢迎加入音频生成模型家族！支持文生音乐推理和低显存推理能力。详情请参考[文档](/docs/zh/Model_Details/MiniMax-Music3.md)和[示例代码](/examples/minimax_music3/)。

- **2026年8月7日** 我们为 Wan 系列新增了 Wan-Animate-2，输入一张参考图和一段驱动视频，即可让参考角色演绎驱动视频中的动作，生成高质量角色动画，包含标准与蒸馏两个变体。详情请参考[文档](/docs/zh/Model_Details/Wan.md)和[示例代码](/examples/wanvideo/)。

- **2026年8月3日** MiniMax-H3 开源，欢迎加入视频生成模型家族！支持文生音视频、首尾帧引导生成、参考驱动生成、低显存推理以及 NF4 量化推理。详情请参考[文档](/docs/zh/Model_Details/MiniMax-H3.md)和[示例代码](/examples/minimax_h3/)。

- **2026年7月28日** LingBot-Video 开源，欢迎加入视频生成模型家族！本次开源包含 Dense-1.3B 和 MoE-30B-A3B 两个版本（MoE 版总参数量 30B、每个 token 激活约 3B），均支持文生视频、图生视频、文生图推理、低显存推理以及 LoRA / 全量训练能力。详情请参考[文档](/docs/zh/Model_Details/LingBot-Video.md)和[示例代码](/examples/lingbot_video/)。特别感谢 [NancyFyong](https://github.com/NancyFyong) 贡献了本模型的接入代码！

- **2026年7月21日** 我们开源了 [DiffSynth-Studio Model Integration Skills](https://www.modelscope.cn/collections/DiffSynth-Studio/DiffSynth-Studio-Model-Integration-Skills)。这是一套可组合的 Agent Skill 合集，将外部扩散模型接入 DiffSynth-Studio 的全流程自动化，大幅提升模型接入标准化程度与效率。从[使用示例](https://www.modelscope.cn/skills/DiffSynth-Studio/diffsynth-integrator/file/view/master/example.md?status=1)开始体验吧！

<details>
<summary>更多</summary>

- **2026年6月29日** Boogu-Image 开源，已支持文生图推理、图像编辑、低显存推理和训练能力。详情请参考[文档](/docs/zh/Model_Details/Boogu-Image.md)和[示例代码](/examples/boogu_image/)。

- **2026年6月24日** Krea-2 开源，我们已提供全面支持。详情请参考[文档](/docs/zh/Model_Details/Krea-2.md)和[示例代码](/examples/krea2/)。

- **2026年6月16日** 我们为 ACE-Step 新增了 Template 模型：[vocals2music](https://www.modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-vocals2music)。详情请参考[文档](/docs/zh/Model_Details/ACE-Step.md)和[示例代码](/examples/ace_step/)。

- **2026年6月15日** 我们开源了 Image-to-LoRA V2，将动辄数小时的图像风格 LoRA 训练压缩到一次模型推理中，探索 LoRA 模型训练的新方式。[技术报告](https://arxiv.org/abs/2606.13809)已公开，本次开源包括三个模型：
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2)：适配模型 Z-Image
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2)：适配模型 FLUX.2-klein-base-4B
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2)：适配模型 Hidream-O1-Image

- **2026年6月5日** Ideogram 4 开源，已支持文生图推理。详情请参考[文档](/docs/zh/Model_Details/Ideogram-4.md)和[示例代码](/examples/ideogram4/)。

- **2026年5月21日** 新增图像质量评估模型的支持，包括 FID、CLIP、Aesthetic、PickScore、ImageReward、HPSv2、HPSv3，详情请参考[文档](/docs/zh/Model_Details/Image-Quality-Metrics.md)和[示例代码](/examples/image_quality_metric/)

- **2026年5月18日** 新增 **CPU Offload Training** 功能，通过将模型权重逐层在 CPU 与 GPU 之间搬运，大幅降低训练时的 GPU 显存占用，让消费级显卡也能进行大模型 LoRA 训练，适配所有模型。只需在训练命令中添加 `--enable_model_cpu_offload` 即可启用（当前仅支持单卡训练）。详情请参考[文档](/docs/zh/Training/Offload_Training.md)。

- **2026年5月14日** HiDream-O1-Image 开源，欢迎加入图像生成模型家族！支持文生图推理、图像编辑推理、低显存推理和训练能力。详情请参考[文档](/docs/zh/Model_Details/HiDream-O1-Image.md)和[示例代码](/examples/hidream_o1_image/)。

- **2026年4月28日** 我们发布了 Diffusion Templates，面向 Diffusion 模型的插件框架，大幅降低了可控生成模型的训练门槛，一起来探索新奇的技术吧！
    * 开源代码：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
    * 技术报告：[arXiv](https://arxiv.org/abs/2604.24351)
    * 项目主页：[GitHub](https://modelscope.github.io/diffusion-templates-web/)
    * 文档参考：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
    * 在线体验：[魔搭社区创空间](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
    * 模型集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope 国际站](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
    * 数据集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope 国际站](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)
- **2026年4月27日** 我们支持了 ACE-Step-1.5！包括文生音乐推理、低显存推理和 LoRA 训练能力。详情请参考[文档](/docs/zh/Model_Details/ACE-Step.md)和[示例代码](/examples/ace_step/)。

- **2026年4月27日** 我们重新支持了模型 Stable Diffusion v1.5 和 SDXL，仅对这两类模型提供学术科研支持。

- **2026年4月14日** JoyAI-Image 开源，欢迎加入图像编辑模型家族！支持指令引导的图像编辑推理、低显存推理和训练能力。详情请参考[文档](/docs/zh/Model_Details/JoyAI-Image.md)和[示例代码](/examples/joyai_image/)。

- **2026年3月19日** 新增对 [openmoss/MOVA-720p](https://modelscope.cn/models/openmoss/MOVA-720p) 和 [openmoss/MOVA-360p](https://modelscope.cn/models/openmoss/MOVA-360p) 模型的支持，包括完整的训练和推理功能。[文档](/docs/zh/Model_Details/Wan.md)和[示例代码](/examples/mova/)现已可用。

- **2026年3月12日** 我们新增了 [LTX-2.3](https://modelscope.cn/models/Lightricks/LTX-2.3) 音视频生成模型的支持，模型支持的功能包括文生音视频、图生音视频、IC-LoRA控制、音频生视频、音视频局部Inpainting，框架支持完整的推理和训练功能。详细信息请参考 [文档](/docs/zh/Model_Details/LTX-2.md) 和 [示例代码](/examples/ltx2/)。

- **2026年3月3日** 我们发布了 [DiffSynth-Studio/Qwen-Image-Layered-Control-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control-V2) 模型，这是 Qwen-Image-Layered-Control 的更新版本。除了原本就支持的文本引导功能，新增了画笔控制的图层拆分能力。

- **2026年3月2日** 新增对[Anima](https://modelscope.cn/models/circlestone-labs/Anima)的支持，详见[文档](docs/zh/Model_Details/Anima.md)。这是一个有趣的动漫风格图像生成模型，我们期待其后续的模型更新。

- **2026年2月26日** 新增对[LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2)音视频生成模型全量微调与LoRA训练支持，详见[文档](docs/zh/Model_Details/LTX-2.md)。

- **2026年2月10日** 新增对[LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2)音视频生成模型的推理支持，详见[文档](docs/zh/Model_Details/LTX-2.md)，后续将推进模型训练的支持。

- **2026年2月2日** Research Tutorial 的第一篇文档上线，带你从零开始训练一个 0.1B 的小型文生图模型，详见[文档](/docs/zh/Research_Tutorial/train_from_scratch.md)、[模型](https://modelscope.cn/models/DiffSynth-Studio/AAAMyModel)，我们希望 DiffSynth-Studio 能够成为一个更强大的 Diffusion 模型训练框架。

- **2026年1月27日** [Z-Image](https://modelscope.cn/models/Tongyi-MAI/Z-Image) 发布，我们的 [Z-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) 模型同步发布，在[魔搭创空间](https://modelscope.cn/studios/DiffSynth-Studio/Z-Image-i2L)可直接体验，详见[文档](/docs/zh/Model_Details/Z-Image.md)。

- **2026年1月19日** 新增对 [FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) 和 [FLUX.2-klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) 模型的支持，包括完整的训练和推理功能。[文档](/docs/zh/Model_Details/FLUX2.md)和[示例代码](/examples/flux2/)现已可用。

- **2026年1月12日** 我们训练并开源了一个文本引导的图层拆分模型（[模型链接](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control)），这一模型输入一张图与一段文本描述，模型会将图像中与文本描述相关的图层拆分出来。更多细节请阅读我们的 blog（[中文版](https://modelscope.cn/learn/4938)、[英文版](https://huggingface.co/blog/kelseye/qwen-image-layered-control)）。

- **2025年12月24日** 我们基于 Qwen-Image-Edit-2511 训练了一个 In-Context Editing LoRA 模型（[模型链接](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)），这个模型可以输入三张图：图A、图B、图C，模型会自行分析图A到图B的变化，并将这样的变化应用到图C，生成图D。更多细节请阅读我们的 blog（[中文版](https://mp.weixin.qq.com/s/41aEiN3lXKGCJs1-we4Q2g)、[英文版](https://huggingface.co/blog/kelseye/qwen-image-edit-2511-icedit-lora)）。

- **2025年12月9日** 我们基于 DiffSynth-Studio 2.0 训练了一个疯狂的模型：[Qwen-Image-i2L](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-i2L)（Image to LoRA）。这一模型以图像为输入，以 LoRA 为输出。尽管这个版本的模型在泛化能力、细节保持能力等方面还有很大改进空间，我们将这些模型开源，以启发更多创新性的研究工作。更多细节，请参考我们的 [blog](https://huggingface.co/blog/kelseye/qwen-image-i2l)。

- **2025年12月4日** DiffSynth-Studio 2.0 发布！众多新功能上线
  - [文档](/docs/zh/README.md)上线：我们的文档还在持续优化更新中
  - [显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)模块升级，支持 Layer 级别的 Disk Offload，同时释放内存与显存
  - 新模型支持
    - Z-Image Turbo: [模型](https://www.modelscope.ai/models/Tongyi-MAI/Z-Image-Turbo)、[文档](/docs/zh/Model_Details/Z-Image.md)、[代码](/examples/z_image/)
    - FLUX.2-dev: [模型](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-dev)、[文档](/docs/zh/Model_Details/FLUX2.md)、[代码](/examples/flux2/)
  - 训练框架升级
    - [拆分训练](/docs/zh/Training/Split_Training.md)：支持自动化地将训练过程拆分为数据处理和训练两阶段（即使训练的是 ControlNet 或其他任意模型），在数据处理阶段进行文本编码、VAE 编码等不需要梯度回传的计算，在训练阶段处理其他计算。速度更快，显存需求更少。
    - [差分 LoRA 训练](/docs/zh/Training/Differential_LoRA.md)：这是我们曾在 [ArtAug](https://www.modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1) 中使用的训练技术，目前已可用于任意模型的 LoRA 训练。
    - [FP8 训练](/docs/zh/Training/FP8_Precision.md)：FP8 在训练中支持应用到任意非训练模型，即梯度关闭或者梯度仅影响 LoRA 权重的模型。
- **2025年11月4日** 支持了 [ByteDance/Video-As-Prompt-Wan2.1-14B](https://modelscope.cn/models/ByteDance/Video-As-Prompt-Wan2.1-14B) 模型，该模型基于 Wan 2.1 训练，支持根据参考视频生成相应的动作。

- **2025年10月30日** 支持了 [meituan-longcat/LongCat-Video](https://www.modelscope.cn/models/meituan-longcat/LongCat-Video) 模型，该模型支持文生视频、图生视频、视频续写。这个模型在本项目中沿用 Wan 的框架进行推理和训练。

- **2025年10月27日** 支持了 [krea/krea-realtime-video](https://www.modelscope.cn/models/krea/krea-realtime-video) 模型，Wan 模型生态再添一员。

- **2025年9月23日** [DiffSynth-Studio/Qwen-Image-EliGen-Poster](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-Poster) 发布！本模型由我们与淘天体验设计团队联合研发并开源。模型基于 Qwen-Image 构建，专为电商海报场景设计，支持精确的分区布局控制。 请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-EliGen-Poster.py)。

- **2025年9月9日** 我们的训练框架支持了多种训练模式，目前已适配 Qwen-Image，除标准 SFT 训练模式外，已支持 Direct Distill，请参考[我们的示例代码](./examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh)。这项功能是实验性的，我们将会继续完善已支持更全面的模型训练功能。

- **2025年8月28日** 我们支持了Wan2.2-S2V，一个音频驱动的电影级视频生成模型。请参见[./examples/wanvideo/](./examples/wanvideo/)。

- **2025年8月21日** [DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2) 发布！相比于 V1 版本，训练数据集变为 [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset)，因此，生成的图像更符合 Qwen-Image 本身的图像分布和风格。 请参考[我们的示例代码](./examples/qwen_image/model_inference_low_vram/Qwen-Image-EliGen-V2.py)。

- **2025年8月21日** 我们开源了 [DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union) 结构控制 LoRA 模型，采用 In Context 的技术路线，支持多种类别的结构控制条件，包括 canny, depth, lineart, softedge, normal, openpose。 请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-In-Context-Control-Union.py)。

- **2025年8月20日** 我们开源了 [DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix) 模型，提升了 Qwen-Image-Edit 对低分辨率图像输入的编辑效果。请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-Edit-Lowres-Fix.py)

- **2025年8月19日** Qwen-Image-Edit 开源，欢迎图像编辑模型新成员！

- **2025年8月18日** 我们训练并开源了 Qwen-Image 的图像重绘 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint)，模型结构采用了轻量化的设计，请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py)。

- **2025年8月15日** 我们开源了 [Qwen-Image-Self-Generated-Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Qwen-Image-Self-Generated-Dataset) 数据集。这是一个使用 Qwen-Image 模型生成的图像数据集，共包含 160,000 张`1024 x 1024`图像。它包括通用、英文文本渲染和中文文本渲染子集。我们为每张图像提供了图像描述、实体和结构控制图像的标注。开发者可以使用这个数据集来训练 Qwen-Image 模型的 ControlNet 和 EliGen 等模型，我们旨在通过开源推动技术发展！

- **2025年8月13日** 我们训练并开源了 Qwen-Image 的 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth)，模型结构采用了轻量化的设计，请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py)。

- **2025年8月12日** 我们训练并开源了 Qwen-Image 的 ControlNet 模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny)，模型结构采用了轻量化的设计，请参考[我们的示例代码](./examples/qwen_image/model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py)。

- **2025年8月11日** 我们开源了 Qwen-Image 的蒸馏加速模型 [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)，沿用了与 [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full) 相同的训练流程，但模型结构修改为了 LoRA，因此能够更好地与其他开源生态模型兼容。

- **2025年8月7日** 我们开源了 Qwen-Image 的实体控制 LoRA 模型 [DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen)。Qwen-Image-EliGen 能够实现实体级可控的文生图。技术细节请参见[论文](https://arxiv.org/abs/2501.01097)。训练数据集：[EliGenTrainSet](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)。

- **2025年8月5日** 我们开源了 Qwen-Image 的蒸馏加速模型 [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)，实现了约 5 倍加速。

- **2025年8月4日** Qwen-Image 开源，欢迎图像生成模型家族新成员！

- **2025年8月1日** [FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev) 开源，这是一个专注于美学摄影的文生图模型。我们第一时间提供了全方位支持，包括低显存逐层 offload、LoRA 训练、全量训练。详细信息请参考 [./examples/flux/](./examples/flux/)。

- **2025年7月28日** Wan 2.2 开源，我们第一时间提供了全方位支持，包括低显存逐层 offload、FP8 量化、序列并行、LoRA 训练、全量训练。详细信息请参考 [./examples/wanvideo/](./examples/wanvideo/)。

- **2025年7月11日** 我们提出 Nexus-Gen，一个将大语言模型（LLM）的语言推理能力与扩散模型的图像生成能力相结合的统一框架。该框架支持无缝的图像理解、生成和编辑任务。
  - 论文: [Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
  - Github 仓库: https://github.com/modelscope/Nexus-Gen
  - 模型: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
  - 训练数据集: [ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
  - 在线体验: [ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)
- **2025年6月15日** ModelScope 官方评测框架 [EvalScope](https://github.com/modelscope/evalscope) 现已支持文生图生成评测。请参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/t2i_eval.html)指南进行尝试。

- **2025年3月25日** 我们的新开源项目 [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) 现已开源！专注于稳定的模型部署，面向工业界，提供更好的工程支持、更高的计算性能和更稳定的功能。

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

## 安装

从源码安装（推荐）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多安装方式，以及非 NVIDIA GPU 的安装，请参考[安装文档](/docs/zh/Pipeline_Usage/Setup.md)。

<details>
<summary>下载源配置</summary>

> 在进行模型推理和训练前，可通过[环境变量](/docs/zh/Pipeline_Usage/Environment_Variables.md)配置模型下载源等。
> 
> 本项目默认从[魔搭社区](https://modelscope.cn/)下载模型。对于非中国区域的用户，可以通过以下配置从[魔搭社区国际站](https://modelscope.ai)下载模型：
> 
> ```shell
> export MODELSCOPE_ENDPOINT=https://modelscope.ai
> ```
> 
> 如需从 [HuggingFace](https://huggingface.co/) 下载模型，请修改[环境变量](/docs/zh/Pipeline_Usage/Environment_Variables.md)，注意不同模型平台上的模型 ID 可能不同：
> 
> ```shell
> export DIFFSYNTH_DOWNLOAD_SOURCE="huggingface"
> ```

</details>

## 基础框架

DiffSynth-Studio 作为基础的模型框架，为主流 Diffusion 模型重新设计了推理和训练流水线，能够实现高效的显存管理、灵活的模型训练。

快速开始，体验热门和最新模型：

| 基础架构 | 模型 ID | 推理 | 低显存推理 | 全量训练 | 全量训练后验证 | LoRA 训练 | LoRA 训练后验证 |
|-|-|-|-|-|-|-|-|
| MiniMax-H3 | [MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py) | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py) |
| MiniMax-H3 | [DiffSynth-Studio/MiniMax-H3-NF4: FL2VA pruned](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) | [code](/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Pruned-FL2VA.py) | [code](/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Pruned-FL2VA.py) | - | - | [code](/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Pruned-FL2VA.sh) | [code](/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Pruned-FL2VA.py) |
| ACE-Step | [ACE-Step/acestep-v15-xl-sft](https://www.modelscope.cn/models/ACE-Step/acestep-v15-xl-sft) | [code](/examples/ace_step/model_inference/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_inference_low_vram/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/full/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_full/acestep-v15-xl-sft.py) | [code](/examples/ace_step/model_training/lora/acestep-v15-xl-sft.sh) | [code](/examples/ace_step/model_training/validate_lora/acestep-v15-xl-sft.py) |
| Z-Image | [Tongyi-MAI/Z-Image-Turbo](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | [code](/examples/z_image/model_inference/Z-Image-Turbo.py) | [code](/examples/z_image/model_inference_low_vram/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/full/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_full/Z-Image-Turbo.py) | [code](/examples/z_image/model_training/lora/Z-Image-Turbo.sh) | [code](/examples/z_image/model_training/validate_lora/Z-Image-Turbo.py) |
| Krea-2 | [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) | [code](/examples/krea2/model_inference/Krea-2-Raw.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Raw.py) | [code](/examples/krea2/model_training/full/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Raw.py) | [code](/examples/krea2/model_training/lora/Krea-2-Raw.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Raw.py) |
| Krea-2 | [krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo) | [code](/examples/krea2/model_inference/Krea-2-Turbo.py) | [code](/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/full/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_full/Krea-2-Turbo.py) | [code](/examples/krea2/model_training/lora/Krea-2-Turbo.sh) | [code](/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py) |

模型一览：

- 图像生成
    - Boogu-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Boogu-Image.html)、[样例代码](/examples/boogu_image/)
    - Krea-2：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Krea-2.html)、[样例代码](/examples/krea2/)
    - Ideogram 4：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Ideogram-4.html)、[样例代码](/examples/ideogram4/)
    - HiDream-O1-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/HiDream-O1-Image.html)、[样例代码](/examples/hidream_o1_image/)
    - JoyAI-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/JoyAI-Image.html)、[样例代码](/examples/joyai_image/)
    - ERNIE-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/ERNIE-Image.html)、[样例代码](/examples/ernie_image/)
    - FLUX.2：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/FLUX2.html)、[样例代码](/examples/flux2/)
    - Z-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Z-Image.html)、[样例代码](/examples/z_image/)
    - Anima：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Anima.html)、[样例代码](/examples/anima/)
    - Qwen-Image：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Qwen-Image.html)、[样例代码](/examples/qwen_image/)
    - FLUX.1：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/FLUX.html)、[样例代码](/examples/flux/)
    - Stable Diffusion XL：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Stable-Diffusion-XL.html)、[样例代码](/examples/stable_diffusion_xl/)
    - Stable Diffusion：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Stable-Diffusion.html)、[样例代码](/examples/stable_diffusion/)
- 视频生成
    - MiniMax-H3：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/MiniMax-H3.html)、[样例代码](/examples/minimax_h3/)
    - LingBot-Video：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/LingBot-Video.html)、[样例代码](/examples/lingbot_video/)
    - LTX-2：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/LTX-2.html)、[样例代码](/examples/ltx2/)
    - Wan：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Wan.html)、[样例代码](/examples/wanvideo/)
- 音频生成
    - MiniMax-Music3：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/MiniMax-Music3.html)、[样例代码](/examples/minimax_music3/)
    - ACE-Step：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/ACE-Step.html)、[样例代码](/examples/ace_step/)
- 评测模型：[文档](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Image-Quality-Metrics.html)、[样例代码](/examples/image_quality_metric/)

[查看全部支持的模型](#全部支持的模型)

## 创新成果

我们相信，一个完善的开源代码框架能够降低技术探索的门槛，我们基于这个代码库搞出了不少有意思的技术。或许你也有许多天马行空的构想，借助 DiffSynth-Studio，你可以快速实现这些想法。

<details>
<summary>TreeAdapter: 由结构化 LoRA 组建的模型系统</summary>

> 用 1w+ LoRA 组成的模型系统挑战上万个稀有物种的精准生成。

* 论文：[TreeAdapter: Hierarchical Taxonomy-Guided Adapter Composition for Fine-Grained Species Image Generation](https://arxiv.org/abs/2607.24215)
* 模型：[ModelScope](https://modelscope.cn/models/DiffSynth-Studio/TreeAdapter-KleinBase4B)

![Image](https://github.com/user-attachments/assets/1b461e0f-60aa-4b38-a44d-d1646cbbbc75)

</details>

<details>
<summary>Image-to-LoRA: 把模型训练压缩到模型推理中</summary>

> 真正的 Meta Learning：模型一端输入图像数据集，另一端直接输出训出来的 LoRA 模型。

* 论文：[Compressing Image Style Training into a Single Model Forward](https://arxiv.org/abs/2606.13809)
* 模型：
    * [DiffSynth-Studio/ZImage-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/ZImage-i2L-v2)：适配模型 Z-Image
    * [DiffSynth-Studio/KleinBase4B-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/KleinBase4B-i2L-v2)：适配模型 FLUX.2-klein-base-4B
    * [DiffSynth-Studio/HidreamO1-i2L-v2](https://modelscope.cn/models/DiffSynth-Studio/HidreamO1-i2L-v2)：适配模型 Hidream-O1-Image

|输入样例1|输出样例1|输入样例2|输出样例2|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/0a1fd252-851f-414e-af24-4c656ab54277)|![Image](https://github.com/user-attachments/assets/96259993-e732-424d-bf07-9ca1ede27890)|![Image](https://github.com/user-attachments/assets/a78573a0-c2cf-4e33-ac21-276078e8cad3)|![Image](https://github.com/user-attachments/assets/8177e883-cfef-4e38-a528-cdef01a9f9b8)|

</details>

<details>
<summary>Diffusion-Templates: 插件化的可控生成框架</summary>

> 一个框架，让每一种可控生成能力都成为插件，让多个模型组合涌现出丰富生成能力。

* 论文：[Diffusion Templates: A Unified Plugin Framework for Controllable Diffusion](https://arxiv.org/abs/2604.24351)
* 项目主页：[GitHub](https://modelscope.github.io/diffusion-templates-web/)
* 文档参考：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
* 在线体验：[魔搭社区创空间](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* 模型集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)、[ModelScope 国际站](https://modelscope.ai/collections/DiffSynth-Studio/KleinBase4B-Templates)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/kleinbase4b-templates)
* 数据集：[ModelScope](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[ModelScope 国际站](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2)、[HuggingFace](https://huggingface.co/collections/DiffSynth-Studio/imagepulsev2)

|参考图|局部编辑|风格迁移|清晰度增强|
|-|-|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_reference.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Brightness_Edit_Inpaint.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Controlnet_Edit_SoftRGB.png)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Upscaler_Sharpness.png)|

</details>

<details>
<summary>SES: 用于奖励对齐图像生成的高效推理阶段缩放</summary>

> 以推理时间为代价，提高生成内容的质量。

- 论文：[Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation](https://arxiv.org/abs/2602.03208)
- 代码样例：[/docs/en/Research_Tutorial/inference_time_scaling.md](/docs/en/Research_Tutorial/inference_time_scaling.md)

|FLUX.1-dev|FLUX.1-dev + SES|Qwen-Image|Qwen-Image + SES|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/5be15dc6-2805-4822-b04c-2573fc0f45f0)|![Image](https://github.com/user-attachments/assets/e71b8c20-1629-41d9-b0ff-185805c1da4e)|![Image](https://github.com/user-attachments/assets/7a73c968-133a-4545-9aa2-205533861cd4)|![Image](https://github.com/user-attachments/assets/c8390b22-14fe-48a0-a6e6-d6556d31235e)|

</details>

<details>
<summary>VIRAL：基于DiT模型的类比视觉上下文推理</summary>

> 根据图1到图2的变化把图3转化为图4，图像编辑模型的能力涌现。

- 论文：[VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers
](https://arxiv.org/abs/2602.03210)
- 代码样例：[/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py](/examples/qwen_image/model_inference/Qwen-Image-Edit-2511-ICEdit.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA)

|Example 1|Example 2|Query|Output|
|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/380d2670-47bf-41cd-b5c9-37110cc4a943)|![Image](https://github.com/user-attachments/assets/7ceaf345-0992-46e6-b38f-394c2065b165)|![Image](https://github.com/user-attachments/assets/f7c26c21-6894-4d9e-b570-f1d44ca7c1de)|![Image](https://github.com/user-attachments/assets/c2bebe3b-5984-41ba-94bf-9509f6a8a990)|

</details>

<details>
<summary>AttriCtrl: 图像生成模型的属性强度控制</summary>

> 用数值属性也能精准地控制图像生成模型。

- 论文：[AttriCtrl: Fine-Grained Control of Aesthetic Attribute Intensity in Diffusion Models
](https://arxiv.org/abs/2508.02151)
- 代码样例：[/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py](/examples/flux/model_inference/FLUX.1-dev-AttriCtrl.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/AttriCtrl-FLUX.1-Dev)

|brightness scale = 0.1|brightness scale = 0.3|brightness scale = 0.5|brightness scale = 0.7|brightness scale = 0.9|
|-|-|-|-|-|
|![Image](https://github.com/user-attachments/assets/e74b32a5-5b2e-4c87-9df8-487c0f8366b7)|![Image](https://github.com/user-attachments/assets/bfe8bec2-9e55-493d-9a26-7e9cce28e03d)|![Image](https://github.com/user-attachments/assets/b099dfe3-ff1f-4b96-894c-d48bbe92db7a)|![Image](https://github.com/user-attachments/assets/0a6b2982-deab-4b0d-91ad-888782de01c9)|![Image](https://github.com/user-attachments/assets/fcecb755-7d03-4020-b83a-13ad2b38705c)|

</details>

<details>
<summary>AutoLoRA: 自动化的 LoRA 检索和融合</summary>

> LoRA 是需求和解决方案一体化的产物，如何更好地利用这些 LoRA？

- 论文：[AutoLoRA: Automatic LoRA Retrieval and Fine-Grained Gated Fusion for Text-to-Image Generation
](https://arxiv.org/abs/2508.02107)
- 代码样例：[/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py](/examples/flux/model_inference/FLUX.1-dev-LoRA-Fusion.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev)

||[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)|
|-|-|-|-|-|
|[LoRA 1](https://modelscope.cn/models/cancel13/cxsk)                              |![Image](https://github.com/user-attachments/assets/01c54d5a-4f00-4c2e-982a-4ec0a4c6a6e3)|![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|
|[LoRA 2](https://modelscope.cn/models/wy413928499/xuancai2)                       |![Image](https://github.com/user-attachments/assets/e6621457-b9f1-437c-bcc8-3e12e41646de)|![Image](https://github.com/user-attachments/assets/43720a9f-aa27-4918-947d-545389375d46)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|
|[LoRA 3](https://modelscope.cn/models/DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1)  |![Image](https://github.com/user-attachments/assets/4b7f721f-a2e5-416c-af2c-b53ef236c321)|![Image](https://github.com/user-attachments/assets/418c725b-6d35-41f4-b18f-c7e3867cc142)|![Image](https://github.com/user-attachments/assets/041a3f9a-c7b4-4311-8582-cb71a7226d80)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|
|[LoRA 4](https://modelscope.cn/models/hongyanbujian/JPL)                          |![Image](https://github.com/user-attachments/assets/802d554e-0402-482c-9f28-87605f8fe318)|![Image](https://github.com/user-attachments/assets/8c8f22fa-9643-4019-b6d7-396d8b7fed9a)|![Image](https://github.com/user-attachments/assets/b54ebaa4-31a7-4536-a2c1-496adba0c013)|![Image](https://github.com/user-attachments/assets/a640fd54-3192-49a0-9281-b43d9ba64f09)|

</details>

<details>
<summary>Nexus-Gen: 统一架构的图像理解、生成、编辑</summary>

> 如果一个模型集齐图像理解、图像生成、图像编辑能力，会发生什么？

- 详细页面：https://github.com/modelscope/Nexus-Gen
- 论文：[Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space](https://arxiv.org/pdf/2504.21356)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2), [HuggingFace](https://huggingface.co/modelscope/Nexus-GenV2)
- 数据集：[ModelScope Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/Nexus-Gen-Training-Dataset)
- 在线体验：[ModelScope Nexus-Gen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/Nexus-Gen)

![](https://github.com/modelscope/Nexus-Gen/raw/main/assets/illustrations/gen_edit.jpg)

</details>

<details>
<summary>ArtAug: 图像生成模型的美学提升</summary>

> 一个 LoRA，大幅提升细节和美感。

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

> 如何用分区图层控制画面内容的位置？

- 论文：[EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
- 代码样例：[/examples/flux/model_inference/FLUX.1-dev-EliGen.py](/examples/flux/model_inference/FLUX.1-dev-EliGen.py)
- 模型：[ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen), [HuggingFace](https://huggingface.co/modelscope/EliGen)
- 在线体验：[ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
- 数据集：[EliGen Train Set](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)

|实体控制区域|生成图像|
|-|-|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|

</details>

<details>
<summary>ExVideo: 视频生成模型的扩展训练</summary>

> 如果视频生成模型只能生成 25 帧，如何才能让它生成更长视频？

- 项目页面：[Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
- 论文：[ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
- 代码样例：请前往[旧版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/ExVideo)查看
- 模型：[ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1), [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

</details>

<details>
<summary>Diffutoon: 高分辨率动漫风格视频渲染</summary>

> 我管你这那的，我就是喜欢二次元！

- 项目页面：[Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
- 论文：[Diffutoon: High-Resolution Editable Toon Shading via Diffusion Models](https://arxiv.org/abs/2401.16224)
- 代码样例：请前往[旧版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/Diffutoon)查看

注意：Diffutoon 示例来自 2.0 之前的代码库，目前不包含在 `main` 分支中。请将上面的旧版本代码与对应版本的 DiffSynth-Studio 一起使用。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

</details>

<details>
<summary>DiffSynth: 本项目的初代版本</summary>

> 在没有视频生成模型的时代，如何利用图像生成模型处理视频？

- 项目页面：[Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/)
- 论文：[DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463)
- 代码样例：请前往[旧版本](https://github.com/modelscope/DiffSynth-Studio/tree/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/diffsynth)查看

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

</details>

## 联系我们

|Discord：https://discord.gg/Mm9suEeUDc|
|-|
|<img width="160" height="160" alt="Image" src="https://github.com/user-attachments/assets/29bdc97b-e35d-4fea-88d6-32e35182e458" />|

## 全部支持的模型

| 基础架构 | 模型 ID | 推理 | 低显存推理 | 全量训练 | 全量训练后验证 | LoRA 训练 | LoRA 训练后验证 |
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
| - | [PickScore](https://github.com/yuvalkirstain/pickscore) | [code](/examples/image_quality_metric/pickscore.py) | - | - | - | - | - |
| - | [ImageReward](https://github.com/zai-org/ImageReward) | [code](/examples/image_quality_metric/image_reward.py) | - | - | - | - | - |
| - | [HPSv2](https://github.com/tgxs002/HPSv2) | [code](/examples/image_quality_metric/hpsv2.py) | - | - | - | - | - |
| - | [HPSv3](https://github.com/MizzenAI/HPSv3) | [code](/examples/image_quality_metric/hpsv3.py) | - | - | - | - | - |
| - | [CLIP Score](https://github.com/openai/CLIP) | [code](/examples/image_quality_metric/clipscore.py) | - | - | - | - | - |
| - | [UnifiedReward 2.0](https://github.com/THUDM/UnifiedReward) | [code](/examples/image_quality_metric/unified_reward_2.py) | - | - | - | - | - |
| - | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) | [code](/examples/image_quality_metric/qwen_image_bench.py) | - | - | - | - | - |
| - | [UnifiedReward Edit](https://github.com/THUDM/UnifiedReward) | [code](/examples/image_quality_metric/unified_reward_edit.py) | - | - | - | - | - |
| - | [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor) | [code](/examples/image_quality_metric/aesthetic.py) | - | - | - | - | - |
| - | [FID](https://github.com/mseitzer/pytorch-fid) | [code](/examples/image_quality_metric/fid.py) | - | - | - | - | - |
