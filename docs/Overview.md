# DiffSynth-Studio 文档

`DiffSynth-Studio` 旨在构建一个通用的 Diffusion 模型框架，支持主流 Diffusion 模型的推理和训练，孵化模型技术的创新成果。

## Section 1: 上手使用

本节介绍 `DiffSynth-Studio` 的基本使用方式，包括如何启用显存管理从而在极低显存的 GPU 上进行推理，以及如何训练任意基础模型、LoRA、ControlNet 等模型。

* [快速开始](./Pipeline_Usage/Quick_Start.md)【TODO】
* [模型推理](./Pipeline_Usage/Model_Inference.md)【TODO】
* [显存管理](./Pipeline_Usage/VRAM_management.md)
* [模型训练](./Pipeline_Usage/Model_Training.md)【TODO】
* [环境变量](./Pipeline_Usage/Environment_Variables.md)

## Section 2: 模型接入

本节介绍如何将模型接入 `DiffSynth-Studio` 从而使用框架基础功能，帮助开发者为本项目提供新模型的支持，或进行私有化模型的推理和训练。

* [接入模型结构](./Developer_Guide/Integrating_Your_Model.md)
* [接入 Pipeline](./Developer_Guide/Building_a_Pipeline.md)
* [接入细粒度显存管理](./Developer_Guide/Enabling_VRAM_management.md)
* [接入模型训练](./Developer_Guide/Training_Diffusion_Models.md)

## Section 3: API 参考

本节介绍 `DiffSynth-Studio` 中的独立核心模块 `diffsynth.core`，介绍内部的功能是如何设计和运作的，开发者如有需要，可将其中的功能模块用于其他代码库的开发中。

* [`diffsynth.core.attention`](./API_Reference/core/attention.md): 注意力机制实现
* [`diffsynth.core.data`](./API_Reference/core/data.md): 数据处理算子与通用数据集
* [`diffsynth.core.gradient`](./API_Reference/core/gradient.md): 梯度检查点
* [`diffsynth.core.loader`](./API_Reference/core/loader.md): 模型下载与加载
* [`diffsynth.core.vram`](./API_Reference/core/vram.md): 显存管理

## Section 4: 学术导引

本节介绍如何利用 `DiffSynth-Studio` 训练新的模型，帮助科研工作者探索新的模型技术。

* 从零开始训练模型【TODO】
* 推理改进优化技术【TODO】
* 设计可控生成模型【TODO】
* 创建新的训练范式【TODO】
