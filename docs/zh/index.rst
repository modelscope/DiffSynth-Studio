DiffSynth-Studio 文档
=====================

.. image:: https://img.shields.io/badge/license-Apache%202.0-blue
   :target: https://github.com/modelscope/DiffSynth-Studio/blob/main/LICENSE
   :alt: GitHub

.. image:: https://img.shields.io/github/stars/modelscope/DiffSynth-Studio?style=social
   :target: https://github.com/modelscope/DiffSynth-Studio
   :alt: GitHub Repo stars

欢迎来到 Diffusion 模型的魔法世界！`DiffSynth-Studio` 是由 `魔搭社区 <https://www.modelscope.cn/>`_ 团队开发和维护的开源 Diffusion 模型引擎。我们期望构建一个通用的 Diffusion 模型框架，以框架建设孵化技术创新，凝聚开源社区的力量，探索生成式模型技术的边界！

.. raw:: html

   <details>
   <summary>文档阅读导引</summary>
   
   ```mermaid
   graph LR;
       我想要使用模型进行推理和训练-->sec1[Section 1: 上手使用];
       我想要使用模型进行推理和训练-->sec2[Section 2: 模型详解];
       我想要使用模型进行推理和训练-->sec3[Section 3: 训练框架];
       我想要基于此框架进行二次开发-->sec3[Section 3: 训练框架];
       我想要基于此框架进行二次开发-->sec4[Section 4: 模型接入];
       我想要基于此框架进行二次开发-->sec5[Section 5: API 参考];
       我想要基于本项目探索新的技术-->sec4[Section 4: 模型接入];
       我想要基于本项目探索新的技术-->sec5[Section 5: API 参考];
       我想要基于本项目探索新的技术-->sec6[Section 6: 学术导引];
       我遇到了问题-->sec7[Section 7: 常见问题];
   ```

   </details>

Section 1: 上手使用
-------------------

本节介绍 ``DiffSynth-Studio`` 的基本使用方式，包括如何启用显存管理从而在极低显存的 GPU 上进行推理，以及如何训练任意基础模型、LoRA、ControlNet 等模型。

.. toctree::
   :maxdepth: 1

   Pipeline_Usage/Setup
   Pipeline_Usage/Model_Inference
   Pipeline_Usage/VRAM_management
   Pipeline_Usage/Model_Training
   Pipeline_Usage/Environment_Variables
   Pipeline_Usage/GPU_support

Section 2: 模型详解
-------------------

本节介绍 ``DiffSynth-Studio`` 所支持的 Diffusion 模型，部分模型 Pipeline 具备可控生成、并行加速等特色功能。

.. toctree::
   :maxdepth: 1

   Model_Details/FLUX
   Model_Details/Wan
   Model_Details/Qwen-Image
   Model_Details/FLUX2
   Model_Details/Z-Image

Section 3: 训练框架
-------------------

本节介绍 ``DiffSynth-Studio`` 中训练框架的设计思路，帮助开发者理解 Diffusion 模型训练算法的原理。

.. toctree::
   :maxdepth: 1

   Training/Understanding_Diffusion_models
   Training/Supervised_Fine_Tuning
   Training/FP8_Precision
   Training/Direct_Distill
   Training/Split_Training
   Training/Differential_LoRA

Section 4: 模型接入
-------------------

本节介绍如何将模型接入 ``DiffSynth-Studio`` 从而使用框架基础功能，帮助开发者为本项目提供新模型的支持，或进行私有化模型的推理和训练。

.. toctree::
   :maxdepth: 1

   Developer_Guide/Integrating_Your_Model
   Developer_Guide/Building_a_Pipeline
   Developer_Guide/Enabling_VRAM_management
   Developer_Guide/Training_Diffusion_Models

Section 5: API 参考
-------------------

本节介绍 ``DiffSynth-Studio`` 中的独立核心模块 ``diffsynth.core``，介绍内部的功能是如何设计和运作的，开发者如有需要，可将其中的功能模块用于其他代码库的开发中。

.. toctree::
   :maxdepth: 1

   API_Reference/core/attention
   API_Reference/core/data
   API_Reference/core/gradient
   API_Reference/core/loader
   API_Reference/core/vram

Section 6: 学术导引
-------------------

本节介绍如何利用 ``DiffSynth-Studio`` 训练新的模型，帮助科研工作者探索新的模型技术。

.. toctree::
   :maxdepth: 1

   Research_Tutorial/train_from_scratch

Section 7: 常见问题
-------------------

本节总结了开发者常见的问题，如果你在使用和开发中遇到了问题，请参考本节内容，如果仍无法解决，请到 GitHub 上给我们提 issue。

.. toctree::
   :maxdepth: 1

   QA

.. note::
   本文档持续更新中，如有任何问题或建议，请在 `GitHub Issues <https://github.com/modelscope/DiffSynth-Studio/issues>`_ 中提出。
