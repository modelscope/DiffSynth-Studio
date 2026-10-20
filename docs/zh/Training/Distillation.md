# 蒸馏加速训练

## 什么是蒸馏加速训练？

Diffusion 模型的推理过程通常需要多步迭代，在提升生成效果的同时也让生成过程变得缓慢。通过蒸馏加速训练，可以减少生成清晰内容所需的步数。蒸馏加速训练技术的本质训练目标是让少量步数的生成效果与大量步数的生成效果对齐。

在训练框架的设计中，我们需要在以下几个方面评估蒸馏加速训练方案的可行性：

* 通用性：训练方案适用于大多数框架内支持的 Diffusion 模型，而非只能对某个特定模型生效，这是代码框架建设的基本要求。
* 稳定性：训练方案需保证训练效果稳定，不需要人工进行精细的参数调整。
* 简洁性：训练方案不会引入额外的复杂模块，根据奥卡姆剃刀（[Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor)）原理，复杂解决方案可能引入潜在风险。
* 生成效果：训练得到的模型在少步数下的生成效果与原模型在多步数下的生成效果相似。

我们尝试了多种蒸馏加速训练方案，实验得到的经验如下表所示：

|训练方案|通用性|稳定性|简洁性|生成效果|
|-|-|-|-|-|
|Direct Distill|★★★|★★★|★★★|★|
|DMD2|★|★|★|★★★|
|OPD（推荐）|★★★|★★|★★★|★★★|

> **现阶段，我们强烈建议开发者使用 OPD 进行蒸馏加速训练。**

## Direct Distill

Direct Distill 是我们设计的端到端直接蒸馏训练方案，用原模型生成采样轨迹，再训练模型以短步数逼近采样轨迹，详见[端到端直接蒸馏加速](./Direct_Distill.md)。

## DMD2

[DMD2](https://arxiv.org/abs/2405.14867)（Distribution Matching Distillation） 是将 Diffusion 与 GAN 两者理论结合起来的产物，其训练依赖手动构造的判别器模型，我们仅在 FLUX.2 模型上适配了 DMD2 训练，详见[样例代码](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/flux2/model_training/special/dmd2)。

## OPD

OPD（On-Policy Distillation）是一种在线蒸馏训练方案，包括 [DiffusionOPD](https://arxiv.org/abs/2605.15055)、[DanceOPD](https://arxiv.org/abs/2606.27377) 等，其算法伪代码如下所示：

```python
with torch.no_grad():
    trajectory = sample_trajectory(student_model)
input = trajectory[randint]
student_output = student_model(input)
with torch.no_grad():
    teacher_output = teacher_model(output) # Multi-step inference
loss = torch.nn.functional.mse(student_output, teacher_output)
```

在 DiffusionOPD 变体中，`input = trajectory[randint]` 被替换为一个 for 循环，用轨迹中的每一个点进行训练；在 DanceOPD 变体中，`input` 仅在低噪声含量阶段进行采样，避免模型训练后的输出出现较大差异。在我们的样例代码中，可通过参数 `--opd_num_distill_steps` 和 `--opd_min_distill_idx` 分别实现这两个变体。

与 Direct Distill 相比，OPD 从学生模型出发进行采样和更新，而不是预先用教师模型生成轨迹，这让训练出来的模型效果更好；与 DMD2 相比，OPD 不需要根据模型结构设计判别器模型，因此通用型很强，且不需要谨慎地调整各种超参数。我们以 FLUX.2、Qwen-Image-2.1 和 MOSS-Image 为例，提供了 OPD 训练的样例代码：

|基础模型|可训练参数|训练代码|训练后验证|蒸馏前步数|蒸馏后步数|
|-|-|-|-|-|-|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/train_FLUX.2-klein-base-4B-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/valudate_FLUX.2-klein-base-4B-8steps_lora.py)|50|8|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|全量|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/train_FLUX.2-klein-base-4B-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/opd/valudate_FLUX.2-klein-base-4B-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-8steps_lora.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|全量|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/train_Qwen-Image-2.1-Turbo-3steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/opd/validate_Qwen-Image-2.1-Turbo-3steps_lora.py)|8|3|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/validate_MOSS-Image-SFT-8steps_lora.py)|50|8|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|全量|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/opd/validate_MOSS-Image-SFT-8steps_full.py)|50|8|

如果需要，你可以轻易地仿照样例代码实现其他模型的 OPD 训练。得益于框架的通用性设计，你甚至可以用任意输入格式的数据集进行训练（例如文生图与图像编辑数据集混合），甚至可以设计新的 Adapter 架构模型专门用来做步数蒸馏。

[Qwen-Image-2.1-TurTurbo](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-2.1-TurTurbo) 是我们训练的蒸馏加速模型，用于验证框架中训练方案的效果。以下是我们在实验中得到的实践性经验，供参考：

* OPD 的训练收敛速度非常快，通常 2GPU*h 内即可产生效果。
* 在 LoRA 训练中，LoRA 的位置应当覆盖输入层、输出层，以及处理 timestep 的层，因为蒸馏加速训练会大幅改变模型的输出形式。
* 以 OPD 训练得到的 LoRA 模型，在少数情况下，适当增加 LoRA 的权重（在权重上乘一个大于 1 的系数）可以更好地激发模型能力。
* OPD 的训练过程并非绝对稳定，采样中产生的误差可能会累积到模型内部，因此并非训练越久越好，必要时可多次训练取平均权重以增加稳定性。
* 当教师模型和学生模型的步数差异很大（例如 50 和 4）时，训练效果并不好，此时可以考虑渐进蒸馏（例如先从 50 蒸馏到 8，再将 8 步模型作为教师模型蒸馏到 4）。
* 学生模型的 sigmas 分布与生图风格相关，当 sigmas 在高噪声阶段停留比较多时，画面内容倾向于变简单；当 sigmas 在低噪声阶段停留比较多时，画面细节倾向于变丰富。
* 启用 CFG 可以减少生成内容的多样性，从而减小训练难度。

## 未来工作

蒸馏加速训练在稳定性、模型效果方面还有很大提升空间，我们会继续优化框架中算法的训练效果。
