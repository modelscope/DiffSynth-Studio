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

[DMD2](https://arxiv.org/abs/2405.14867)（Distribution Matching Distillation）是一种结合分布匹配与 GAN 的模型蒸馏训练方案。训练时，冻结的教师模型提供目标分布的预测，fake-score 模型估计学生的生成分布，两者共同构造学生的分布匹配梯度；GAN 损失进一步利用真实图像提供训练信号，提升蒸馏模型生成图像的细节。

训练中包含三种模型角色：

* 学生模型（student）：通过普通 Euler 轨迹生成样本，只对随机选中的采样步骤保留梯度，优化分布匹配和 GAN 损失。
* 教师模型（teacher）：冻结参数，提供目标分布的预测，并使用 `--dmd2_teacher_cfg` 控制 CFG。
* fake-score 模型：在学生生成样本上学习去噪，追踪学生的生成分布。GAN 分类头连接到 fake-score 模型的中间特征，训练时同时区分真实图像和学生生成图像。

真实图像只用于 GAN，为恢复图像细节提供训练信号，不参与分布匹配损失。关闭 GAN 时，数据集中的图像仅作为 `height` 和 `width` 的占位，用于确定生成尺寸，不会被编码，其像素内容也不会参与训练。

以 FLUX.2、Qwen-Image-2.1、MOSS-Image 和 FLUX.3 为例，我们提供了以下训练及验证代码：

|基础模型|可训练参数|训练代码|训练后验证|蒸馏前步数|蒸馏后步数|
|-|-|-|-|-|-|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/dmd2/train_FLUX.2-klein-base-4B-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/dmd2/validate_FLUX.2-klein-base-4B-8steps_lora.py)|50|8|
|[black-forest-labs/FLUX.2-klein-base-4B](https://www.modelscope.cn/models/black-forest-labs/FLUX.2-klein-base-4B)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/dmd2/train_FLUX.2-klein-base-4B-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/special/dmd2/validate_FLUX.2-klein-base-4B-8steps_full.py)|50|8|
|[black-forest-labs/flux-3-image-klein-base](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/special/dmd2/train_flux-3-image-klein-base-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/special/dmd2/validate_flux-3-image-klein-base-8steps_lora.py)|50|8|
|[black-forest-labs/flux-3-image-klein-base](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/special/dmd2/train_flux-3-image-klein-base-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/special/dmd2/validate_flux-3-image-klein-base-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/train_Qwen-Image-2.1-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/validate_Qwen-Image-2.1-8steps_lora.py)|50|8|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/train_Qwen-Image-2.1-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/validate_Qwen-Image-2.1-8steps_full.py)|50|8|
|[Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/train_Qwen-Image-2.1-Turbo-3steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/validate_Qwen-Image-2.1-Turbo-3steps_lora.py)|8|3|
|[Qwen/Qwen-Image-2.1-Turbo](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/train_Qwen-Image-2.1-Turbo-3steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/special/dmd2/validate_Qwen-Image-2.1-Turbo-3steps_full.py)|8|3|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|LoRA|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/dmd2/train_MOSS-Image-SFT-8steps_lora.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/dmd2/validate_MOSS-Image-SFT-8steps_lora.py)|50|8|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|Full|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/dmd2/train_MOSS-Image-SFT-8steps_full.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/special/dmd2/validate_MOSS-Image-SFT-8steps_full.py)|50|8|

使用这些示例时，需要注意：

* `--dmd2_student_sigmas` 和 `--dmd2_student_timesteps` 指定学生时间序列，两者应对应。可以使用 `examples/dev_tools/compute_dmd2_sigmas.py` 计算；验证时应使用与训练一致的时间表，学生采样使用 CFG=1。
* `--dmd2_teacher_cfg` 控制教师预测的 CFG 强度，决定学生拟合的引导目标；设为 1 时不进行额外 CFG 引导。该参数只作用于教师，学生和 fake-score 的预测使用 CFG=1，验证时不需要沿用教师的 CFG。
* `--dmd2_fake_warmup_steps` 控制初始 critic-only 更新次数，设为 0 可关闭 warmup；`--dmd2_fake_updates_per_generator` 控制 warmup 后的更新比例。这里的步数按完成梯度累积后的优化窗口计数。
* `--learning_rate` 控制学生学习率；`--dmd2_fake_learning_rate` 控制 fake-score 和分类头学习率，未指定时跟随学生学习率。
* 默认启用 GAN，训练数据需包含真实图像；`--dmd2_disable_gan` 可关闭 GAN，此时不编码真实图像。启用 GAN 时，`--dmd2_feature_module_of_dit_for_gan` 指定提取特征的主干层，`--dmd2_classifier_channels` 控制分类头宽度。不同模型的特征布局由各自的训练模块适配。
* `--dmd2_noise_sampling` 支持 `uniform` 和 `beta`，后者使用 `Beta(2.5, 1)`，更侧重较大 sigma 对应的高噪声区间，并按 `--dmd2_min_timestep`、`--dmd2_max_timestep` 限制噪声范围。该选项作用于 DM 和 fake 去噪，GAN 始终使用均匀加噪。
* checkpoint 只导出学生参数。LoRA 可通过 `--lora_checkpoint` 热启动，全量学生可通过 `--resume_from_checkpoint` 加载已有权重；这不恢复 fake-score、分类头、optimizer 或随机状态，不是完整断点续训。验证脚本中的 checkpoint 路径需替换为实际训练输出。
* 当前 DMD2 训练不支持 DeepSpeed 和 `--enable_model_cpu_offload`。

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

> 如何将多步的 teacher 模型输出合成为一个输出？
> 
> 假设当前 student 模型的 sigma 区间为 [`sigma_start`, `sigma_end`]，teacher 模型会在这个区间内做多步推理。参数 `--opd_align_at_sigma_zero` 控制 teacher 监督目标对齐到哪个端点：默认情况下，目标对齐到区间的右端点 `sigma_end`，让 student 在这一步的更新恰好到达教师的区间终点（`x_teacher(sigma_end)` 由 teacher 轨迹的最后一步外推得到）；开启参数 `--opd_align_at_sigma_zero` 后，端点改为 `sigma=0`，也就是用"当前噪声样本到干净样本"的平均速度作为监督信号。该参数只改变 teacher 的监督目标，不改变 student 前向；由于最后一个区间的 `sigma_end` 本身就是 0，两种设置在该区间上等价。

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
