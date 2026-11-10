# 训练目标、蒸馏与训练状态

适用于修改训练目标、奖励/偏好学习、OPD/DMD2及训练算法比较。核对 `diffsynth/diffusion/{loss,training_module,runner,logger,parsers,dmd2}.py`、目标训练脚本及 `docs/zh/Training/{Understanding_Diffusion_models,Distillation,Direct_Distill,Differential_LoRA}.md`。专用脚本的能力不能从通用 parser 推断。

## 修改目标的实现路线

| 问题 | 可研究因素 | 需要实际实现/验证 |
| --- | --- | --- |
| 去噪学习 | timestep分布/边界、noise scale、loss weighting、噪声与样本配对 | 原始数据 → noising → 模型预测 → target/weight；采样分布变化的影响 |
| 局部与条件学习 | mask/区域权重、条件dropout、条件一致性、保真与编辑目标 | 明确监督位置、mask对齐、归一化及无条件分支 |
| 正则与知识保持 | 特征/感知约束、底模正则、旧任务重放、蒸馏项 | 新旧任务划分、项的权重和梯度、泛化及遗忘 |
| 多任务/多模态 | 混合数据、不同loss权重、动态权重与更新比例 | 每项数据/目标契约、各分支梯度和独立指标 |
| 偏好/奖励 | 搜索构造优劣对、奖励加权、DPO类目标、可微reward | 真实算法公式、参考模型/采样、所需梯度及项目训练入口 |
| 优化算法 | trainable组件、LoRA位置/rank、学习率、optimizer、EMA/交替更新 | 有效参数与更新数、初始化、状态保存及公平预算 |

当前 `FlowMatchSFTLoss` 随机取timestep、对input_latents加噪、调用model_fn，以scheduler target的MSE乘training_weight；部分字段如noise_scale或timestep边界在loss里有消费者，但不代表目标CLI已暴露它们。需要在项目明确接通输入、目标与参数，再验证，不把存在同名变量当作用户可直接传flag。

loss名不决定它适用于所有模型。DDIM/FlowMatch的预测含义、audio/video tuple、首帧条件和packed布局都需按实际实现适配。新感知/reward项通常要用可微decode及预处理，不能直接调用带no_grad或返回Python数值的Metric；见 [采样与推理搜索](sampling-and-search.md)。

当前没有查到通用DPO/GRPO训练入口，研究教程只是提到用搜索结果继续做DPO/差分训练。活动checkout若新增实现应重新检查；否则在项目实现训练模块和必要runner，不能使用猜测的 `--task dpo`。偏好对需有同一prompt/条件的真实winner/loser和构造规则；差分LoRA与DPO不是同一算法，不能互相冒称。

## 选择蒸馏起点

| 路线 | 当前入口与状态 | 研究中应明确的差异 |
| --- | --- | --- |
| OPD | `examples/{flux2,qwen_image_21,moss_image}/model_training/special/opd/` | 学生轨迹采样，教师在对应区间提供监督；LoRA/全量是不同实现 |
| DMD2 | `diffsynth/diffusion/dmd2.py` 和 `examples/{flux2,flux3,qwen_image_21,moss_image}/model_training/special/dmd2/` | student、冻结teacher、fake-score与可选GAN分类头，专用多优化器runner |
| Direct Distill | 通用loss及部分目标train.py任务表 | 文档标为已停止维护，保留作历史复现/研究对照，默认新方案优先核对OPD |
| 轨迹模仿 | `loss.py:TrajectoryImitationLoss`、`examples/z_image/model_training/special/trajectory_imitation/` | 实验性、模型特定；核对teacher、调度、LPIPS依赖和验证配置，不当成通用现成方案 |

上述家族目录仅是导航，精确变体、数据条件、LoRA/全量支持需读取对应shell和验证脚本。比较方法时先确认目标checkout有可执行配方；有loss名字不代表普通train.py接受同名task。

## OPD

Flux2使用独立 `train_opd_lora.py` / `train_opd_full.py` 和 `train_FLUX.2-klein-base-4B-8steps_{lora,full}.sh`，验证文件实际拼作 `valudate_FLUX.2-klein-base-4B-8steps_{lora,full}.py`。保留真实路径；其他家族按存在文件匹配。

读取teacher/student模型构造、角色切换、`fetch_trajectory`、sigma对齐、学生前向与loss。LoRA实现可通过切换adapter scale复用底模，全量实现另持有teacher；记录构造/热启动顺序下的实际teacher身份，不能只按模型名推断。

可研究：学生步数与sigma网格、teacher/student CFG、选择多少轨迹点、跳过多少高噪声点、`opd_align_at_sigma_zero` 的端点目标、LoRA目标层与rank、渐进蒸馏、不同任务/条件的混合。`--opd_num_distill_steps`、`--opd_min_distill_idx` 与轨迹长度需一致，并验证选择范围与监督分母有效。多点loss当前求和，改变点数会同时改变loss尺度，应说明归一化或其对有效学习率的影响。

同目录 `compute_sigmas.py` 根据scheduler和extra_kwargs计算序列。示例50→8步、FLUX.2和dynamic_shift_len=4096不是通用默认；保存实际sigmas/timesteps、shift、CFG、尺寸及checkpoint关系，训练后推理显式使用匹配的学生调度。不能只减少 `num_inference_steps` 就宣称蒸馏完成。

## DMD2

先从精确模型专用 `special/dmd2/train.py` 继承/适配，保留 `prepare_batch`、feature提取与reshape、teacher/fake/student角色和多优化器runner。GAN hook依赖真实模型层及特征布局，不可把一个模型的feature path套到另一个模型。

可研究GAN开关及权重、fake warmup、critic/student更新比例、各自学习率、uniform/beta噪声采样与范围、教师CFG、学生时间表、LoRA/全量参数预算。当前GAN需要真实图像；关闭GAN不编码真实图像，但示例仍用图像确定尺寸，不自动变成无需该字段的text-only Dataset。

当前 `launch_dmd2_training_task` 明确拒绝DeepSpeed与model CPU offload；不能直接复制SFT的低显存组合。预算包含teacher/fake/student、可选分类头、双optimizer、轨迹和反向；LoRA分支可复用权重，不能简单断言一定有三份完整底模，也不能按单模型推理估算峰值。

teacher CFG只作用于教师；学生/fake预测使用CFG=1，验证遵循学生时间表和CFG设置。保存的是学生权重：不恢复fake-score、分类头、optimizers或RNG。`--lora_checkpoint` 或全量的 `--resume_from_checkpoint` 是学生热启动；完整续训需在项目runner保存/恢复全部算法状态。

## 预算、保存与恢复

通用runner的DataLoader按单样本使用，扩大有效batch主要通过累积/多卡；不得凭空加batch_size参数并假设已有模型支持。比较时记录microbatch、world size、累积、实际样本数及有效更新数，定制batching需验证真实模型/数据布局。

通用runner每个microbatch调用 `ModelLogger.on_step_end`，DMD2在sync_gradients时调用；相同step文件名不代表相同优化更新数，DMD2还要单列generator和critic更新。项目step限制若指优化更新，应在同步边界计数，兼顾梯度溢出导致的跳步，并在所有rank一致退出。`save_steps` 是保存间隔，不是停止条件。

`ModelLogger` 默认只导出可训练参数，不自动保存完整训练状态。项目新runner若承诺完整恢复，应覆盖模型/Adapter、optimizer、LR scheduler、计数、RNG、数据顺序/进度及teacher/EMA/fake/critic等算法状态；可使用兼容的Accelerate状态接口或项目实现，需核对分布式与自定义对象。只需验证推理时，权重重载即可，但不要称为完整续训。

做有限训练、保存、全新进程加载并验证相同固定输入；需要完整恢复时再验证恢复后的更新连续性和缺失项。报告训练成本与学生推理成本，分别比较未蒸馏底模多步、底模直接少步、学生少步，以及必要算法消融。不能只拿教师多步与学生少步比较速度就证明蒸馏质量有效。
