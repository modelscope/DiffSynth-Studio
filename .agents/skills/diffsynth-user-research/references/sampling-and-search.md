# 采样、引导与推理搜索

适用于时间表/求解器/CFG实验、候选搜索以及可微推理优化。框架入口为 `diffsynth/diffusion/{flow_match,ddim_scheduler,base_pipeline}.py`、目标 Pipeline 的去噪循环与 `docs/zh/Research_Tutorial/inference_time_scaling.md`；先核对活动版本。

## 采样和引导实验

| 探索方向 | 具体干预与对照 | 核验点 |
| --- | --- | --- |
| 时间表 | 步数、sigma分布、shift、按分辨率调整、分阶段预算 | 比较实际时间表及真实前向次数，避免调用时被重设 |
| 求解器 | Euler与项目新积分方法、确定性与随机/ancestral更新 | 新方法可能每步多次评估；固定NFE或时间再比较质量 |
| 噪声路径 | 初始频谱、噪声尺度、再加噪、部分去噪 | 保存实际噪声/采样RNG，检查与训练分布的偏离 |
| 引导 | CFG、随timestep变化、正负条件、分支特定LoRA | CFG与embedded guidance含义不同；记录分支真实执行 |
| 分阶段/自适应 | early exit、不同阶段分配步数、局部重生成 | 项目循环需实现停止条件；分别记录各阶段成本和失败 |

多数 Pipeline 在 `__call__` 开始调用 `scheduler.set_timesteps()`；在外部提前写 `sigmas` 可能无效。检查调用链后，通过项目 scheduler、精确模型循环或已有自定义时间表参数实现，记录运行中真正使用的序列。只增加未消费的参数不构成方法。

不要假定 `timestep=sigma*1000` 对所有模型成立。当前 FlowMatch 各模板有不同时间尺度/偏移，DDIM 还区分 epsilon/v_prediction。改变噪声插值或模型预测定义时，需同步推导并实现 noising、training target、预测转换和 step，不能只替换采样公式后声称理论一致。现有 `AncestralFlowMatchScheduler` 等只能在兼容循环下复用。

## 无梯度候选搜索

先定义优化对象、候选生成规则和评分目标，再选随机、多目标或其他项目搜索策略，不为使用现成算法替换用户问题。噪声搜索从目标Pipeline的真实初始化构造推导shape；例如Flux2可用 `generate_noise` / `initial_noise`，但会在Unit里转成序列，不把外部空间噪声与内部packed latent混用。

将单次普通采样作为低成本锚点，random/best-of-N与目标搜索方法做等预算比较，必要时再比较把相同计算用于增加单图步数。初始候选、额外渲染/复评、缓存命中和失败分别记账，区分目标函数调用、真实生成、评分次数与总时间；重试不得突破总上限。算法内部、初始条件和评分器的随机性分别控制，只固定一份base噪声不足以复现整个搜索。

项目目标函数包装器记录候选ID、输入/噪声或可重建信息、输出、原始分数和失败。排序值须为有限标量；事先规定异常、NaN/Inf和失败候选的处理，保留状态并计入尝试预算，不静默丢弃或用任意零分替代。没有有效候选时报告搜索失败，不把未评分的默认返回值冒充最优结果。

现成搜索示例可查 `diffsynth/utils/ses/ses.py:ses_search`：当前SES固定首个样本的高频，低频搜索从零均值、单位方差开始，而非围绕base低频做局部扰动；base不自动评分，内部CPU随机性需另控。仅在适配的单样本空间latent上复用；更改初始化/先验属于新干预，其他参数和依赖按需查源码，不把此算法当作所有搜索任务的默认路线。

评分器可用 `diffsynth.metrics` 中的具体实现，但输入/返回值不同。PickScore的 `compute(prompt, images)` 返回列表，`[0]`只适用于确认单图返回的情况；不要混用教程自定义PickScore类与框架Metric接口。评分权重、预处理、设备和耗时另计，资源紧张时分批/顺序评分。

## 可微优化与奖励梯度

优化noise/embedding/控制参数、可微轨迹、解码后reward等方法，需要自己的Tensor计算路径：参数 → 条件编码/去噪循环 → 可微decode/预处理 → Tensor目标。复用真实Unit/model_fn/scheduler时逐段核对梯度、dtype、shape及更新规则。

普通 Pipeline `__call__` 通常有no_grad，`vae_output_to_image` 转uint8/PIL；PickScore等Metric用no_grad，并通过 `Metric.tensor_to_list` detach。外层 `torch.enable_grad()`、给initial_noise设requires_grad，或对最终Python分数重新包装Tensor，都不能接回这些路径。

需要在项目实现可微评分预处理与模型调用，冻结评分器/底模参数但保留对输入的梯度；确认预处理没有PIL/NumPy转换或不可微操作。若优化多个采样步骤，说明截断/完整反传及近似，考虑激活和多步图的资源成本。先验证一个明确活跃的优化变量能得到有限梯度并更新，再跑有限优化；最终用常规生成路径复现并独立评价。

选参prompt与最终测试prompt分开；保存所有方法的候选和失败，评价偏好、多样性、条件遵循与人工样例。针对同一reward搜索到更高分只证明该目标改善，不能推出通用质量提高。推理搜索也不等于训练；要把搜索产物用于学习，继续阅读 [训练目标与蒸馏](objectives-and-distillation.md)。
