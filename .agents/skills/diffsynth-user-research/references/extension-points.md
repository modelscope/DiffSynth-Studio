# 研究扩展点与实现验证

适用于定位方法实现位置、修改计算图、内部机制与梯度研究。路径均相对于用户选择的 Studio checkout，先核对该版本；不需要为了项目实验先完成上游模型注册。

## 从假设找到最小改动层

沿实际链路阅读：`metadata → Dataset/operator → get_pipeline_inputs → units → model_fn → scheduler.step → decode`。训练再追踪 `loss → backward → optimizer → logger`。不是每个模型都经过全部环节，例如像素空间模型可能没有独立 VAE。

| 研究干预 | 优先实现位置 | 应核验的边界 |
| --- | --- | --- |
| 数据配对、裁剪、增强、重采样 | Dataset/operator、项目数据准备代码 | 监督目标和条件、实际加载类型、随机性和重复含义 |
| 条件编码/特征转换 | 新增或替换 PipelineUnit | 字段生产/消费、正负分支、输出是否被后续覆盖 |
| 每步条件注入、层/头消融、时间门控 | 项目 wrapper/model_fn 或模型 block | timestep尺度、返回值类型、完整参数传递、干预复位 |
| 时间网格、噪声路径、积分方法 | 项目 scheduler 与去噪循环 | 调用时是否重新设表、模型预测含义、训练/采样一致性 |
| 新监督目标、正则、奖励/偏好损失 | 项目 TrainingModule/loss | noising/target/weight、采样分布、梯度路径 |
| 多个角色/优化器、交替更新、EMA等新算法状态 | 项目专用 runner | 更新频率、累积同步、角色冻结、保存范围 |
| 新 Adapter/小模型 | 项目 nn.Module 与 Pipeline | 初始化、形状、保存加载及精度/设备策略 |
| 量化、attention或执行策略 | 已有 quant/attention/VRAM 配置；缺实现再扩展 | 实际生效路径、可微性、误差及完整成本 |

优先使用现有明确接口。`pipe.model_fn` 是可替换的调用入口，但并非统一 callback API；包装时保留目标 Pipeline 所有输入与返回结构，不能假定只有一个 Tensor。替换 Unit 时保留原顺序和必要行为。需要修改循环的算法可以复制精确模型的循环到项目中，不为追求少改代码而隐藏缺失逻辑。

## Unit 的字段与分支

读取 `diffsynth/diffusion/base_pipeline.py` 的 `PipelineUnit`、`PipelineUnitRunner`、`PipelineUnitGraph` 和目标 Unit：

- `input_params` 来自 shared；`input_params_posi/nega` 映射正负分支；`output_params` 和 `onload_model_names` 还影响训练图拆分。只声明字段或 loader 不表示模型消费了它。
- `seperate_cfg`（源码如此拼写）按 CFG 执行两侧；`take_over` 可同时接管三份输入字典。检查 cfg=1 时负侧处理是否跳过/复用，不能仅凭 negative 参数存在判断已执行负向计算。
- Unit 取输入时可能得到 None，应遵循目标实现的缺省契约。研究必须验证“有条件/无条件”“干预开/关”确实改变相应消费者，不能以未报错为生效证据。
- 调用模型的 Unit 要声明并激活对应组件；项目新增组件不能靠给 model_fn 增加一个字段就自动获得加载、训练或 offload 支持。

## 训练图与梯度

`requires_grad_(False)` 冻结参数仍允许对输入求梯度；`torch.no_grad()`、`inference_mode`、detach、转 PIL/NumPy/标量会切断相应路径。先画出本方法真正需要的路径，例如 `loss → frozen DiT → template cache → trainable controller`，只把无需梯度的计算移到预处理。

对于新可训练模块，有限试验需检查 loss 有限、目标参数进入 optimizer、预期活跃路径存在有限梯度、更新后相关参数变化、冻结参数未被更新。LoRA 零初始化、门控或未命中的 MoE expert 可能暂无梯度，不要求每个参数首步都非零。联合训练各组件时应分别检查，避免只有旧组件在更新。

`split_pipeline_units()` 根据声明图与可训练组件拆分。新增字段需核对 `GeneralUnit_RemoveCache` 的过滤及下游消费者；声明不全会丢失输入或错误缓存。当前 `GeneralUnit_TemplateForward.output_params` 只声明 `kv_cache`，新模板若输出其他字段，要验证/调整项目层图声明和保留规则，不能假定所有 cache 类型天然适配拆分训练。

声明依赖图不是autograd真实计算图的证明。改动Unit、过滤规则或拆分训练时，在相同数据、参数、噪声/timestep与随机条件下，按声明的精度容差比较在线/拆分路径的关键Tensor、loss及目标参数梯度；若方法有意改变计算，则比较关闭干预后的基线路径。不能只以字段齐全或forward成功宣称训练语义等价。

缓存仅用于语义匹配的冻结计算。更换编码器/预处理、需要重新采样的增强、prompt dropout、可训练特征等均可能要求重建或禁用缓存；保存配置身份和数据映射，不把冻结一次随机结果称为与在线实验等价。

## 机制分析与干预

可围绕层、head、token、时间步、模态和条件分支提出问题：特征/attention观察、激活或KV替换、消融、梯度敏感性、低秩与频谱分析、条件信息在不同噪声阶段的影响。它们是研究设计选项，不是框架已经提供的一键分析工具。

只保存所需统计，避免长期保留整张计算图。checkpoint可能重算 forward；hook调用次数不必等于去噪步骤数。offload wrapper、compile、packed tokens和分布式分片也可能改变观察位置，应标注原始模型层与实际执行对象、token/帧映射，并用小规格验证计数。attention后端未必返回权重；需要显式 attention map 时在项目实现可比较的路径，报告由此引入的数值/成本变化。

使用 wrapper/hook 的实验结束后恢复原入口、移除handle、清理cache/动态LoRA并复验基线。多个候选共用可变字典、scheduler或权重时，要隔离状态，避免上一候选污染下一候选。能用清晰项目类表达时，优先于不可追溯的全局 monkey patch。

## 原型与框架接入

项目可直接实例化自定义模型并挂载到项目 Pipeline，从零教程就是这条路径。`ModelPool` 自动识别已有 checkpoint 依赖注册配置，结构变化后不能继续伪装成旧 hash。正式发布支持时再参考 `docs/zh/Developer_Guide/{Integrating_Your_Model,Building_a_Pipeline,Training_Diffusion_Models,Enabling_VRAM_management}.md`，补注册/转换/模块映射等；项目原型只声明已经验证的能力。
