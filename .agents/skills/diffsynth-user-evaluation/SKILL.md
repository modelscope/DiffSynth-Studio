---
name: diffsynth-user-evaluation
description: >
  Evaluate existing or newly generated images, video, audio and inference results.
  Lead standalone quality scoring, paired model/LoRA/checkpoint comparisons and
  inference speed or memory benchmarks using DiffSynth-Studio.
  Research experiments are led by diffsynth-user-research; ordinary generation
  by diffsynth-user-inference. Not for model-integration numerical equivalence tests.
  已有生成结果评价、指标计算、模型对比和性能基准测试。
---

# DiffSynth-Studio 效果与性能评估

先定义比较问题和有效指标，再生成或读取对照。运行正确、内容质量、数值差异与速度是不同证据；单个 reward 不能回答所有问题。

独立评分或基准测试由本 Skill 主导；论文复现、算法改进及训练消融由 Research 主导，本 Skill 提供评价细节。普通生成和产物完整性检查由 Inference 处理，不把本 Skill 设为所有推理或训练任务的必经步骤。

## 1. 写清比较对象和问题

识别评价任务、输入模态和必要的 prompt/源图等条件，明确关心编辑指令与保真、生成偏好、少步质量、速度还是资源。比较模型/LoRA/checkpoint/量化/采样配置时记录候选身份、基础模型版本及可取得的训练信息；只计算已有图片的 LPIPS/PickScore 等指标时，不要求用户先指定生成模型。来源未知的信息如实注明，仍须确定评分器权重、版本与预处理。

预先确定独立输入、条件、seed 列表、输出规格、主指标和评价预算。机制消融尽量固定非研究因素；不同方法比较允许各自合理且预先确定的配置，记录步数、CFG 等差异及选参预算。根据比较目标统一关键预算，例如真实模型前向次数（NFE）、评分次数或总耗时，不强求所有方法同时具有相同步数、CFG 或候选数；搜索还要计入候选生成、复评和失败成本。选参使用过的数据不要作为唯一最终验证集。

已有输出先利用，不重复训练或下载无关模型。若只要求评价协议，交付数据组织、脚本和空结果表，不顺带启动生成。

需要新生成对照时，本 Skill 保持比较协议和分析的主导权，按需读取同级 [diffsynth-user-inference](../diffsynth-user-inference/SKILL.md) 的模型加载、输入处理、低显存和产物检查规则，不另建通用推理流程。只要求评价已有文件时不启动生成；缺少必要输入或模型时明确未执行项。专项不可用时依据活动源码完成必要部分，不虚构已运行结果。

## 2. 选择能够回答问题的评价

在当前 `diffsynth/metrics/__init__.py`、各指标实现和 `examples/image_quality_metric/` 核对接口、对象类型、返回值与评分权重。下面的接口说明是源码导航，运行时仍需与活动版本对应。

| 问题 | 可用方法 | 解释边界 |
| --- | --- | --- |
| 文本与图像对齐/偏好 | PickScore 等对应评分器 | 评分器偏好不等于通用真实质量 |
| 编辑 | 源图/结果/指令的成对评价与人工检查 | 同时看修改是否正确和非目标内容保真 |
| 两组分布 | 足量参考与生成样本的 FID | 不是单图评分，不对极小样本夸大结论 |
| 成对图像差异 | LPIPS 等 | 差异小不自动表示结果更好 |
| 视频/音频 | 时序、运动、音质等任务评价 | 图像 reward 不能替代模态专属维度 |

加载评分器也要占 GPU/RAM 和时间。资源紧张时顺序评分；评分模型下载、预处理与推理成本单独记录。

FID 即使在极少样本上返回数值，也不证明估计可靠。比较候选时固定参考集、采样规模、特征提取器和预处理，记录参考/生成集合的来源与数量；样本不足时只作探索性观察，不把数值当作可靠的分布质量结论。

## 3. 执行配对比较并保留完整结果

为每个 input_id、seed 和候选保存实际配置、输出、成功/失败与分数。不要只保留最高分候选，也不要把失败或缺失条目默默移出结果。

成对评价先按稳定的 input_id/seed 建立显式配对清单，核对数量、重复和缺失。当前 LPIPS 目录接口只匹配共同文件 stem，同 stem 多文件只取排序首项，并直接聚合均值；不能据一次目录调用声称已评价全部文件或已保留逐样本结果。需要逐样本分析时逐对计算并保存，另报未配对项；集合不一致时说明有效配对子集，不能隐藏缺失造成的选择偏差。

LPIPS 的 `target_size=512` 不保证所有输入都按512评分：单图对尺寸相同时走原尺寸；目录模式仅在全部匹配图像的尺寸不统一时才整体启用 resize/center crop。需要 LPIPS@512 时在任务脚本显式统一预处理，并记录缩放、裁剪、骨干和实际评分尺寸，不仅设置 target_size。

生成时间与评分时间分开。性能边界同步，区分加载/量化/编译、首次、预热后稳态、编码/保存和端到端；资源分别命名 allocated、reserved、设备已用与 RAM。

对近似注意力、量化或少步采样，不套用模型移植的严格 allclose 当内容质量标准。若需要数值比较，先说明它回答的具体问题。

## 4. 分析差异、失败和局限

给逐样本/seed 的配对差异，再汇总均值/中位数和范围；排序任务报告胜/负/平局及无效评分率，不把候选编号当连续质量分求均值。缺失、解析失败、None 或非有限分数独立记录，不用0填充或静默纳入均值，报告有效分母及排除原因。

同一组输入上的候选优先使用配对统计；不将同一 prompt 的多个 seed、同一视频的多个片段视为完全独立任务。正式结论按独立 prompt、主体或来源分组估计不确定性，同时报告个例退化与生成/评分失败率；小样本不声称显著性，单 seed 仅作探索依据。多指标沿用事先确定的主指标，不因结果有利而临时更换主要结论依据。

附可审阅样例，解释一致改善、个例退化、失败率与任务相关维度。编辑保留源图，搜索保留候选，视频/音频检查实际媒体。

交付固定输入清单、配置、逐项结果、汇总、样例路径和复跑命令。缺模型/硬件时清楚标记未运行，不填虚构分数或性能数字。

## 当前图像指标接口

从 `diffsynth/metrics/__init__.py` 与 `examples/image_quality_metric/` 核对实际导出/调用；指标不是同一个通用接口。

```python
from diffsynth.metrics import PickScoreMetric, ModelConfig
metric = PickScoreMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="PickScore/model.safetensors"),
    device="cuda",
)
## image 必须为当前 compute 支持的对象
score = metric.compute(prompt, image)[0]
```

这会加载评分模型，不是零开销标量函数。`[0]` 对应已确认的单图返回；批量时保留每项分数与输入映射。先检查权重/依赖与 GPU/RAM，再决定是否常驻。

| 指标 | 当前接口/用途提示 |
| --- | --- |
| PickScore/部分 CLIP/HPS/ImageReward | 查各自 compute 的 prompt/images 类型和返回值，不套统一索引 |
| Aesthetic | compute(images) |
| FID | compute(reference_images, generated_images, batch_size=None, num_workers=None)，示例是两组目录；不适合单图评分 |
| LPIPS | compute(image_a,image_b)；目录模式按共同 stem 配对后返回均值，预处理规则见第3节；感知差异不等于通用美学质量 |
| QwenImageBench | evaluate(prompt,images,dimensions=None) 多维任务评价 |
| UnifiedRewardEdit | evaluate(instruction,[source,edited],task="edit_pointwise_score")[0]；pairwise 用 source/edit1/edit2 |

UnifiedRewardEdit 还需正确 processor/model配置。编辑评价同时保存源图与编辑结果。图像 reward 不能代替视频时序/运动或音频评价；没有合适现成指标时给人工维度/外部工具接入点并注明限制。

UnifiedRewardEdit 的 `evaluate()` 返回每项结构化结果的列表，`compute()`/`score()` 只取其中主分数。按实际 task 解释，不套统一标量：

| task | 单项输入 | 需要核对的输出 |
| --- | --- | --- |
| `edit_pointwise_score` | `[source, edited]` | `editing_success`、`overediting` 和汇总 `score`；当前汇总可能忽略缺失分量，只有 score 有值不表示评分完整 |
| `edit_pairwise_rank` | `[source, edited1, edited2]` | 结构化 `winner` 及主分数0/1/2；仅在 winner 确认有效时解释为平局/第一项胜/第二项胜 |
| `edit_pairwise_score` | `[source, edited1, edited2]` | `image_1_score`、`image_2_score`；主分数是两项列表，不是单一标量 |

优先保存 `evaluate()` 的结构化结果，核验 task 所需字段、分量完整性与数值有效性。当前 pairwise rank 的主分数在 `winner=None` 或未识别时也会回退为0，不能直接判为平局；记录解析失败并复核，不把无效返回纳入有效评价统计。
