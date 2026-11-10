# 多模态与执行效率研究

适用于视频、音频、音视频联合及量化/缓存/执行策略实验。按当前任务只读相关部分，依目标Pipeline、loss与配方确认支持；推理能力不代表同条件已有训练入口。

## 视频与音视频联合

| 研究问题 | 当前入口举例 | 必要对照和观察 |
| --- | --- | --- |
| 运动/相机/人物条件 | Wan的camera、VACE、Animate、S2V、speed control | 条件遵循、主体保持、运动范围、时间一致性；先验证开关生效 |
| 长视频与局部重生成 | Wan滑窗/LongCat，LTX2和MiniMax-H3的retake | 窗长/重叠、边界、长期漂移、区域外保持及完整成本 |
| 关键帧与上下文 | LTX2 input_images/in_context_videos，MiniMax-H3 keyframes/references | 数量、位置、强度、参考顺序、稀疏条件与泛化 |
| 单/双阶段与动态计算 | LTX2 stage2/upscaler，LingBot refiner，模型专用切换 | 各阶段尺寸、调度、LoRA加载状态、预算与质量提升来源 |
| 联合学习与同步 | MiniMax-H3、LTX2、MOVA专用Pipeline/loss | 单模态/联合目标、模态权重、不同条件与时间表；音画同步和模态各自质量 |

源码分别为 `diffsynth/pipelines/{wan_video,ltx2_audio_video,minimax_h3_audio_video,mova_audio_video,lingbot_video}.py`，配合 `docs/zh/Model_Details/{Wan,LTX-2,MiniMax-H3,LingBot-Video}.md` 及目标示例。不要把另一个模型的音频/参考字段直接搬来。

固定并记录真实尺寸、帧数/fps、秒数、采样率/声道、时空压缩、条件索引；按实际输出核验，不只记录请求值。首帧/末帧/参考条件与监督方向要明确。Dataset采样fps与Pipeline输入fps可能来自不同字段，需追踪CLI、metadata及消费者。改变时间压缩/分辨率也可能改变调度或噪声shape。

当前MiniMax-H3训练暴露 `--audio_loss_weight` 和 `--training_cfg_scale`，对应 `FlowMatchSFTMiniMaxH3AudioVideoLoss`。audio_loss_weight=0只移除audio loss贡献，仍执行音频加噪及前向；不能当成删除音频计算或证明模型无音频分支。training_cfg_scale>1增加no_grad无条件分支并还原训练预测，缓存需包含匹配输入；不能只更改训练flag而复用不兼容缓存。

分开控制loss消融、输入条件消融和结构/计算消融。比较joint training时分别测两种模态及同步，避免视频改善但音频退化被平均分掩盖。retake保留源媒体、改动区域和区间外结果；局部效果好不代表边界平滑或全程稳定。

## 音频与音乐

| 入口 | 可研究机制 | 验证维度与边界 |
| --- | --- | --- |
| `ace_step.py` / ACE-Step示例 | prompt/lyrics、BPM/调式/拍号、音色参考、cover/repaint强度与区间 | 音质、歌词/节拍遵循、音色保持、片段边界；精确变体的条件支持分别查 |
| `diffsynth_music.py` / DiffSynth-Music示例 | 节拍、人声/伴奏、韵律、参考、target_track | 结构化条件契约、轨道关系、节奏和长程一致性 |
| `yue2.py` / `YuE2-Edit.py` | `plan()`输出ABC乐谱、cot模式、乐谱编辑后渲染 | 符号结构遵循与最终音频，计入自回归和声学阶段成本；当前文档未提供训练支持 |
| `minimax_music3.py` | 条件编码、语义与分块声学生成 | 长度/段落、chunk边界与完整生成时间；训练支持另核对 |

从 `diffsynth/utils/data/audio.py`、目标operator和编码器读取waveform/latent契约。训练数据latent与原始音频不是可互换类型；采样率、声道和时长要实际验收。评价同时检查可解码性、音质、条件遵循、长程结构和任务特有维度；图像reward和静态媒体元信息不能替代听感或音画语义同步。

没有集成的模态指标时，可以设计人工评价或按用户范围接入外部工具，但需说明未验证范围和依赖，不宣称框架已有一键指标。先利用现成可编辑条件完成免训练研究，再按需要实现新的目标或架构。

## 数值精度与压缩

研究入口：`diffsynth/core/quant/{config,base}.py`、目标backend、`utils/quant/serialization.py` 及 `docs/zh/Pipeline_Usage/Quantization.md`。可研究bit/码率、目标层、敏感层排除、混合精度、权重误差与生成误差、量化后LoRA及组合精度。

`QuantizeConfig`支持target_modules/exclude_modules；`MixedQuantizeConfig`要求各配置命中层不重叠。先核对实际命中、共享权重、注册模型的生效配置，不能只看传入参数。`measure_quantization_error`测的是权重误差，不能替代输出质量；`dequant_once`可用于分析量化误差与动态执行开销，但不等于保持低比特常驻存储。

对比原精度、单方法、混合/保护层时固定同模型权重和真实输入/噪声，分别观察层/噪声阶段敏感性、质量与性能。冻结量化层参与Adapter训练时验证对输入可微；`check_differentiable`可做局部检查，仍需目标训练链验证。不能把低比特底模+LoRA称为全量低比特权重训练。

保存量化权重需backend序列化与加载支持；新进程验证量化状态、辅助metadata、LoRA匹配和实际输出。当前量化+disk加载需预量化兼容路径；不能将在线量化直接与disk任意拼接。框架FP8存储/暂存与某个backend原生低精度计算分开解释，不承诺“量化一定提速”。

## 缓存、attention与资源策略

| 探索方向 | 实际入口 | 有效性与成本证据 |
| --- | --- | --- |
| KV/计算复用 | 模型专用KV、TeaCache、预计算Unit | 是否真实复用、cache失效、近似误差、不同条件下的污染 |
| attention | `core/attention/`注册/路由、模型局部attention | 实际backend/fallback、mask/causal/window语义、前反向和质量 |
| compile | `BasePipeline.compile_pipeline`、目标compilable_models | 编译范围、图断裂/重编译、首次与稳态、shape与输入分布 |
| offload与分布式 | VRAM、offload_training、序列并行、训练配置 | 峰值显存/RAM、传输/I/O/通信、吞吐及数值一致性 |
| 拆分预处理 | `PipelineUnitGraph`与训练:data_process/train | 缓存构建成本/磁盘、随机增强语义及梯度边界 |

attention环境变量在模块导入时选择全局实现；运行中只改环境变量可能不生效，优先独立进程配置。即使选择某backend，mask/window/causal或use_flex/compatibility分支仍可能转向其他路径；测量实际执行，不能把fallback结果当成目标kernel性能。模型内部也未必全部经过统一attention_forward。

序列并行需区分推理与训练：`WanVideoPipeline` 和 `MovaAudioVideoPipeline` 的 `from_pretrained()` 有 `use_usp` 入口，但当前 `examples/{wanvideo,mova}/model_training/train.py` 均未传递该参数，不能从推理示例推断标准训练入口已支持 USP（Unified Sequence Parallel）。研究训练序列并行时，逐项验证训练入口、分布式初始化、实际 Attention/通信路径、前反向梯度、参数同步和 checkpoint 保存/重载；DDP 不会自动提供序列并行，项目新增实现需单独验证。

推理资源优化与训练梯度/optimizer策略分别核查；TemplatePipeline无自身VRAM管理、DMD2不支持model CPU offload等专用约束优先。不要为性能比较静默改变用户指定规格/精度；把有损与无损、算法改变与执行优化分开报告。

固定设备/软件/模型身份、形状和工作负载，在同步边界计时并记录重复次数。分开加载/转换/量化、编译、预处理、首次、预热稳态、传输/保存、评分及端到端；显存allocated/reserved/设备已用与host RAM分开命名，offload还看磁盘和I/O。缓存/编译不能只报摊销后的最快一轮而隐藏准备成本。兼顾质量、失败率、多样性及资源，报告适用工作负载或Pareto取舍；未运行不填写加速倍数。
