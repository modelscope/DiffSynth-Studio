# 调度器

调度器（Scheduler）控制模型的整个去噪（或采样）过程。在加载 Pipeline 时，DiffSynth 会自动选择最适合当前 Pipeline 的调度器，``无需额外配置``。

我们支持的调度器包括：

- **EnhancedDDIMScheduler**：扩展了去噪扩散概率模型（DDPM）中的去噪过程，引入了非马尔可夫指导。

- **FlowMatchScheduler**：实现了 [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) 中提出的流量匹配采样方法。

- **ContinuousODEScheduler**：基于常微分方程（ODE）的调度器。
