# 推理加速

扩散模型的去噪过程通常耗时较长。为提升推理速度，可采用多种加速技术，包含多卡并行推理、计算图编译等无损加速方案，以及 Cache、量化等有损加速方案。

当前扩散模型大多基于 [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748) 构建，高效注意力机制同样是常用的加速手段。DiffSynth-Studio 目前已支持部分无损加速推理功能。本节重点从多卡并行推理和计算图编译两个维度介绍加速方法。

## 高效注意力机制
注意力机制的加速细节请参考 [注意力机制实现](../API_Reference/core/attention.md)。

## 多卡并行推理
使用方法请参考[并行推理](Parallel_Inference.md)。

## 计算图编译
PyTorch 2.0 提供了自动计算图编译接口 [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)，能够将 PyTorch 代码即时编译为优化内核，从而提升运行速度。由于扩散模型的推理耗时集中在 DiT 的多步去噪阶段，且 DiT 主要由基础模块堆叠而成，为缩短编译时间，DiffSynth 的 compile 功能采用仅针对基础 Transformer 模块的 [区域编译](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) 策略。

### Compile 使用示例
相比常规推理，只需在调用 pipeline 前执行 `pipe.compile_pipeline()` 即可开启编译加速。具体函数定义请参阅[源代码](https://github.com/modelscope/DiffSynth-Studio/blob/166e6d2d38764209f66a74dd0fe468226536ad0f/diffsynth/diffusion/base_pipeline.py#L342)。

`compile_pipeline` 的输入参数主要包含两类。

第一类是编译模型参数 `compile_models`。以 Qwen-Image Pipeline 为例，若仅编译 DiT 模型，保持该参数为空即可。若需额外编译 VAE 等模型，可传入 `compile_models=["vae", "dit"]`。除 DiT 外，其余模型均采用整体编译策略，即把模型的 forward 函数完整编译为计算图。

第二类是编译策略参数。涵盖 `mode`, `dynamic`, `fullgraph` 及其他自定义选项。这些参数会直接传递给 `torch.compile` 接口。若未深入了解这些参数的具体机制，建议保持默认设置。

- `mode` 指定编译模式，包含 `"default"`, `"reduce-overhead"`, `"max-autotune"` 和 `"max-autotune-no-cudagraphs"`。由于 cudagraph 对计算图要求较为严格（例如可能需要配合 `torch.compiler.cudagraph_mark_step_begin()` 使用），`"reduce-overhead"` 和 `"max-autotune"` 模式可能编译失败。
- `dynamic` 决定是否启用动态形状。对于多数生成模型，修改 prompt、开启 CFG 或调整分辨率都会改变计算图的输入张量形状。设置为 `dynamic=True` 会增加首次运行的编译时长，但支持动态形状，形状改变时无需重编译。设置为 `dynamic=False` 时首次编译较快，但任何改变输入形状的操作都会触发重新编译。对大部分场景，建议设定为 `dynamic=True`。
- `fullgraph` 设为 `True` 时，底层会尝试将目标模型编译为单一计算图，若失败则报错。设为 `False` 时，底层会在无法连接处设置断点，将模型编译为多个独立计算图。开发者可开启 `True` 来优化编译性能，普通用户建议仅使用 `False`。
- 其他参数配置请查阅 [PyTorch 的 API 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

### 为不同模型指定不同编译选项
一次 `compile_pipeline` 调用的 `mode`, `dynamic`, `fullgraph` 会应用到该次调用涉及的**所有**模型。若不同模型需要不同选项，分多次调用并显式指定 `compile_models`：

```python
pipe.compile_pipeline(compile_models=["dit"])
pipe.compile_pipeline(compile_models=["vae"], dynamic=False)
```

`compilable_models` 声明的是该 pipeline 支持编译的模型全集，裸调用 `pipe.compile_pipeline()` 会把同一套选项应用到全集。

### Compile 功能开发者文档
若需为新接入的 pipeline 提供 compile 支持，应在 pipeline 中配置 `compilable_models` 属性以指定默认编译模型。针对该 pipeline 的 DiT 模型类，还需配置 `_repeated_blocks` 以指定参与区域编译的基础模块类型。

以 Qwen-Image 为例，其 pipeline 配置如下。
```python
self.compilable_models = ["dit"]
```

其 DiT 配置如下。
```python
class QwenImageDiT(torch.nn.Module):
    _repeated_blocks = ["QwenImageTransformerBlock"]
```
