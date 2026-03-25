# 推理加速

扩散模型的去噪过程通常耗时较长。为提升推理速度，可采用多种加速技术，包含多卡并行推理、计算图编译等无损加速方案，以及 Cache、量化等有损加速方案。

当前扩散模型大多基于 Diffusion Transformer 构建，高效注意力机制同样是常用的加速手段。DiffSynth-Studio 目前已支持部分无损加速推理功能。本节重点从多卡并行推理和计算图编译两个维度介绍加速方法。

## 高效注意力机制
注意力机制的加速细节请参考 [注意力机制实现](../API_Reference/core/attention.md)。

## 多卡并行推理
DiffSynth-Studio 采用统一序列并行的多卡推理方案。在 DiT 中将 token 序列拆分至多张显卡进行并行处理。底层基于 [xDiT](https://github.com/xdit-project/xDiT) 实现。需要注意，统一序列并行会引入额外通信开销，实际加速比通常低于显卡数量。

目前 DiffSynth-Studio 已支持 [Wan](../Model_Details/Wan.md) 和 [MOVA](../Model_Details/Wan.md) 模型的统一序列并行加速。

首先安装 `xDiT` 依赖。
```bash
pip install "xfuser[flash-attn]>=0.4.3"
```

然后使用 `torchrun` 启动多卡推理。
```bash
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

构建 pipeline 时配置 `usp=True` 即可实现 USP 并行推理。代码示例如下。
```python
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import torch.distributed as dist

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_usp=True,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Text-to-video
video = pipe(
    prompt="一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
if dist.get_rank() == 0:
    save_video(video, "video1.mp4", fps=15, quality=5)
```

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
- 其他参数配置请查阅 [api 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)。

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