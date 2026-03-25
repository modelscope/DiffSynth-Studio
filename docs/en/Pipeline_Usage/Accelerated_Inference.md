# Inference Acceleration

The denoising process of diffusion models is typically time-consuming. To improve inference speed, various acceleration techniques can be applied, including lossless acceleration solutions such as multi-GPU parallel inference and computation graph compilation, as well as lossy acceleration solutions like Cache and quantization.

Currently, most diffusion models are built on Diffusion Transformers (DiT), and efficient attention mechanisms are also a common acceleration method. DiffSynth-Studio currently supports certain lossless acceleration inference features. This section focuses on introducing acceleration methods from two dimensions: multi-GPU parallel inference and computation graph compilation.

## Efficient Attention Mechanisms

For details on the acceleration of attention mechanisms, please refer to [Attention Mechanism Implementation](https://www.google.com/search?q=../API_Reference/core/attention.md).

## Multi-GPU Parallel Inference

DiffSynth-Studio adopts a multi-GPU inference solution using Unified Sequence Parallel (USP). It splits the token sequence in the DiT across multiple GPUs for parallel processing. The underlying implementation is based on [xDiT](https://github.com/xdit-project/xDiT). Please note that unified sequence parallelism introduces additional communication overhead, so the actual speedup ratio is usually lower than the number of GPUs.

Currently, DiffSynth-Studio supports unified sequence parallel acceleration for the [Wan](https://www.google.com/search?q=../Model_Details/Wan.md) and [MOVA](https://www.google.com/search?q=../Model_Details/Wan.md) models.

First, install the `xDiT` dependency.

```bash
pip install "xfuser[flash-attn]>=0.4.3"
```

Then, use `torchrun` to launch multi-GPU inference.

```bash
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

When building the pipeline, simply configure `use_usp=True` to enable USP parallel inference. A code example is shown below.

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

## Computation Graph Compilation

PyTorch 2.0 provides an automatic computation graph compilation interface, [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), which can just-in-time (JIT) compile PyTorch code into optimized kernels, thereby improving execution speed. Since the inference time of diffusion models is concentrated in the multi-step denoising phase of the DiT, and the DiT is primarily stacked with basic blocks, DiffSynth's compile feature uses a [regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) strategy targeting only the basic Transformer blocks to reduce compilation time.

### Compile Usage Example

Compared to standard inference, you simply need to execute `pipe.compile_pipeline()` before calling the pipeline to enable compilation acceleration. For the specific function definition, please refer to the [source code](https://github.com/modelscope/DiffSynth-Studio/blob/166e6d2d38764209f66a74dd0fe468226536ad0f/diffsynth/diffusion/base_pipeline.py#L342).

The input parameters for `compile_pipeline` consist mainly of two types.

The first type is the compiled model parameters, `compile_models`. Taking the Qwen-Image Pipeline as an example, if you only want to compile the DiT model, you can keep this parameter empty. If you need to additionally compile models like the VAE, you can pass `compile_models=["vae", "dit"]`. Aside from DiT, all other models use a full-graph compilation strategy, meaning the model's forward function is completely compiled into a computation graph.

The second type is the compilation strategy parameters. This covers `mode`, `dynamic`, `fullgraph`, and other custom options. These parameters are directly passed to the `torch.compile` interface. If you are not deeply familiar with the specific mechanics of these parameters, it is recommended to keep the default settings.

  * `mode` specifies the compilation mode, including `"default"`, `"reduce-overhead"`, `"max-autotune"`, and `"max-autotune-no-cudagraphs"`. Because cudagraph has stricter requirements on computation graphs (for example, it might need to be used in conjunction with `torch.compiler.cudagraph_mark_step_begin()`), the `"reduce-overhead"` and `"max-autotune"` modes might fail to compile.
  * `dynamic` determines whether to enable dynamic shapes. For most generative models, modifying the prompt, enabling CFG, or adjusting the resolution will change the shape of the input tensors to the computation graph. Setting `dynamic=True` will increase the compilation time of the first run, but it supports dynamic shapes, meaning no recompilation is needed when shapes change. When set to `dynamic=False`, the first compilation is faster, but any operation that alters the input shape will trigger a recompilation. For most scenarios, setting it to `dynamic=True` is recommended.
  * `fullgraph`, when set to `True`, makes the underlying system attempt to compile the target model into a single computation graph, throwing an error if it fails. When set to `False`, the underlying system will set breakpoints where connections cannot be made, compiling the model into multiple independent computation graphs. Developers can set it to `True` to optimize compilation performance, but regular users are advised to only use `False`.
  * For other parameter configurations, please consult the [API documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

### Compile Feature Developer Documentation

If you need to provide compile support for a newly integrated pipeline, you should configure the `compilable_models` attribute in the pipeline to specify the default models to compile. For the DiT model class of that pipeline, you also need to configure `_repeated_blocks` to specify the types of basic blocks that will participate in regional compilation.

Taking Qwen-Image as an example, its pipeline configuration is as follows:

```python
self.compilable_models = ["dit"]
```

Its DiT configuration is as follows:

```python
class QwenImageDiT(torch.nn.Module):
    _repeated_blocks = ["QwenImageTransformerBlock"]
```
