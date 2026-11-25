# Inference Acceleration

The denoising process of diffusion models is typically time-consuming. To improve inference speed, various acceleration techniques can be applied, including lossless acceleration solutions such as multi-GPU parallel inference and computation graph compilation, as well as lossy acceleration solutions like Cache and quantization.

Currently, most diffusion models are built on [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748), and efficient attention mechanisms are also a common acceleration method. DiffSynth-Studio currently supports certain lossless acceleration inference features. This section focuses on introducing acceleration methods from two dimensions: multi-GPU parallel inference and computation graph compilation.

## Efficient Attention Mechanisms

For details on the acceleration of attention mechanisms, please refer to [Attention Mechanism Implementation](../API_Reference/core/attention.md).

## Multi-GPU Parallel Inference
See [Parallel Inference](Parallel_Inference.md) for usage.

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
  * For other parameter configurations, please consult the [PyTorch API documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

### Different Compile Options for Different Models

The `mode`, `dynamic`, and `fullgraph` arguments of a single `compile_pipeline` call apply to **every** model covered by that call. If different models need different options, make several calls and pass `compile_models` explicitly:

```python
pipe.compile_pipeline(compile_models=["dit"])
pipe.compile_pipeline(compile_models=["vae"], dynamic=False)
```

`compilable_models` declares the full set of models a pipeline supports compiling; a bare `pipe.compile_pipeline()` applies one set of options to that whole set.

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
