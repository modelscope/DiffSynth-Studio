# Frequently Asked Questions

## Why doesn't the training framework support batch size > 1?

* **Larger batch sizes no longer achieve significant acceleration**: Due to acceleration technologies such as flash attention that have fully improved GPU utilization, larger batch sizes will only bring greater VRAM usage without significant acceleration. The experience with small models like Stable Diffusion 1.5 is no longer applicable to the latest large models.
* **Larger batch sizes can be achieved through other solutions**: Multi-GPU training and Gradient Accumulation can both mathematically equivalently achieve larger batch sizes.
* **Larger batch sizes contradict the framework's general design**: We hope to build a general training framework. Many models cannot accommodate larger batch sizes, such as text encodings of different lengths and images of different resolutions, which cannot be merged into larger batches.

## Why aren't redundant parameters removed from certain models?

In some models, redundant parameters exist. For example, in Qwen-Image's DiT model, the text portion of the last layer does not participate in any calculations. This is a minor bug left by the model developers. Setting it as trainable directly will also cause errors in multi-GPU training.

To maintain compatibility with other models in the open-source community, we have decided to retain these parameters. These redundant parameters can avoid errors in multi-GPU training through the `--find_unused_parameters` parameter.

## Why does FP8 quantization show no acceleration effect?

Native FP8 computation relies on Hopper architecture GPUs and has significant precision errors. It is currently immature technology, so this project does not support native FP8 computation.

FP8 computation in VRAM management refers to storing model parameters in memory or VRAM with FP8 precision and temporarily converting them to other precisions when needed for computation. Therefore, it can only reduce VRAM usage without acceleration effects.

## Why doesn't the training framework support native FP8 precision training?

Even with suitable hardware conditions, we currently have no plans to support native FP8 precision training.

* The main challenge of native FP8 precision training is precision overflow caused by gradient explosion. To ensure training stability, the model structure needs to be redesigned accordingly. However, no model developers are willing to do so at present.
* Additionally, models trained with native FP8 precision can only be computed with BF16 precision during inference without Hopper architecture GPUs, theoretically resulting in generation quality inferior to FP8.

Therefore, native FP8 precision training technology is extremely immature. We will observe the technological developments in the open-source community.

## How to dynamically load LoRA models during inference?

We support two loading methods for LoRA models. See [LoRA Loading](/docs/en/Pipeline_Usage/Model_Inference.md#loading-lora) for details:

* Cold Loading: When [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md) is not enabled for the base model, LoRA will be fused into the base model weights. In this case, inference speed remains unchanged, and LoRA cannot be unloaded after loading.
* Hot Loading: When [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md) is enabled for the base model, LoRA will not be fused into the base model weights. In this case, inference speed will slow down, and LoRA can be unloaded after loading via `pipe.clear_lora()`.
