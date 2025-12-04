# 在训练中启用 FP8 精度

尽管 `DiffSynth-Studio` 在模型推理中支持[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)，但其中的大部分减少显存占用的技术不适合用于训练中，Offload 会导致极为缓慢的训练过程。

FP8 精度是唯一可在训练过程中启用的显存管理策略，但本框架目前不支持原生 FP8 精度训练，原因详见 [Q&A: 为什么训练框架不支持原生 FP8 精度训练？](/docs/zh/QA.md#为什么训练框架不支持原生-fp8-精度训练)，仅支持将参数不被梯度更新的模型（不需要梯度回传，或梯度仅更新其 LoRA）以 FP8 精度进行存储。

## 启用 FP8

在我们提供的训练脚本中，通过参数 `--fp8_models` 即可快速设置以 FP8 精度存储的模型。以 Qwen-Image 的 LoRA 训练为例，我们提供了启用 FP8 训练的脚本，位于 [`/examples/qwen_image/model_training/special/fp8_training/Qwen-Image-LoRA.sh`](/examples/qwen_image/model_training/special/fp8_training/Qwen-Image-LoRA.sh)。训练完成后，可通过脚本 [`/examples/qwen_image/model_training/special/fp8_training/validate.py`](/examples/qwen_image/model_training/special/fp8_training/validate.py) 验证训练效果。

请注意，这种 FP8 显存管理策略不支持梯度更新，当某个模型被设置为可训练时，不能为这个模型开启 FP8 精度，支持开启 FP8 的模型包括两类：

* 参数不可训练，例如 VAE 模型
* 梯度不更新其参数，例如 LoRA 训练中的 DiT 模型

经实验验证，开启 FP8 后的 LoRA 训练效果没有明显的图像质量下降，但理论上误差是确实存在的，如果在使用本功能时遇到训练效果不如 BF16 精度训练的问题，请通过 GitHub issue 给我们提供反馈。

## 训练框架设计思路

训练框架完全沿用推理的显存管理，在训练中仅通过 `DiffusionTrainingModule` 中的 `parse_model_configs` 解析显存管理配置。
