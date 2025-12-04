# Enabling FP8 Precision in Training

Although `DiffSynth-Studio` supports [VRAM management](/docs/en/Pipeline_Usage/VRAM_management.md) in model inference, most of the techniques for reducing VRAM usage are not suitable for training. Offloading would cause extremely slow training processes.

FP8 precision is the only VRAM management strategy that can be enabled during training. However, this framework currently does not support native FP8 precision training. For reasons, see [Q&A: Why doesn't the training framework support native FP8 precision training?](/docs/en/QA.md#why-doesnt-the-training-framework-support-native-fp8-precision-training). It only supports storing models whose parameters are not updated by gradients (models that do not require gradient backpropagation, or whose gradients only update their LoRA) in FP8 precision.

## Enabling FP8

In our provided training scripts, you can quickly set models to be stored in FP8 precision through the `--fp8_models` parameter. Taking Qwen-Image LoRA training as an example, we provide a script for enabling FP8 training located at [`/examples/qwen_image/model_training/special/fp8_training/Qwen-Image-LoRA.sh`](/examples/qwen_image/model_training/special/fp8_training/Qwen-Image-LoRA.sh). After training is completed, you can verify the training results with the script [`/examples/qwen_image/model_training/special/fp8_training/validate.py`](/examples/qwen_image/model_training/special/fp8_training/validate.py).

Please note that this FP8 VRAM management strategy does not support gradient updates. When a model is set to be trainable, FP8 precision cannot be enabled for that model. Models that support FP8 include two types:

* Parameters are not trainable, such as VAE models
* Gradients do not update their parameters, such as DiT models in LoRA training

Experimental verification shows that LoRA training with FP8 enabled does not cause significant image quality degradation. However, theoretical errors do exist. If you encounter training results inferior to BF16 precision training when using this feature, please provide feedback through GitHub issues.

## Training Framework Design Concept

The training framework completely reuses the inference VRAM management, and only parses VRAM management configurations through `parse_model_configs` in `DiffusionTrainingModule` during training.