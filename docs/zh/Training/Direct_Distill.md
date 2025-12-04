# 端到端的蒸馏加速训练

## 蒸馏加速训练

Diffusion 模型的推理过程通常需要多步迭代，在提升生成效果的同时也让生成过程变得缓慢。通过蒸馏加速训练，可以减少生成清晰内容所需的步数。蒸馏加速训练技术的本质训练目标是让少量步数的生成效果与大量步数的生成效果对齐。

蒸馏加速训练的方法是多样的，例如

* 对抗式训练 ADD（Adversarial Diffusion Distillation）
    * 论文：https://arxiv.org/abs/2311.17042
    * 模型：[stabilityai/sdxl-turbo](https://modelscope.cn/models/stabilityai/sdxl-turbo)
* 渐进式训练 Hyper-SD
    * 论文：https://arxiv.org/abs/2404.13686
    * 模型：[ByteDance/Hyper-SD](https://www.modelscope.cn/models/ByteDance/Hyper-SD)

## 直接蒸馏

在训练框架层面，支持这类蒸馏加速训练方案是极其困难的。在训练框架的设计中，我们需要保证训练方案满足以下条件：

* 通用性：训练方案适用于大多数框架内支持的 Diffusion 模型，而非只能对某个特定模型生效，这是代码框架建设的基本要求。
* 稳定性：训练方案需保证训练效果稳定，不需要人工进行精细的参数调整，ADD 中的对抗式训练则无法保证稳定性。
* 简洁性：训练方案不会引入额外的复杂模块，根据奥卡姆剃刀（[Occam's Razor](https://en.wikipedia.org/wiki/Occam%27s_razor)）原理，复杂解决方案可能引入潜在风险，Hyper-SD 中的 Human Feedback Learning 让训练过程变得过于复杂。

因此，在 `DiffSynth-Studio` 的训练框架中，我们设计了一个端到端的蒸馏加速训练方案，我们称为直接蒸馏（Direct Distill），其训练过程的伪代码如下：

```
seed = xxx
with torch.no_grad():
    image_1 = pipe(prompt, steps=50, seed=seed, cfg=4)
image_2 = pipe(prompt, steps=4, seed=seed, cfg=1)
loss = torch.nn.functional.mse_loss(image_1, image_2)
```

是的，非常端到端的训练方案，稍加训练就可以有立竿见影的效果。

## 直接蒸馏训练的模型

我们用这个方案基于 Qwen-Image 训练了两个模型：

* [DiffSynth-Studio/Qwen-Image-Distill-Full](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full): 全量蒸馏训练
* [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA): LoRA 蒸馏训练

点击模型链接即可前往模型页面查看模型效果。

## 在训练框架中使用蒸馏加速训练

首先，需要生成训练数据，请参考[模型推理](/docs/zh/Pipeline_Usage/Model_Inference.md)部分编写推理代码，以足够多的推理步数生成训练数据。

以 Qwen-Image 为例，以下代码可以生成一张图片：

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

然后，我们把必要的信息编写成[元数据文件](/docs/zh/API_Reference/core/data.md#元数据)：

```csv
image,prompt,seed,rand_device,num_inference_steps,cfg_scale
distill_qwen/image.jpg,"精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",0,cpu,4,1
```

这个样例数据集可以直接下载：

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

然后开始 LoRA 蒸馏加速训练：

```shell
bash examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh
```

请注意，在[训练脚本参数](/docs/zh/Pipeline_Usage/Model_Training.md#脚本参数)中，数据集的图像分辨率设置要避免触发缩放处理。当设定 `--height` 和 `--width` 以启用固定分辨率时，所有训练数据必须是以完全一致的宽高生成的；当设定 `--max_pixels` 以启用动态分辨率时，`--max_pixels` 的数值必须大于或等于任一训练图像的像素面积。

## 训练框架设计思路

直接蒸馏与[标准监督训练](/docs/zh/Training/Supervised_Fine_Tuning.md)相比，仅训练的损失函数不同，直接蒸馏的损失函数是 `diffsynth.diffusion.loss` 中的 `DirectDistillLoss`。

## 未来工作

直接蒸馏是通用性很强的加速方案，但未必是效果最好的方案，所以我们暂未把这一技术以论文的形式发布。我们希望把这个问题交给学术界和开源社区共同解决，期待开发者能够给出更完善的通用训练方案。
