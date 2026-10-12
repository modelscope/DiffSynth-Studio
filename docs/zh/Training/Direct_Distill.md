# 端到端直接蒸馏加速（已停止维护）

（现阶段，我们强烈建议开发者使用 OPD 进行蒸馏加速训练，本文档中的内容仅用于科研用途。）

## 算法原理

在 `DiffSynth-Studio` 的训练框架中，我们设计了一个端到端的蒸馏加速训练方案，我们称为直接蒸馏（Direct Distill），其训练过程的伪代码如下：

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

首先，需要生成训练数据，请参考[模型推理](../Pipeline_Usage/Model_Inference.md)部分编写推理代码，以足够多的推理步数生成训练数据。

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

然后，我们把必要的信息编写成[元数据文件](../API_Reference/core/data.md#元数据)：

```csv
image,prompt,seed,rand_device,num_inference_steps,cfg_scale
distill_qwen/image.jpg,"精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",0,cpu,4,1
```

这个样例数据集可以直接下载：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

然后开始 LoRA 蒸馏加速训练：

```shell
bash examples/qwen_image/model_training/lora/Qwen-Image-Distill-LoRA.sh
```

请注意，在[训练脚本参数](../Pipeline_Usage/Model_Training.md#脚本参数)中，数据集的图像分辨率设置要避免触发缩放处理。当设定 `--height` 和 `--width` 以启用固定分辨率时，所有训练数据必须是以完全一致的宽高生成的；当设定 `--max_pixels` 以启用动态分辨率时，`--max_pixels` 的数值必须大于或等于任一训练图像的像素面积。
