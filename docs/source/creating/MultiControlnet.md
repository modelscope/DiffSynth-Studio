# 猫猫、少女、FLUX、ControlNet——多 ControlNet 模型的灵活运用

文生图模型 FLUX 发布之后，开源社区为其适配了用于控制生成内容的模型——ControlNet，DiffSynth-Studio 为这些模型提供了支持，我们支持任意多个 ControlNet 模型的组合调用，即使这些模型的结构不同。本篇文章将展示这些 ControlNet 模型的灵活用法。

## Canny/Depth/Normal: 点对点结构控制

结构控制是 ControlNet 模型最基础的能力，通过使用 Canny 提取出边缘信息，或者使用深度图和法线贴图，都可以用于表示图像的结构，进而作为图像生成过程中的控制信息。

例如，我们生成一只猫猫，然后使用支持多控制条件的模型 InstantX/FLUX.1-dev-Controlnet-Union-alpha，同时启用 Canny 和 Depth 控制，让环境变为黄昏。

模型链接：https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np

download_models(["Annotators:Depth"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "InstantX/FLUX.1-dev-Controlnet-Union-alpha"])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="canny",
        model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        scale=0.3
    ),
    ControlNetConfigUnit(
        processor_id="depth",
        model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        scale=0.3
    ),
])
image_1 = pipe(
    prompt="a cat is running",
    height=1024, width=1024,
    seed=4
)
image_1.save("image_5.jpg")
image_2 = pipe(
    prompt="sunshine, a cat is running",
    controlnet_image=image_1,
    height=1024, width=1024,
    seed=5
)
image_2.save("image_6.jpg")
```

|![image_5](https://github.com/user-attachments/assets/19d2abc4-36ae-4163-a8da-df5732d1a737)|![image_6](https://github.com/user-attachments/assets/28378271-3782-484c-bd51-3d3311dd85c6)|
|-|-|

ControlNet 对于结构的控制力度是可以调节的，例如在下面这里例子中，我们把小姐姐从夏天移动到冬天时，适当调低 ControlNet 的控制力度，模型就会根据画面内容作出调整，为小姐姐换上温暖的衣服。

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np

download_models(["Annotators:Depth"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "InstantX/FLUX.1-dev-Controlnet-Union-alpha"])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="canny",
        model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        scale=0.3
    ),
    ControlNetConfigUnit(
        processor_id="depth",
        model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        scale=0.3
    ),
])
image_1 = pipe(
    prompt="a beautiful Asian girl, full body, red dress, summer",
    height=1024, width=1024,
    seed=6
)
image_1.save("image_7.jpg")
image_2 = pipe(
    prompt="a beautiful Asian girl, full body, red dress, winter",
    controlnet_image=image_1,
    height=1024, width=1024,
    seed=7
)
image_2.save("image_8.jpg")
```

|![image_7](https://github.com/user-attachments/assets/a7b8555b-bfd9-4e92-aa77-16bca81b07e3)|![image_8](https://github.com/user-attachments/assets/a1bab36b-6cce-4f29-8233-4cb824b524a8)|
|-|-|

## Upscaler/Tile/Blur: 高清图像生成

支持高清化的 ControlNet 模型有很多，例如

模型链接: https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler, https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha, https://modelscope.cn/models/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro

这些模型可以把模糊的、含噪点的低质量图像处理成清晰的图像。在 DiffSynth-Studio 中，框架原生支持的高分辨率分块处理技术可以突破模型的分辨率限制，实现 2048 甚至更高分辨率的图像生成，进一步放大了这些模型的能力。在下面的例子中，我们可以看到高清放大到 2048 分辨率的图片中，猫猫的毛发纤毫毕现，人物的皮肤纹理精致逼真。

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np

model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "jasperai/Flux.1-dev-Controlnet-Upscaler"])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="tile",
        model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
        scale=0.7
    ),
])
image_1 = pipe(
    prompt="a photo of a cat, highly detailed",
    height=768, width=768,
    seed=0
)
image_1.save("image_1.jpg")
image_2 = pipe(
    prompt="a photo of a cat, highly detailed",
    controlnet_image=image_1.resize((2048, 2048)),
    input_image=image_1.resize((2048, 2048)), denoising_strength=0.99,
    height=2048, width=2048, tiled=True,
    seed=1
)
image_2.save("image_2.jpg")
```

|![image_1](https://github.com/user-attachments/assets/9038158a-118c-4ad7-ab01-22865f6a06fc)|![image_2](https://github.com/user-attachments/assets/88583a33-cd74-4cb9-8fd4-c6e14c0ada0c)|
|-|-|

```python
model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "jasperai/Flux.1-dev-Controlnet-Upscaler"])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="tile",
        model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
        scale=0.7
    ),
])
image_1 = pipe(
    prompt="a beautiful Chinese girl, delicate skin texture",
    height=768, width=768,
    seed=2
)
image_1.save("image_3.jpg")
image_2 = pipe(
    prompt="a beautiful Chinese girl, delicate skin texture",
    controlnet_image=image_1.resize((2048, 2048)),
    input_image=image_1.resize((2048, 2048)), denoising_strength=0.99,
    height=2048, width=2048, tiled=True,
    seed=3
)
image_2.save("image_4.jpg")
```

|![image_3](https://github.com/user-attachments/assets/13061ecf-bb57-448a-82c6-7e4655c9cd85)|![image_4](https://github.com/user-attachments/assets/0b7ae80f-de58-4d1d-a49c-ad17e7631bdc)|
|-|-|

## Inpaint: 局部重绘

Inpaint 模型可以对图像中的特定区域进行重绘，比如，我们可以给猫猫戴上墨镜。

模型链接: https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np

model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="inpaint",
        model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
        scale=0.9
    ),
])
image_1 = pipe(
    prompt="a cat sitting on a chair",
    height=1024, width=1024,
    seed=8
)
image_1.save("image_9.jpg")
mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask[100:350, 350: -300] = 255
mask = Image.fromarray(mask)
mask.save("mask_9.jpg")
image_2 = pipe(
    prompt="a cat sitting on a chair, wearing sunglasses",
    controlnet_image=image_1, controlnet_inpaint_mask=mask,
    height=1024, width=1024,
    seed=9
)
image_2.save("image_10.jpg")
```

|![image_9](https://github.com/user-attachments/assets/babddad0-2d67-4624-b77a-c953250ebdab)|![mask_9](https://github.com/user-attachments/assets/d5bc2878-1817-457a-bdfa-200f955233d3)|![image_10](https://github.com/user-attachments/assets/e3197f2c-190b-4522-83ab-a2e0451b39f6)|
|-|-|-|

但是我们注意到，猫猫的头部动作发生了变化，如果我们想要保留原来的结构特征，可以使用 canny、depth、normal 模型，DiffSynth-Studio 为不同结构的 ControlNet 提供了无缝的兼容支持。配合一个 normal ControlNet，我们可以保证局部重绘时画面结构不变。

模型链接：https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Surface-Normals

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np

model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=[
    "FLUX.1-dev",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="inpaint",
        model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
        scale=0.9
    ),
    ControlNetConfigUnit(
        processor_id="normal",
        model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Surface-Normals/diffusion_pytorch_model.safetensors",
        scale=0.6
    ),
])
image_1 = pipe(
    prompt="a beautiful Asian woman looking at the sky, wearing a blue t-shirt.",
    height=1024, width=1024,
    seed=10
)
image_1.save("image_11.jpg")
mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask[-400:, 10:-40] = 255
mask = Image.fromarray(mask)
mask.save("mask_11.jpg")
image_2 = pipe(
    prompt="a beautiful Asian woman looking at the sky, wearing a yellow t-shirt.",
    controlnet_image=image_1, controlnet_inpaint_mask=mask,
    height=1024, width=1024,
    seed=11
)
image_2.save("image_12.jpg")
```

|![image_11](https://github.com/user-attachments/assets/c028e6fc-5125-4cba-b35a-b6211c2e6600)|![mask_11](https://github.com/user-attachments/assets/1928ee9a-7594-4c6e-9c71-5bd0b043d8f4)|![image_12](https://github.com/user-attachments/assets/97b3b9e1-f821-405e-971b-9e1c31a209aa)|
|-|-|-|

## MultiControlNet+MultiDiffusion: 精细的高阶控制

DiffSynth-Studio 不仅支持多个不同结构的 ControlNet 同时生效，还支持使用不同提示词分区控制图中内容，还支持超高分辨率大图的分块处理，这让我们能够作出极为精细的高阶控制。接下来，我们展示一张精美图片的创作过程。

首先使用提示词“a beautiful Asian woman and a cat on a bed. The woman wears a dress”生成一只猫猫和一位少女。

![image_13](https://github.com/user-attachments/assets/8da006e4-0e68-4fa5-b407-31ef5dbe8e5a)

然后，启用 Inpaint ControlNet 和 Canny ControlNet

模型链接: https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta, https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha

分两个区域进行控制：

|Prompt: an orange cat, highly detailed|Prompt: a girl wearing a red camisole|
|-|-|
|![mask_13_1](https://github.com/user-attachments/assets/188530a0-913c-48db-a7f1-62f0384bfdc3)|![mask_13_2](https://github.com/user-attachments/assets/99c4d0d5-8cc3-47a0-8e56-ceb37db4dfdc)|

生成的结果：

![image_14](https://github.com/user-attachments/assets/f5b9d3dd-a690-4597-91a8-a019c6fc2523)

背景有点模糊，我们使用去模糊 LoRA，进行图生图

模型链接：https://modelscope.cn/models/LiblibAI/FLUX.1-dev-LoRA-AntiBlur

![image_15](https://github.com/user-attachments/assets/32ed2667-2260-4d80-aaa9-4435d6920a2a)

整个画面清晰多了，接下来使用高清化模型，把分辨率增加到 4096*4096！

模型链接：https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler

![image_17](https://github.com/user-attachments/assets/1a688a12-1544-4973-8aca-aa3a23cb34c1)

放大来看看

![image_17_cropped](https://github.com/user-attachments/assets/461a1fbc-9ffa-4da5-80fd-e1af9667c804)

这一系列例子可以用以下代码“一条龙”式地生成：

```python
from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np


download_models(["Annotators:Depth", "Annotators:Normal"])
download_customized_models(
    model_id="LiblibAI/FLUX.1-dev-LoRA-AntiBlur",
    origin_file_path="FLUX-dev-lora-AntiBlur.safetensors",
    local_dir="models/lora"
)
model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=[
    "FLUX.1-dev",
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="inpaint",
        model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
        scale=0.9
    ),
    ControlNetConfigUnit(
        processor_id="canny",
        model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        scale=0.5
    ),
])

image_1 = pipe(
    prompt="a beautiful Asian woman and a cat on a bed. The woman wears a dress.",
    height=1024, width=1024,
    seed=100
)
image_1.save("image_13.jpg")

mask_global = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask_global = Image.fromarray(mask_global)
mask_global.save("mask_13_global.jpg")

mask_1 = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask_1[300:-100, 30: 450] = 255
mask_1 = Image.fromarray(mask_1)
mask_1.save("mask_13_1.jpg")

mask_2 = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask_2[500:-100, -400:] = 255
mask_2[-200:-100, -500:-400] = 255
mask_2 = Image.fromarray(mask_2)
mask_2.save("mask_13_2.jpg")

image_2 = pipe(
    prompt="a beautiful Asian woman and a cat on a bed. The woman wears a dress.",
    controlnet_image=image_1, controlnet_inpaint_mask=mask_global,
    local_prompts=["an orange cat, highly detailed", "a girl wearing a red camisole"], masks=[mask_1, mask_2], mask_scales=[10.0, 10.0],
    height=1024, width=1024,
    seed=101
)
image_2.save("image_14.jpg")

model_manager.load_lora("models/lora/FLUX-dev-lora-AntiBlur.safetensors", lora_alpha=2)
image_3 = pipe(
    prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. clear background.",
    negative_prompt="blur, blurry",
    input_image=image_2, denoising_strength=0.7,
    height=1024, width=1024,
    cfg_scale=2.0, num_inference_steps=50,
    seed=102
)
image_3.save("image_15.jpg")

pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="tile",
        model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
        scale=0.7
    ),
])
image_4 = pipe(
    prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. highly detailed, delicate skin texture, clear background.",
    controlnet_image=image_3.resize((2048, 2048)),
    input_image=image_3.resize((2048, 2048)), denoising_strength=0.99,
    height=2048, width=2048, tiled=True,
    seed=103
)
image_4.save("image_16.jpg")

image_5 = pipe(
    prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. highly detailed, delicate skin texture, clear background.",
    controlnet_image=image_4.resize((4096, 4096)),
    input_image=image_4.resize((4096, 4096)), denoising_strength=0.99,
    height=4096, width=4096, tiled=True,
    seed=104
)
image_5.save("image_17.jpg")
```

DiffSynth-Studio 和 ControlNet 的强大潜力已经展现在你的眼前了，快去体验 AIGC 技术的乐趣吧！

## 已支持的 FLUX ControlNet 列表

* https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Depth
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Surface-Normals
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler
* https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha
* https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta
* https://modelscope.cn/models/Shakker-Labs/FLUX.1-dev-ControlNet-Depth
* https://modelscope.cn/models/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
