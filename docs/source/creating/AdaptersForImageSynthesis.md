# ControlNet、LoRA、IP-Adapter——精准控制技术

在文生图模型的基础上，还可以使用各种 Adapter 架构的模型对生成过程进行控制。

接下来的例子会用到很多模型，我们先把它们下载好。

* 一个广受好评的 Stable Diffusion XL 架构动漫风格模型
* 一个支持多种控制模式的 ControlNet 模型
* 一个 Stable Diffusion XL 模型的 LoRA 模型
* 一个 IP-Adapter 模型及其对应的图像编码器

```python
from diffsynth import download_models

download_models([
    "BluePencilXL_v200",
    "ControlNet_union_sdxl_promax",
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
    "IP-Adapter-SDXL"
])
```

用基础文生图功能生成一张图

```python
from diffsynth import ModelManager, SDXLImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion_xl/bluePencilXL_v200.safetensors"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)
torch.manual_seed(1)
image = pipe(
    prompt="masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    cfg_scale=6, num_inference_steps=60,
)
image.save("image.jpg")
```

![image](https://github.com/user-attachments/assets/cc094e8f-ff6a-4f9e-ba05-7a5c2e0e609f)

接下来，我们让这位水下翩翩起舞的少女变成火系魔法师！启用 ControlNet 保持画面结构的同时，修改提示词。

```python
from diffsynth import ModelManager, SDXLImagePipeline, ControlNetConfigUnit
import torch
from PIL import Image

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit("depth", "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors", scale=1)
])
torch.manual_seed(2)
image = pipe(
    prompt="masterpiece, best quality, solo, long hair, wavy hair, pink hair, red eyes, red dress, medium breasts, dress, fire ball, fire background, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, white background",
    cfg_scale=6, num_inference_steps=60,
    controlnet_image=Image.open("image.jpg")
)
image.save("image_controlnet.jpg")
```

![image_controlnet](https://github.com/user-attachments/assets/d50d173e-e81a-4d7e-93e3-b2787d69953e)

很酷对不对？还有更酷的，加个 LoRA，让画面更贴近手绘漫画的扁平风格。这个 LoRA 需要一定的触发词才能生效，这在原作者的模型页面有提到，记得在提示词的开头加上触发词哦。

```python
from diffsynth import ModelManager, SDXLImagePipeline, ControlNetConfigUnit
import torch
from PIL import Image

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors"
])
model_manager.load_lora("models/lora/zyd232_ChineseInkStyle_SDXL_v1_0.safetensors", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit("depth", "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors", scale=1.0)
])
torch.manual_seed(3)
image = pipe(
    prompt="zydink, ink sketch, flat anime, masterpiece, best quality, solo, long hair, wavy hair, pink hair, red eyes, red dress, medium breasts, dress, fire ball, fire background, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, white background",
    cfg_scale=6, num_inference_steps=60,
    controlnet_image=Image.open("image.jpg")
)
image.save("image_lora.jpg")
```

![image_lora](https://github.com/user-attachments/assets/c599b2f8-8351-4be5-a6ae-8380889cb9d8)

还没结束呢！找一张水墨风的中国画作为风格引导，启动 IP-Adapter，让古典艺术和现代美学碰撞！

|就用这张图作为风格引导吧|![ink_style](https://github.com/user-attachments/assets/e47c5a03-9c7b-402b-b260-d8bfd56abbc5)|
|-|-|

```python
from diffsynth import ModelManager, SDXLImagePipeline, ControlNetConfigUnit
import torch
from PIL import Image

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors",
    "models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin",
    "models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors",
])
model_manager.load_lora("models/lora/zyd232_ChineseInkStyle_SDXL_v1_0.safetensors", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit("depth", "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors", scale=1.0)
])
torch.manual_seed(2)
image = pipe(
    prompt="zydink, ink sketch, flat anime, masterpiece, best quality, solo, long hair, wavy hair, pink hair, red eyes, red dress, medium breasts, dress, fire ball, fire background, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, white background",
    cfg_scale=6, num_inference_steps=60,
    controlnet_image=Image.open("image.jpg"),
    ipadapter_images=[Image.open("ink_style.jpg")],
    ipadapter_use_instant_style=True, ipadapter_scale=0.5
)
image.save("image_ipadapter.jpg")
```

![image_ipadapter](https://github.com/user-attachments/assets/e5924aef-03b0-4462-811f-a60e2523fd7f)

用 Diffusion 生成图像的乐趣在于，各种生态模型的组合，可以实现各种奇思妙想。
