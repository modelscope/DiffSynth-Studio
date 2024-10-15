# 文生图、图生图、高分辨率修复——初识绚丽的 Diffusion

加载文生图模型，这里我们使用一个 Civiai 上一个动漫风格的模型作为例子。

```python
import torch
from diffsynth import ModelManager, SDImagePipeline, download_models

download_models(["AingDiffusion_v12"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion/aingdiffusion_v12.safetensors"])
pipe = SDImagePipeline.from_model_manager(model_manager)
```

生成一张图小试身手。

```python
torch.manual_seed(0)
image = pipe(
    prompt="masterpiece, best quality, a girl with long silver hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=512, width=512, num_inference_steps=80,
)
image.save("image.jpg")
```

嗯，一个可爱的小姐姐。

![image](https://github.com/user-attachments/assets/999100d2-1c39-4f18-b37e-aa9d5b4e519c)

用图生图功能把她的头发变成红色，只需要添加 `input_image` 和 `denoising_strength` 两个参数。其中 `denoising_strength` 用于控制加噪声的强度，为 0 时生成的图与输入的图完全一致，为 1 时完全随机生成图。

```python
torch.manual_seed(1)
image_edited = pipe(
    prompt="masterpiece, best quality, a girl with long red hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=512, width=512, num_inference_steps=80,
    input_image=image, denoising_strength=0.6,
)
image_edited.save("image_edited.jpg")
```

嗯，一个红色头发的可爱小姐姐。

![image_edited](https://github.com/user-attachments/assets/e3de8bc1-037f-4d4d-aacf-8919143c2375)

由于模型本身是在 512*512 分辨率下训练的，所以图片看起来有点模糊，不过我们可以利用模型自身的能力润色这张图，为其填充细节。具体来说，就是提高分辨率后进行图生图。

```python
torch.manual_seed(2)
image_highres = pipe(
    prompt="masterpiece, best quality, a girl with long red hair",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    height=1024, width=1024, num_inference_steps=80,
    input_image=image_edited.resize((1024, 1024)), denoising_strength=0.6,
)
image_highres.save("image_highres.jpg")
```

嗯，一个清晰的红色头发可爱小姐姐。

![image_highres](https://github.com/user-attachments/assets/4466353e-662c-49f5-9211-b11bb0bb7fb7)

值得注意的是，图生图和高分辨率修复功能是全局支持的，目前我们所有的图像生成流水线都可以这样使用。
