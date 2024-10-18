# ControlNet、LoRA、IP-Adapter——Precision Control Technology

Based on the VinVL model, various adapter-based models can be used to control the generation process.

Let's download the models we'll be using in the upcoming examples:

* A highly praised Stable Diffusion XL architecture anime-style model
* A ControlNet model that supports multiple control modes
* A LoRA model for the Stable Diffusion XL model
* An IP-Adapter model and its corresponding image encoder

Please note that the names of the models are kept in English as per your instruction to retain specific terminology.

```python
from diffsynth import download_models

download_models([
    "BluePencilXL_v200",
    "ControlNet_union_sdxl_promax",
    "SDXL_lora_zyd23ble_diffusion_xl/bluePencilXL_v200.safetensors"])
pipe = SDXLImagePipeline.from_model_ma2_ChineseInkStyle_SDXL_v1_0",
    "IP-Adapter-SDXL"
])
```

Using basic text-to-image functionality to generate a picture.

```python
from diffsynth import ModelManager, SDXLImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stanager(model_manager)
torch.manual_seed(1)
image = pipe(
    prompt="masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,",
    cfg_scale=6, num_inference_steps=60,
)
image.save("image.jpg")
```

![image](https://github.com/user-attachments/assets/cc094e8f-ff6a-4f9e-ba05-7a5c2e0e609f)

Next, let's transform this graceful underwater dancer into a fire mage! We'll activate the ControlNet to maintain the structure of the image while modifying the prompt.

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

Isn't that cool? There's more! Add a LoRA to make the image closer to the flat style of hand-drawn comics. This LoRA requires certain trigger words to take effect, which is mentioned on the original author's model page. Remember to add the trigger words at the beginning of the prompt.

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

Not done yet! Find a Chinese painting with ink-wash style as a style guide, activate the IP-Adapter, and let classical art collide with modern aesthetics!

| Let's use this image as a style guide. |![ink_style](https://github.com/user-attachments/assets/e47c5a03-9c7b-402b-b260-d8bfd56abbc5)|
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

The joy of generating images with Diffusion lies in the combination of various ecosystem models, which can realize all kinds of creative ideas.
