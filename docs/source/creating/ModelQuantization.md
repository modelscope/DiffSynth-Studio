# 量化、卸载——显存优化的技术

Flux.1 的发布让文生图开源社区再次活跃起来，但是其12B的参数量限制了显存低于24GB设备的运行。Diffsynth对Flux支持了量化（quantization）和卸载（offload）这两种优化显存的技术，降低了使用Flux的硬件设备门槛，本篇文章将介绍它们的原理和使用方式。


## 量化

模型量化指的是一种将高精度数据类型映射成低精度数据类型的技术，从而以损失少量精度为代价降低计算的时间和空间消耗。Flux.1 默认使用的数据类型为bfloat16，即每个参数占用16 bit（2 byte），我们使用torch支持的float8_e4m3fn加载模型，就能以更低的显存(节约10GB左右显存)消耗生成和原先几乎相同质量的图片。

```python
import torch
from diffsynth import download_models, ModelManager, FluxImagePipeline

download_models(["FLUX.1-dev"])

model_manager = ModelManager(
    torch_dtype=torch.bfloat16,
)
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
])
model_manager.load_models(
    ["models/FLUX/FLUX.1-dev/flux1-dev.safetensors"],
    torch_dtype=torch.float8_e4m3fn # Load the DiT model in FP8 format.
)

pipe = FluxImagePipeline.from_model_manager(model_manager, device="cuda")
pipe.dit.quantize() 

prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(9)
image = pipe(
    prompt=prompt,
    num_inference_steps=50, embedded_guidance=3.5
)
image.save("image_1024.jpg")
```


<div align="center">
    <figure style="display: inline-block; margin-right: 20px;">
        <img src="https://github.com/user-attachments/assets/d4c1699c-447b-4a5b-b453-4aa4d5ac066f" alt="图片1" width="300">
        <figcaption>float8_e4m3fn</figcaption>
    </figure>
    <figure style="display: inline-block;">
        <img src="https://github.com/user-attachments/assets/51b8854d-fafa-4d11-b1c6-8004bbd792e7" alt="图片2" width="300">
        <figcaption>bfloat16</figcaption>
    </figure>
</div>
<br>

Diffsynth还支持ControlNet的量化，只需要在加载模型时指定数据类型为  ```torch.float8_e4m3fn```, 并且在生成图片前调用对应ControlNet模型的```quantize()```方法即可：
```python
model_manager.load_models(
    ["models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors"],
    torch_dtype=torch.float8_e4m3fn 
)
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(
        processor_id="tile",
        model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
        scale=0.7
    ),
],device="cuda")
for model in pipe.controlnet.models:
    model.quantize()
```

除了推理阶段，Diffsynth也支持在Lora训练阶段使用模型量化，只需要在训练参数中额外添加`--quantize "float8_e4m3fn"`。

## 卸载

模型卸载技术的思想很简单，只在需要模型进行计算的时候才将模型加载到GPU显存上，使用完毕后将模型卸载至CPU内存中，牺牲模型加载和卸载的时间换取显存消耗。除了本体外，文生图模型的pipeline通常还包括VAE、Text Encoder等模型，在生成图片时会依次调用它们。使用卸载技术可以将显存需求降低至它们之中最大的模型。
Diffsynth支持对所有文生图模型使用卸载技术，要启用模型卸载，需要指定模型被加载至CPU上，pipeline运行在GPU上，再调用`enable_cpu_offload()`启用模型卸载，以Flux为例：

```python
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu") 
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager,device="cuda")
pipe.enable_cpu_offload()
```

## 总结

模型量化和卸载都能有效降低显存消耗，并且互相兼容。模型卸载不会降低生成的图像质量，并且额外消耗的时间不多（受模型大小和设备通讯影响，通常每张图不超过3秒），因此在显存不足时优先推荐使用模型卸载。模型量化会损失部分图像质量，但在float8下质量差别不大。两种显存优化技术同时使用，可以将运行Flux的显存消耗从37GB降低至15GB。

## 支持量化的模型
### Flux

* https://modelscope.cn/models/AI-ModelScope/FLUX.1-dev
* https://modelscope.cn/models/AI-ModelScope/FLUX.1-schnell
### ControlNets

* https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Depth
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Surface-Normals
* https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler
* https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha
* https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta
