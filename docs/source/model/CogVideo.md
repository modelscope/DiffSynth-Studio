# CogVideoX

## 相关链接

* 论文：[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072)
* 模型
    * CogVideoX-5B
        * [HuggingFace](https://huggingface.co/THUDM/CogVideoX-5b)
        * [ModelScope](https://modelscope.cn/models/ZhipuAI/CogVideoX-5b)

## 模型介绍

CogVideoX 是由智谱团队训练并开源的视频生成模型，模型结构分为 Text Encoder、VAE、DiT。

* Text Encoder 模型为 T5，与 Stable Diffusion 3 以及 FLUX 一致。
* VAE 部分为 3D 的 Causal VAE，将 8x8x4 的区域压缩成一个 Embedding。其中视频的第一帧单独处理，后续的每 4 帧合并为一组 Embedding。
* DiT 部分采用了与 Stable Diffusion 3 类似的结构，对视频进行 patch 化之后由连读的多个 transformer 模块处理。

![image](https://github.com/user-attachments/assets/d1abec28-4a51-41b7-9f1d-be62d1975f52)

CogVideoX-5B 模型可以生成长达 49 帧视频，FPS 为 8，效果如下：

<video width="512" height="256" controls>
  <source src="an astronaut riding a horse on Mars." type="video/mp4">
您的浏览器不支持Video标签。
</video>

## 代码样例

```python
from diffsynth import ModelManager, save_video, CogVideoPipeline
import torch


model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["CogVideoX-5B"])
pipe = CogVideoPipeline.from_model_manager(model_manager)
video = pipe(
    prompt="a dog",
    height=480, width=720,
    cfg_scale=7.0, num_inference_steps=200
)
save_video(video, "video.mp5", fps=8, quality=5)
```
