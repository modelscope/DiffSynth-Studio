# ExVideo

## 相关链接

* 论文：[ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning](https://arxiv.org/abs/2406.14130)
* 模型
    * ExVideo-CogVideoX
        * [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1)
        * [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1)
    * ExVideo-SVD
        * [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)
        * [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1)

## 模型介绍

ExVideo 是一种视频生成模型的后训练（post-training）方法，旨在增强视频生成模型的能力，使其能够生成更长的视频。目前，ExVideo 已经发布了两个版本，分别将 Stable Video Diffusion 扩展到 128 帧、将 CogVideoX-5B 扩展到 129 帧。

在基于 Stable Video Diffusion 的 ExVideo 扩展模块中，静态的位置编码被替换为了可训练的参数矩阵，并在时序模块中添加了额外的单位卷积（Identidy 3D Convolution），在保留预训练模型本身能力的前提下，使其能够捕获更长时间尺度上的信息，从而生成更长视频。而在基于 CogVideoX-5B 的 ExVideo 扩展模块中，由于模型基础架构为 DiT，为保证计算效率，扩展模块采用 LoRA 的形式构建。

![](https://github.com/user-attachments/assets/94aa31ba-3ee3-4421-9713-83333a165660)

为了在有限的计算资源上实现长视频的训练，ExVideo 做了很多工程优化，包括：

* Parameter freezing：冻结除了扩展模块以外的所有参数
* Mixed precision：扩展模块部分以全精度维护，其他部分以 BFloat16 精度维护
* Gradient checkpointing：在前向传播时丢弃中间变量，并反向传播时重新计算
* Flash attention：在所有注意力机制上启用加速过的注意力实现
* Shard optimizer states and gradients：基于 DeepSpeed 把部分参数分拆到多个 GPU 上

Stable Video Diffusion + ExVideo 的生成效果：

<video width="512" height="256" controls>
  <source src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc" type="video/mp4">
您的浏览器不支持Video标签。
</video>

CogVideoX-5B + ExVideo 的生成效果：

<video width="512" height="256" controls>
  <source src="https://github.com/user-attachments/assets/321ee04b-8c17-479e-8a95-8cbcf21f8d7e" type="video/mp4">
您的浏览器不支持Video标签。
</video>

## 代码样例

ExVideo-SVD

```python
from diffsynth import save_video, ModelManager, SVDVideoPipeline
import torch, requests
from PIL import Image


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             model_id_list=["stable-video-diffusion-img2vid-xt", "ExVideo-SVD-128f-v1"])
pipe = SVDVideoPipeline.from_model_manager(model_manager)

# Generate a video
torch.manual_seed(0)
image = Image.open(requests.get("https://www.modelscope.cn/api/v1/studio/ECNU-CILab/ExVideo-SVD-128f-v1/repo?Revision=master&FilePath=images%2F0.png", stream=True).raw)
image.save("image.png")
video = pipe(
    input_image=image.resize((512, 512)),
    num_frames=128, fps=30, height=512, width=512,
    motion_bucket_id=127,
    num_inference_steps=50,
    min_cfg_scale=2, max_cfg_scale=2, contrast_enhance_scale=1.2
)
save_video(video, "video.mp4", fps=30)
```

ExVideo-CogVideoX

```python
from diffsynth import ModelManager, CogVideoPipeline, save_video, download_models
import torch


download_models(["CogVideoX-5B", "ExVideo-CogVideoX-LoRA-129f-v1"])
model_manager = ModelManager(torch_dtype=torch.bfloat16)
model_manager.load_models([
    "models/CogVideo/CogVideoX-5b/text_encoder",
    "models/CogVideo/CogVideoX-5b/transformer",
    "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
])
model_manager.load_lora("models/lora/ExVideo-CogVideoX-LoRA-129f-v1.safetensors")
pipe = CogVideoPipeline.from_model_manager(model_manager)

torch.manual_seed(6)
video = pipe(
    prompt="an astronaut riding a horse on Mars.",
    height=480, width=720, num_frames=129,
    cfg_scale=7.0, num_inference_steps=100,
)
save_video(video, "video_with_lora.mp4", fps=8, quality=5)
```