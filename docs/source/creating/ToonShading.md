# 当图像模型遇见 AnimateDiff

我们已经领略到了 Stable Diffusion 模型及其生态模型的强大图像生成能力，现在我们引入一个新的模块：AnimateDiff，这样一来就可以把图像模型的能力迁移到视频中。在本篇文章中，我们为您展示基于 DiffSynth-Studio 搭建的动漫风格视频渲染方案：Diffutoon。

## 下载模型

接下来的例子会用到很多模型，我们先把它们下载好。

* 一个动漫风格的 Stable Diffusion 架构模型
* 两个 ControlNet 模型
* 一个 Textual Inversion 模型
* 一个 AnimateDiff 模型

```python
from diffsynth import download_models

download_models([
    "AingDiffusion_v12",
    "AnimateDiff_v2",
    "ControlNet_v11p_sd15_lineart",
    "ControlNet_v11f1e_sd15_tile",
    "TextualInversion_VeryBadImageNegative_v1.3"
])
```

## 下载视频

你可以随意选择任何你喜欢的视频，我们使用[这个视频](https://www.bilibili.com/video/BV1iG411a7sQ)作为演示，你可以通过以下命令下载这个视频文件，但请注意，在没有获得视频原作者的商用版权时，请不要将其用作商业用途。

```
modelscope download --dataset Artiprocher/examples_in_diffsynth data/examples/diffutoon/input_video.mp4 --local_dir ./
```

## 生成动漫

```python
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
import torch

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion/aingdiffusion_v12.safetensors",
    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
    "models/ControlNet/control_v11p_sd15_lineart.pth",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
])

# Build pipeline
pipe = SDVideoPipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path="models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=0.5
        )
    ]
)
pipe.prompter.load_textual_inversions(["models/textual_inversion/verybadimagenegative_v1.3.pt"])

# Load video
video = VideoData(
    video_file="data/examples/diffutoon/input_video.mp4",
    height=1536, width=1536
)
input_video = [video[i] for i in range(30)]

# Generate
torch.manual_seed(0)
output_video = pipe(
    prompt="best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=7, clip_skip=2,
    input_frames=input_video, denoising_strength=1.0,
    controlnet_frames=input_video, num_frames=len(input_video),
    num_inference_steps=10, height=1536, width=1536,
    animatediff_batch_size=16, animatediff_stride=8,
)

# Save video
save_video(output_video, "output_video.mp4", fps=30)
```

## 效果展示

<video width="1024" height="512" controls>
  <source src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd" type="video/mp4">
您的浏览器不支持Video标签。
</video>
