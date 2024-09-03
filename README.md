# DiffSynth Studio
[![PyPI](https://img.shields.io/pypi/v/DiffSynth)](https://pypi.org/project/DiffSynth/)
[![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/DiffSynth-Studio.svg)](https://GitHub.com/modelscope/DiffSynth-Studio/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/DiffSynth-Studio)](https://GitHub.com/modelscope/DiffSynth-Studio/commit/)

<p align="center">
<a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Introduction

DiffSynth Studio is a Diffusion engine. We have restructured architectures including Text Encoder, UNet, VAE, among others, maintaining compatibility with models from the open-source community while enhancing computational performance. We provide many interesting features. Enjoy the magic of Diffusion models!

Until now, DiffSynth Studio has supported the following models:

* [CogVideo](https://huggingface.co/THUDM/CogVideoX-5b)
* [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev)
* [ExVideo](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1)
* [Kolors](https://huggingface.co/Kwai-Kolors/Kolors)
* [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
* [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
* [Hunyuan-DiT](https://github.com/Tencent/HunyuanDiT)
* [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
* [ESRGAN](https://github.com/xinntao/ESRGAN)
* [Ip-Adapter](https://github.com/tencent-ailab/IP-Adapter)
* [AnimateDiff](https://github.com/guoyww/animatediff/)
* [ControlNet](https://github.com/lllyasviel/ControlNet)
* [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
* [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## News

- **August 22, 2024.** CogVideoX-5B is supported in this project. See [here](/examples/video_synthesis/). We provide several interesting features for this text-to-video model, including
  - Text to video
  - Video editing
  - Self-upscaling
  - Video interpolation

- **August 22, 2024.** We have implemented an interesting painter that supports all text-to-image models. Now you can create stunning images using the painter, with assistance from AI!
  - Use it in our [WebUI](#usage-in-webui).

- **August 21, 2024.** FLUX is supported in DiffSynth-Studio.
  - Enable CFG and highres-fix to improve visual quality. See [here](/examples/image_synthesis/README.md)
  - LoRA, ControlNet, and additional models will be available soon.

- **June 21, 2024.** 🔥🔥🔥 We propose ExVideo, a post-tuning technique aimed at enhancing the capability of video generation models. We have extended Stable Video Diffusion to achieve the generation of long videos up to 128 frames.
  - [Project Page](https://ecnu-cilab.github.io/ExVideoProjectPage/)
  - Source code is released in this repo. See [`examples/ExVideo`](./examples/ExVideo/).
  - Models are released on [HuggingFace](https://huggingface.co/ECNU-CILab/ExVideo-SVD-128f-v1) and [ModelScope](https://modelscope.cn/models/ECNU-CILab/ExVideo-SVD-128f-v1).
  - Technical report is released on [arXiv](https://arxiv.org/abs/2406.14130).
  - You can try ExVideo in this [Demo](https://huggingface.co/spaces/modelscope/ExVideo-SVD-128f-v1)!

- **June 13, 2024.** DiffSynth Studio is transferred to ModelScope. The developers have transitioned from "I" to "we". Of course, I will still participate in development and maintenance.

- **Jan 29, 2024.** We propose Diffutoon, a fantastic solution for toon shading.
  - [Project Page](https://ecnu-cilab.github.io/DiffutoonProjectPage/)
  - The source codes are released in this project.
  - The technical report (IJCAI 2024) is released on [arXiv](https://arxiv.org/abs/2401.16224).

- **Dec 8, 2023.** We decide to develop a new Project, aiming to release the potential of diffusion models, especially in video synthesis. The development of this project is started.

- **Nov 15, 2023.** We propose FastBlend, a powerful video deflickering algorithm.
  - The sd-webui extension is released on [GitHub](https://github.com/Artiprocher/sd-webui-fastblend).
  - Demo videos are shown on Bilibili, including three tasks.
    - [Video deflickering](https://www.bilibili.com/video/BV1d94y1W7PE)
    - [Video interpolation](https://www.bilibili.com/video/BV1Lw411m71p)
    - [Image-driven video rendering](https://www.bilibili.com/video/BV1RB4y1Z7LF)
  - The technical report is released on [arXiv](https://arxiv.org/abs/2311.09265).
  - An unofficial ComfyUI extension developed by other users is released on [GitHub](https://github.com/AInseven/ComfyUI-fastblend).

- **Oct 1, 2023.** We release an early version of this project, namely FastSDXL. A try for building a diffusion engine.
  - The source codes are released on [GitHub](https://github.com/Artiprocher/FastSDXL).
  - FastSDXL includes a trainable OLSS scheduler for efficiency improvement.
    - The original repo of OLSS is [here](https://github.com/alibaba/EasyNLP/tree/master/diffusion/olss_scheduler).
    - The technical report (CIKM 2023) is released on [arXiv](https://arxiv.org/abs/2305.14677).
    - A demo video is shown on [Bilibili](https://www.bilibili.com/video/BV1w8411y7uj).
    - Since OLSS requires additional training, we don't implement it in this project.

- **Aug 29, 2023.** We propose DiffSynth, a video synthesis framework.
  - [Project Page](https://ecnu-cilab.github.io/DiffSynth.github.io/).
  - The source codes are released in [EasyNLP](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth).
  - The technical report (ECML PKDD 2024) is released on [arXiv](https://arxiv.org/abs/2308.03463).


## Installation

Install from source code (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Or install from pypi:

```
pip install diffsynth
```

## Usage (in Python code)

The Python examples are in [`examples`](./examples/). We provide an overview here.

### Download Models

Download the pre-set models. Model IDs can be found in [config file](/diffsynth/configs/model_config.py).

```python
from diffsynth import download_models

download_models(["FLUX.1-dev", "Kolors"])
```

Download your own models.

```python
from diffsynth.models.downloader import download_from_huggingface, download_from_modelscope

# From Modelscope (recommended)
download_from_modelscope("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.bin", "models/kolors/Kolors/vae")
# From Huggingface
download_from_huggingface("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.safetensors", "models/kolors/Kolors/vae")
```

### Video Synthesis

#### Text-to-video using CogVideoX-5B

CogVideoX-5B is released by ZhiPu. We provide an improved pipeline, supporting text-to-video, video editing, self-upscaling and video interpolation. [`examples/video_synthesis`](./examples/video_synthesis/)

The video on the left is generated using the original text-to-video pipeline, while the video on the right is the result after editing and frame interpolation.

https://github.com/user-attachments/assets/26b044c1-4a60-44a4-842f-627ff289d006

#### Long Video Synthesis

We trained an extended video synthesis model, which can generate 128 frames. [`examples/ExVideo`](./examples/ExVideo/)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc


#### Toon Shading

Render realistic videos in a flatten style and enable video editing features. [`examples/Diffutoon`](./examples/Diffutoon/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/20528af5-5100-474a-8cdc-440b9efdd86c

#### Video Stylization

Video stylization without video models. [`examples/diffsynth`](./examples/diffsynth/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

### Image Synthesis

Generate high-resolution images, by breaking the limitation of diffusion models! [`examples/image_synthesis`](./examples/image_synthesis/).

LoRA fine-tuning is supported in [`examples/train`](./examples/train/).

|FLUX|Stable Diffusion 3|
|-|-|
|![image_1024_cfg](https://github.com/user-attachments/assets/6af5b106-0673-4e58-9213-cd9157eef4c0)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|

|Kolors|Hunyuan-DiT|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf)|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5)|

|Stable Diffusion|Stable Diffusion XL|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|

## Usage (in WebUI)

Create stunning images using the painter, with assistance from AI!

https://github.com/user-attachments/assets/95265d21-cdd6-4125-a7cb-9fbcf6ceb7b0

**This video is not rendered in real-time.**

Before launching the WebUI, please download models to the folder `./models`. See [here](#download-models).

* `Gradio` version

```
pip install gradio
```

```
python apps/gradio/DiffSynth_Studio.py
```

![20240822102002](https://github.com/user-attachments/assets/59613157-de51-4109-99b3-97cbffd88076)

* `Streamlit` version

```
pip install streamlit streamlit-drawable-canvas
```

```
python -m streamlit run apps/streamlit/DiffSynth_Studio.py
```

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/93085557-73f3-4eee-a205-9829591ef954
