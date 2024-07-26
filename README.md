# DiffSynth Studio

<p align="center">
<a href="https://trendshift.io/repositories/10946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10946" alt="modelscope%2FDiffSynth-Studio | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## Introduction

DiffSynth Studio is a Diffusion engine. We have restructured architectures including Text Encoder, UNet, VAE, among others, maintaining compatibility with models from the open-source community while enhancing computational performance. We provide many interesting features. Enjoy the magic of Diffusion models!

Until now, DiffSynth Studio has supported the following models:

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

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## Usage (in Python code)

The Python examples are in [`examples`](./examples/). We provide an overview here.

### Long Video Synthesis

We trained an extended video synthesis model, which can generate 128 frames. [`examples/ExVideo`](./examples/ExVideo/)

https://github.com/modelscope/DiffSynth-Studio/assets/35051019/d97f6aa9-8064-4b5b-9d49-ed6001bb9acc

### Image Synthesis

Generate high-resolution images, by breaking the limitation of diffusion models! [`examples/image_synthesis`](./examples/image_synthesis/).

LoRA fine-tuning is supported in [`examples/train`](./examples/train/).

|Model|Example|
|-|-|
|Stable Diffusion|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|
|Stable Diffusion XL|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|
|Stable Diffusion 3|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|
|Kolors|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf)|
|Hunyuan-DiT|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5)|

### Toon Shading

Render realistic videos in a flatten style and enable video editing features. [`examples/Diffutoon`](./examples/Diffutoon/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/20528af5-5100-474a-8cdc-440b9efdd86c

### Video Stylization

Video stylization without video models. [`examples/diffsynth`](./examples/diffsynth/)

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

## Usage (in WebUI)

```
python -m streamlit run DiffSynth_Studio.py
```

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/93085557-73f3-4eee-a205-9829591ef954
