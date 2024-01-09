# DiffSynth Studio

## Introduction

DiffSynth is a new Diffusion engine. We have restructured architectures including Text Encoder, UNet, VAE, among others, maintaining compatibility with models from the open-source community while enhancing computational performance. This version is currently in its initial stage, supporting SD and SDXL architectures. In the future, we plan to develop more interesting features based on this new codebase.

## Installation

Create Python environment:

```
conda env create -f environment.yml
```

We find that sometimes `conda` cannot install `cupy` correctly, please install it manually. See [this document](https://docs.cupy.dev/en/stable/install.html) for more details.

Enter the Python environment:

```
conda activate DiffSynthStudio
```

## Usage (in WebUI)

```
python -m streamlit run Diffsynth_Studio.py
```

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/93085557-73f3-4eee-a205-9829591ef954

## Usage (in Python code)

### Example 1: Stable Diffusion

We can generate images with very high resolution. Please see `examples/sd_text_to_image.py` for more details.

|512*512|1024*1024|2048*2048|4096*4096|
|-|-|-|-|
|![512](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/55f679e9-7445-4605-9315-302e93d11370)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/9087a73c-9164-4c58-b2a0-effc694143fb)|![4096](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edee9e71-fc39-4d1c-9ca9-fa52002c67ac)|

### Example 2: Stable Diffusion XL

Generate images with Stable Diffusion XL. Please see `examples/sdxl_text_to_image.py` for more details.

|1024*1024|2048*2048|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/584186bc-9855-4140-878e-99541f9a757f)|

### Example 3: Stable Diffusion XL Turbo

Generate images with Stable Diffusion XL Turbo. You can see `examples/sdxl_turbo.py` for more details, but we highly recommend you to use it in the WebUI.

|"black car"|"red car"|
|-|-|
|![black_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7fbfd803-68d4-44f3-8713-8c925fec47d0)|![black_car_to_red_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/aaf886e4-c33c-4fd8-98e2-29eef117ba00)|

### Example 4: Toon Shading

A very interesting example. Please see `examples/sd_toon_shading.py` for more details.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/607c199b-6140-410b-a111-3e4ffb01142c

### Example 5: Text to Video

Given a prompt, DiffSynth Studio can generate a video using a Stable Diffusion model and an AnimateDiff model. We can break the limitation of number of frames! See `examples/sd_text_to_video.py`.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/8f556355-4079-4445-9b48-e9da77699437

### Example 6: Video Stylization

We provide an example for video stylization. In this pipeline, the rendered video is completely different from the original video, thus we need a powerful deflickering algorithm. We use FastBlend to implement the deflickering module. Please see `examples/sd_video_rerender.py` for more details.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea
