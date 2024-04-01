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
python -m streamlit run DiffSynth_Studio.py
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

### Example 4: Toon Shading (Diffutoon)

This example is implemented based on [Diffutoon](https://arxiv.org/abs/2401.16224). This approach is adept for rendering high-resoluton videos with rapid motion. You can easily modify the parameters in the config dict. See `examples/diffutoon_toon_shading.py`. We also provide [an example on Colab](https://colab.research.google.com/github/Artiprocher/DiffSynth-Studio/blob/main/examples/Diffutoon.ipynb).

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

### Example 5: Toon Shading with Editing Signals (Diffutoon)

This example is implemented based on [Diffutoon](https://arxiv.org/abs/2401.16224), supporting video editing signals. See `examples\diffutoon_toon_shading_with_editing_signals.py`. The editing feature is also supported in the [Colab example](https://colab.research.google.com/github/Artiprocher/DiffSynth-Studio/blob/main/examples/Diffutoon.ipynb).

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/20528af5-5100-474a-8cdc-440b9efdd86c

### Example 6: Toon Shading (in native Python code)

This example is provided for developers. If you don't want to use the config to manage parameters, you can see `examples/sd_toon_shading.py` to learn how to use it in native Python code.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/607c199b-6140-410b-a111-3e4ffb01142c

### Example 7: Text to Video

Given a prompt, DiffSynth Studio can generate a video using a Stable Diffusion model and an AnimateDiff model. We can break the limitation of number of frames! See `examples/sd_text_to_video.py`.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/8f556355-4079-4445-9b48-e9da77699437

### Example 8: Video Stylization

We provide an example for video stylization. In this pipeline, the rendered video is completely different from the original video, thus we need a powerful deflickering algorithm. We use FastBlend to implement the deflickering module. Please see `examples/sd_video_rerender.py` for more details.

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

### Example 9: Prompt Processing

If you are not native English user, we provide translation service for you. Our prompter can translate other language to English and refine it using "BeautifulPrompt" models. Please see `examples/sd_prompt_refining.py` for more details.

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/ebb25ca8-7ce1-4d9e-8081-59a867c70c4d)|![1_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a7e79853-3c1a-471a-9c58-c209ec4b76dd)|![2_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a292b959-a121-481f-b79c-61cc3346f810)|![3_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1c19b54e-5a6f-4d48-960b-a7b2b149bb4c)|

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English. Then the [refining model](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd) will refine the translated prompt for better visual quality.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/778b1bd9-44e0-46ac-a99c-712b3fc9aaa4)|![1](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/c03479b8-2082-4c6e-8e1c-3582b98686f6)|![2](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edb33d21-3288-4a55-96ca-a4bfe1b50b00)|![3](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7848cfc1-cad5-4848-8373-41d24e98e584)|
