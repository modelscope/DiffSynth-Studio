# Image Synthesis

Image synthesis is the base feature of DiffSynth Studio.

### Example: Stable Diffusion

We can generate images with very high resolution. Please see [`sd_text_to_image.py`](./sd_text_to_image.py) for more details.

|512*512|1024*1024|2048*2048|4096*4096|
|-|-|-|-|
|![512](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/55f679e9-7445-4605-9315-302e93d11370)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/9087a73c-9164-4c58-b2a0-effc694143fb)|![4096](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edee9e71-fc39-4d1c-9ca9-fa52002c67ac)|

### Example: Stable Diffusion XL

Generate images with Stable Diffusion XL. Please see [`sdxl_text_to_image.py`](./sdxl_text_to_image.py) for more details.

|1024*1024|2048*2048|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/584186bc-9855-4140-878e-99541f9a757f)|

### Example: Stable Diffusion XL Turbo

Generate images with Stable Diffusion XL Turbo. You can see [`sdxl_turbo.py`](./sdxl_turbo.py) for more details, but we highly recommend you to use it in the WebUI.

|"black car"|"red car"|
|-|-|
|![black_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7fbfd803-68d4-44f3-8713-8c925fec47d0)|![black_car_to_red_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/aaf886e4-c33c-4fd8-98e2-29eef117ba00)|

### Example: Prompt Processing

If you are not native English user, we provide translation service for you. Our prompter can translate other language to English and refine it using "BeautifulPrompt" models. Please see [`sd_prompt_refining.py`](./sd_prompt_refining.py) for more details.

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/ebb25ca8-7ce1-4d9e-8081-59a867c70c4d)|![1_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a7e79853-3c1a-471a-9c58-c209ec4b76dd)|![2_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a292b959-a121-481f-b79c-61cc3346f810)|![3_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1c19b54e-5a6f-4d48-960b-a7b2b149bb4c)|

Prompt: "一个漂亮的女孩". The [translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) will translate it to English. Then the [refining model](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd) will refine the translated prompt for better visual quality.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/778b1bd9-44e0-46ac-a99c-712b3fc9aaa4)|![1](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/c03479b8-2082-4c6e-8e1c-3582b98686f6)|![2](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edb33d21-3288-4a55-96ca-a4bfe1b50b00)|![3](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7848cfc1-cad5-4848-8373-41d24e98e584)|
