# Image Synthesis

Image synthesis is the base feature of DiffSynth Studio. We can generate images with very high resolution.

### Example: Stable Diffusion

Example script: [`sd_text_to_image.py`](./sd_text_to_image.py)

|512*512|1024*1024|2048*2048|4096*4096|
|-|-|-|-|
|![512](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/55f679e9-7445-4605-9315-302e93d11370)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/9087a73c-9164-4c58-b2a0-effc694143fb)|![4096](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edee9e71-fc39-4d1c-9ca9-fa52002c67ac)|

### Example: Stable Diffusion XL

Example script: [`sdxl_text_to_image.py`](./sdxl_text_to_image.py)

|1024*1024|2048*2048|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/584186bc-9855-4140-878e-99541f9a757f)|

### Example: Stable Diffusion 3

Example script: [`sd3_text_to_image.py`](./sd3_text_to_image.py)

LoRA Training: [`../train/stable_diffusion_3/`](../train/stable_diffusion_3/)

|1024*1024|2048*2048|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|![image_2048](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/1386c802-e580-4101-939d-f1596802df9d)|

### Example: Hunyuan-DiT

Example script: [`hunyuan_dit_text_to_image.py`](./hunyuan_dit_text_to_image.py)

LoRA Training: [`../train/hunyuan_dit/`](../train/hunyuan_dit/)

|1024*1024|2048*2048|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/60b022c8-df3f-4541-95ab-bf39f2fa8bb5)|![image_2048](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/87919ea8-d428-4963-8257-da05f3901bbb)|

### Example: Stable Diffusion XL Turbo

Example script: [`sdxl_turbo.py`](./sdxl_turbo.py)

We highly recommend you to use this model in the WebUI.

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

### Example: Stable Diffusion 3 with Textual Inversions (Experimental)

Since Stable Diffusion 3 utilizes the same text encoder as Stable Diffusion 1.x, it supports the textual inversions designed for Stable Diffusion 1.x. However, we found that the textual inversions may cause unpredictable effects to the model. We can only guarantee that these textual inversions can be loaded into the model. The example script is [`sd3_text_to_image_textual_inversion.py`](./sd3_text_to_image_textual_inversion.py)

Prompt: "a girl, highly detailed, absurd res, perfect image". Without any textual inversions.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![image_without_textual_inversion_0](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4e918bf8-6081-4f79-a043-87adc4047d92)|![image_without_textual_inversion_1](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/2e90a01f-6a83-46ba-99b6-ab085582a5b7)|![image_without_textual_inversion_2](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/83570a6f-cddd-4d0a-8b2f-f50388e2ca8a)|![image_without_textual_inversion_3](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/f4d0f2d4-80ee-4281-923e-77d87e3d37b1)|

Prompt: "a girl, highly detailed, absurd res, perfect image". With [`verybadimagenegative_v1.3`](https://civitai.com/models/11772/verybadimagenegative) on the negative side.

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![image_with_textual_inversion_0](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/1b3485ee-e7c1-4306-8f93-c9f32d1ac937)|![image_with_textual_inversion_1](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/5d7c6c4b-afdf-42b0-8e94-1959f1a44491)|![image_with_textual_inversion_2](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/92e93c4e-2781-41df-a246-2d2e9bde97c4)|![image_with_textual_inversion_3](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/070966a0-3d5c-48d8-8199-9d7c80408689)|
