# Image Synthesis

Image synthesis is the base feature of DiffSynth Studio. We can generate images with very high resolution.

### Example: FLUX

Example script: [`flux_text_to_image.py`](./flux_text_to_image.py) and [`flux_text_to_image_low_vram.py`](./flux_text_to_image_low_vram.py)(low VRAM).

The original version of FLUX doesn't support classifier-free guidance; however, we believe that this guidance mechanism is an important feature for synthesizing beautiful images. You can enable it using the parameter `cfg_scale`, and the extra guidance scale introduced by FLUX is `embedded_guidance`.

|1024*1024 (original)|1024*1024 (classifier-free guidance)|2048*2048 (highres-fix)|
|-|-|-|
|![image_1024](https://github.com/user-attachments/assets/ce01327f-068f-45f5-aba9-0fa45eb26199)|![image_1024_cfg](https://github.com/user-attachments/assets/6af5b106-0673-4e58-9213-cd9157eef4c0)|![image_2048_highres](https://github.com/user-attachments/assets/a4bb776f-d9f0-4450-968c-c5d090a3ab4c)|

### Example: Stable Diffusion

Example script: [`sd_text_to_image.py`](./sd_text_to_image.py)

LoRA Training: [`../train/stable_diffusion/`](../train/stable_diffusion/)

|512*512|1024*1024|2048*2048|4096*4096|
|-|-|-|-|
|![512](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/55f679e9-7445-4605-9315-302e93d11370)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/9087a73c-9164-4c58-b2a0-effc694143fb)|![4096](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edee9e71-fc39-4d1c-9ca9-fa52002c67ac)|

### Example: Stable Diffusion XL

Example script: [`sdxl_text_to_image.py`](./sdxl_text_to_image.py)

LoRA Training: [`../train/stable_diffusion_xl/`](../train/stable_diffusion_xl/)

|1024*1024|2048*2048|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/584186bc-9855-4140-878e-99541f9a757f)|

### Example: Stable Diffusion 3

Example script: [`sd3_text_to_image.py`](./sd3_text_to_image.py)

LoRA Training: [`../train/stable_diffusion_3/`](../train/stable_diffusion_3/)

|1024*1024|2048*2048|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4df346db-6f91-420a-b4c1-26e205376098)|![image_2048](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/1386c802-e580-4101-939d-f1596802df9d)|

### Example: Kolors

Example script: [`kolors_text_to_image.py`](./kolors_text_to_image.py)

LoRA Training: [`../train/kolors/`](../train/kolors/)

|1024*1024|2048*2048|
|-|-|
|![image_1024](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/53ef6f41-da11-4701-8665-9f64392607bf)|![image_2048](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/66bb7a75-fe31-44e5-90eb-d3140ee4686d)|

Kolors also support the models trained for SD-XL. For example, ControlNets and LoRAs. See [`kolors_with_sdxl_models.py`](./kolors_with_sdxl_models.py)

LoRA: https://civitai.com/models/73305/zyd232s-ink-style

|Base model|with LoRA (alpha=0.5)|with LoRA (alpha=1.0)|with LoRA (alpha=1.5)|
|-|-|-|-|
|![image_0 0](https://github.com/user-attachments/assets/a222eae3-6e0a-4ea6-b301-99e74e2bc11a)|![image_0 5](https://github.com/user-attachments/assets/e429c501-530c-43f6-a30b-9f97996c91a2)|![image_1 0](https://github.com/user-attachments/assets/0ddeed4b-250d-4b5c-a4fa-2db50f63bf1c)|![image_1 5](https://github.com/user-attachments/assets/db35a89d-6325-4422-921e-14fb6ad66c92)|

ControlNet: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0

|Reference image|Depth image|with ControlNet|with ControlNet|
|-|-|-|-|
|![image_0 0](https://github.com/user-attachments/assets/a222eae3-6e0a-4ea6-b301-99e74e2bc11a)|![controlnet_input](https://github.com/user-attachments/assets/d16b2785-bc1f-4184-b170-ae90f1d704c1)|![image_depth_1](https://github.com/user-attachments/assets/90a94780-7b56-4786-8a25-aae118eda171)|![image_depth_2](https://github.com/user-attachments/assets/05eb1309-9c98-49e7-a8ee-f376ceedf18e)|

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
