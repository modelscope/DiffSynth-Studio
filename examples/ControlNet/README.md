# ControlNet

We provide extensive ControlNet support. Taking the FLUX model as an example, we support many different ControlNet models that can be freely combined, even if their structures differ. Additionally, ControlNet models are compatible with high-resolution refinement and partition control techniques, enabling very powerful controllable image generation.

These examples are in [`flux_controlnet.py`](./flux_controlnet.py).

## Canny/Depth/Normal: Structure Control

Structural control is the most fundamental capability of the ControlNet model. By using Canny to extract edge information, or by utilizing depth maps and normal maps, we can extract the structure of an image, which can then serve as control information during the image generation process.

Model link: https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha

For example, if we generate an image of a cat and use a model like InstantX/FLUX.1-dev-Controlnet-Union-alpha that supports multiple control conditions, we can simultaneously enable both Canny and Depth controls to transform the environment into a twilight setting.

|![image_5](https://github.com/user-attachments/assets/19d2abc4-36ae-4163-a8da-df5732d1a737)|![image_6](https://github.com/user-attachments/assets/28378271-3782-484c-bd51-3d3311dd85c6)|
|-|-|

The control strength of ControlNet for structure can be adjusted. For example, in the case below, when we move the girl from summer to winter, we can appropriately lower the control strength of ControlNet so that the model will adapt to the content of the image and change her into warm clothes.

|![image_7](https://github.com/user-attachments/assets/a7b8555b-bfd9-4e92-aa77-16bca81b07e3)|![image_8](https://github.com/user-attachments/assets/a1bab36b-6cce-4f29-8233-4cb824b524a8)|
|-|-|

## Upscaler/Tile/Blur: High-Resolution Image Synthesis

There are many ControlNet models that support high definition, such as:

Model link: https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler, https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha, https://modelscope.cn/models/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro

These models can transform blurry, noisy low-quality images into clear ones. In DiffSynth-Studio, the native high-resolution patch processing technology supported by the framework can overcome the resolution limitations of the models, enabling image generation at resolutions of 2048 or even higher, significantly enhancing the capabilities of these models. In the example below, we can see that in the high-definition image enlarged to 2048 resolution, the cat's fur is rendered in exquisite detail, and the skin texture of the characters is delicate and realistic.

|![image_1](https://github.com/user-attachments/assets/9038158a-118c-4ad7-ab01-22865f6a06fc)|![image_2](https://github.com/user-attachments/assets/88583a33-cd74-4cb9-8fd4-c6e14c0ada0c)|
|-|-|

|![image_3](https://github.com/user-attachments/assets/13061ecf-bb57-448a-82c6-7e4655c9cd85)|![image_4](https://github.com/user-attachments/assets/0b7ae80f-de58-4d1d-a49c-ad17e7631bdc)|
|-|-|

## Inpaint: Image Restoration

The Inpaint ControlNet model can repaint specific areas in an image. For example, we can put sunglasses on a cat.

Model link: https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta

|![image_9](https://github.com/user-attachments/assets/babddad0-2d67-4624-b77a-c953250ebdab)|![mask_9](https://github.com/user-attachments/assets/d5bc2878-1817-457a-bdfa-200f955233d3)|![image_10](https://github.com/user-attachments/assets/e3197f2c-190b-4522-83ab-a2e0451b39f6)|
|-|-|-|

However, we noticed that the head movements of the cat have changed. If we want to preserve the original structural features, we can use the Canny, Depth, and Normal models. DiffSynth-Studio provides seamless support for ControlNet of different structures. By using a Normal ControlNet, we can ensure that the structure of the image remains unchanged during local redrawing.

Model link: https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Surface-Normals

|![image_11](https://github.com/user-attachments/assets/c028e6fc-5125-4cba-b35a-b6211c2e6600)|![mask_11](https://github.com/user-attachments/assets/1928ee9a-7594-4c6e-9c71-5bd0b043d8f4)|![image_12](https://github.com/user-attachments/assets/97b3b9e1-f821-405e-971b-9e1c31a209aa)|
|-|-|-|

## MultiControlNet+MultiDiffusion: Fine-Grained Control

DiffSynth-Studio not only supports the simultaneous activation of multiple ControlNet structures, but also allows for the partitioned control of content within an image using different prompts. Additionally, it supports the chunk processing of ultra-high-resolution large images, enabling us to achieve extremely detailed high-level control. Next, we will showcase the creative process behind a beautiful image.

First, use the prompt "a beautiful Asian woman and a cat on a bed. The woman wears a dress" to generate a cat and a young girl.

![image_13](https://github.com/user-attachments/assets/8da006e4-0e68-4fa5-b407-31ef5dbe8e5a)

Then, enable Inpaint ControlNet and Canny ControlNet.

Model link: https://modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta, https://modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha

We control the image using two component.

|Prompt: an orange cat, highly detailed|Prompt: a girl wearing a red camisole|
|-|-|
|![mask_13_1](https://github.com/user-attachments/assets/188530a0-913c-48db-a7f1-62f0384bfdc3)|![mask_13_2](https://github.com/user-attachments/assets/99c4d0d5-8cc3-47a0-8e56-ceb37db4dfdc)|

Generate!

![image_14](https://github.com/user-attachments/assets/f5b9d3dd-a690-4597-91a8-a019c6fc2523)

The background is a bit blurry, so we use deblurring LoRA for image-to-image generation.

Model link: https://modelscope.cn/models/LiblibAI/FLUX.1-dev-LoRA-AntiBlur

![image_15](https://github.com/user-attachments/assets/32ed2667-2260-4d80-aaa9-4435d6920a2a)

The entire image is much clearer now. Next, let's use the high-definition model to increase the resolution to 4096*4096!

Model link: https://modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler

![image_17](https://github.com/user-attachments/assets/1a688a12-1544-4973-8aca-aa3a23cb34c1)

Zoom in to see details.

![image_17_cropped](https://github.com/user-attachments/assets/461a1fbc-9ffa-4da5-80fd-e1af9667c804)

Enjoy!
