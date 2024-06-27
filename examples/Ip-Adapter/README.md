# IP-Adapter

IP-Adapter is a interesting model, which can adopt the content or style of another image to generate a new image.

## Example: Content Controlling in Stable Diffusion

Based on Stable Diffusion, we can transfer the object to another scene. See [`sd_ipadapter.py`](./sd_ipadapter.py).

|First, we generate a car. The prompt is "masterpiece, best quality, a car".|Next, utilizing IP-Adapter, we move the car to the road. The prompt is "masterpiece, best quality, a car running on the road".|
|-|-|
|![car](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8530a2f0-f610-4269-a22c-ac6c2f21fc18)|![car_on_the_road](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/b8ccddb2-c423-46d8-bd1a-327fcc074a36)|

## Example: Content and Style Controlling in Stable Diffusion XL

The IP-Adapter model based on Stable Diffusion XL is more powerful. You have the option to use the content or style. See [`sdxl_ipadapter.py`](./sdxl_ipadapter.py).

* Content controlling (original usage of IP-Adapter)

|First, we generate a rabbit.|Next, enable IP-Adapter and let the rabbit jump.|For comparision, disable IP-Adapter to see the generated image.|
|-|-|-|
|![rabbit](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4b452634-ec57-414f-897a-f8c50c74a650)|![rabbit_to_jumping_rabbit](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/b93c5495-0b77-4d97-bcd3-3942858288f2)|![rabbit_to_jumping_rabbit_without_ipa](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/52f37195-65b3-4a38-8d9b-73df37311c15)|


* Style controlling (InstantStyle)

|First, we generate a rabbit.|Next, enable InstantStyle and convert the rabbit to a cat.|For comparision, disable IP-Adapter to see the generated image.|
|-|-|-|
|![rabbit](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/4b452634-ec57-414f-897a-f8c50c74a650)|![rabbit_to_cat](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/a006b281-f643-4ea9-b0da-712289c96059)|![rabbit_to_cat_without_ipa](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/189bd11e-7a10-4c09-8554-0eebde9150fd)|

## Example: Image Fusing (Experimental)

Since IP-Adapter can control the content based on more than one image, we can do something interesting. See [`sdxl_ipadapter_multi_reference.py`](sdxl_ipadapter_multi_reference.py).

We have two pokemons here:

|Charizard|Pikachu|
|-|-|
|![](https://media.52poke.com/wiki/7/7e/006Charizard.png)|![](https://media.52poke.com/wiki/0/0d/025Pikachu.png)|

Fuse!

|Pikazard ???|
|-|
|![Pikazard](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/807cdb31-94f5-4cc2-a978-3c6a7ffedc5b)|
