# DiffSynth

DiffSynth is the initial version of our video synthesis framework. In this framework, you can apply video deflickering algorithms to the latent space of diffusion models. You can refer to the [original repo](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth) for more details.

We provide an example for video stylization. In this pipeline, the rendered video is completely different from the original video, thus we need a powerful deflickering algorithm. We use FastBlend to implement the deflickering module. Please see [`sd_video_rerender.py`](./sd_video_rerender.py).

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea
