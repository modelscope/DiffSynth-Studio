# TeaCache

TeaCache ([Timestep Embedding Aware Cache](https://github.com/ali-vilab/TeaCache)) is a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference.

## Examples

### FLUX

Script: [./flux_teacache.py](./flux_teacache.py)

Model: FLUX.1-dev

Steps: 50

GPU: A100

|TeaCache is disabled|tea_cache_l1_thresh=0.2|tea_cache_l1_thresh=0.8|
|-|-|-|
|23s|13s|5s|
|![image_None](https://github.com/user-attachments/assets/2bf5187a-9693-44d3-9ebb-6c33cd15443f)|![image_0 2](https://github.com/user-attachments/assets/5532ba94-c7e2-446e-a9ba-1c68c0f63350)|![image_0 8](https://github.com/user-attachments/assets/d8cfdd74-8b45-4048-b1b7-ce480aa23fa1)

### Hunyuan Video

Script: [./hunyuanvideo_teacache.py](./hunyuanvideo_teacache.py)

Model: Hunyuan Video

Steps: 30

GPU: A100

The following video was generated using TeaCache. It is nearly identical to [the video without TeaCache enabled](https://github.com/user-attachments/assets/48dd24bb-0cc6-40d2-88c3-10feed3267e9), but with double the speed.

https://github.com/user-attachments/assets/cd9801c5-88ce-4efc-b055-2c7737166f34
