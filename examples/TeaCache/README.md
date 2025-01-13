# TeaCache

TeaCache ([Timestep Embedding Aware Cache](https://github.com/ali-vilab/TeaCache)) is a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference.

## Examples

We provide examples on FLUX.1-dev. See [./flux_teacache.py](./flux_teacache.py).

Steps: 50

GPU: A100

|TeaCache is disabled|tea_cache_l1_thresh=0.2|tea_cache_l1_thresh=0.4|tea_cache_l1_thresh=0.6|tea_cache_l1_thresh=0.8|
|-|-|-|-|-|
|23s|13s|9s|6s|5s|
|![image_None](https://github.com/user-attachments/assets/2bf5187a-9693-44d3-9ebb-6c33cd15443f)|![image_0 2](https://github.com/user-attachments/assets/5532ba94-c7e2-446e-a9ba-1c68c0f63350)|![image_0 4](https://github.com/user-attachments/assets/4c57c50d-87cd-493b-8603-1da57ec3b70d)|![image_0 6](https://github.com/user-attachments/assets/1d95a3a9-71f9-4b1a-ad5f-a5ea8d52eca7)|![image_0 8](https://github.com/user-attachments/assets/d8cfdd74-8b45-4048-b1b7-ce480aa23fa1)
