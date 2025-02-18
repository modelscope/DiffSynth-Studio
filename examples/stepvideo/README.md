# Stepvideo

StepVideo is a state-of-the-art (SoTA) text-to-video pre-trained model with 30 billion parameters and the capability to generate videos up to 204 frames.

* Model: https://modelscope.cn/models/stepfun-ai/stepvideo-t2v/summary
* GitHub: https://github.com/stepfun-ai/Step-Video-T2V
* Technical report: https://arxiv.org/abs/2502.10248

## Examples

For original BF16 version, please see [`./stepvideo_text_to_video.py`](./stepvideo_text_to_video.py). 80G VRAM required.

We also support auto-offload, which can reduce the VRAM requirement to **24GB**; however, it requires 2x time for inference. Please see [`./stepvideo_text_to_video_low_vram.py`](./stepvideo_text_to_video_low_vram.py).

https://github.com/user-attachments/assets/5954fdaa-a3cf-45a3-bd35-886e3cc4581b

For FP8 quantized version, please see [`./stepvideo_text_to_video_quantized.py`](./stepvideo_text_to_video_quantized.py). 40G VRAM required.

https://github.com/user-attachments/assets/f3697f4e-bc08-47d2-b00a-32d7dfa272ad
