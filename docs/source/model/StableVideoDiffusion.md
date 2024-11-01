# Stable Video Diffusion

## 相关链接

* 论文：[Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127)
* 模型
    * Stable Video Diffusion v1
        * [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-video-diffusion-img2vid)
    * Stable Video Diffusion v1-xt
        * [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-video-diffusion-img2vid-xt)
    * Stable Video Diffusion v1.1-xt
        * [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
        * [ModelScope](https://modelscope.cn/models/cjc1887415157/stable-video-diffusion-img2vid-xt-1-1)

## 模型介绍

Stable Video Diffusion 模型是 StabilityAI 训练并开源的图生视频模型，该模型与 Stable Diffusion 模型类似，采用三段式的模型架构。

* Image Encoder 采用了 CLIP 模型中的 ViT 部分，用于将输入的图像转化为 Embedding。
* VAE 分为 Encoder 和 Decoder 部分，Encoder 部分与 Stable Diffusion v1.x 完全相同，仅在图像层面对视频进行逐帧压缩；Decoder 部分在 Stable Diffusion v1.x VAE Decoder 的基础上增加了 3D 的卷积层并进一步进行了训练，用于消除逐帧处理过程中的闪烁问题。
* UNet 部分同时将 Image Encoder 和 VAE Encoder 的输出作为输入，用于在 Latent Space 中进行迭代去噪。

Stable Video Diffusion 模型可以把输入的图像作为视频第一帧，并生成后续的 24 帧。但值得注意的是，虽然理论上可以继续分段生成更长视频，但分段之间缺乏连续性，因此我们不建议用这个模型分段生成较长视频。

Stable Video Diffusion 的生成效果：

<video width="512" height="256" controls>
  <source src="https://github.com/user-attachments/assets/2696b50c-96b8-48fd-a30e-7f69c3c6839c" type="video/mp4">
您的浏览器不支持Video标签。
</video>

## 代码样例

```python
from diffsynth import save_video, ModelManager, SVDVideoPipeline
from PIL import Image


model_manager = ModelManager(model_id_list=["stable-video-diffusion-img2vid-xt"])
pipe = SVDVideoPipeline.from_model_manager(model_manager)
video = pipe(
    input_image=Image.open("your_input_image.png").resize((1024, 576)),
    num_frames=25, fps=15, height=576, width=1024,
    motion_bucket_id=127,
    num_inference_steps=50
)
save_video(video, "output_video.mp4", fps=15, quality=5)
```
