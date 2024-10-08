# 扩展功能

本文档介绍了一些在 DiffSynth 实现的 Diffusion 模型之外的相关技术，这些模型在图像和视频处理方面具有显著的应用潜力。

- **[RIFE](https://github.com/hzwer/ECCV2022-RIFE)**：RIFE 是一个基于实时中间流估计的帧插值方法。采用 IFNet 结构的模型，能够以很快的速度端到端估计中间流。RIFE 不依赖于预训练的光流模型，能够支持任意时间步的帧插值，通过时间编码输入进行处理。

    在这段代码中，我们用 RIFE 模型把视频的帧数提升到原来的两倍。

    ```python
    from diffsynth import VideoData, ModelManager, save_video
    from diffsynth.extensions.RIFE import RIFEInterpolater

    model_manager = ModelManager(model_id_list=["RIFE"])
    rife = RIFEInterpolater.from_model_manager(model_manager)
    video = VideoData("input_video.mp4", height=512, width=768).raw_data()
    video = rife.interpolate(video)
    save_video(video, "output_video.mp4", fps=60)
    ```

- **[ESRGAN](https://github.com/xinntao/ESRGAN)**: ESRGAN 是一个图像超分辨率模型，能够实现四倍的分辨率提升。该方法通过优化网络架构、对抗损失和感知损失，显著提升了生成图像的真实感。

    在这段代码中，我们用 ESRGAN 模型把图像分辨率提升到原来的四倍。

    ```python
    from PIL import Image
    from diffsynth import ModelManager
    from diffsynth.extensions.ESRGAN import ESRGAN

    model_manager = ModelManager(model_id_list=["ESRGAN_x4"])
    rife = ESRGAN.from_model_manager(model_manager)
    image = Image.open("input_image.jpg")
    image = rife.upscale(image)
    image.save("output_image.jpg")
    ```

- **[FastBlend](https://arxiv.org/abs/2311.09265)**: FastBlend 不依赖模型的视频去闪烁算法，在使用图像生成模型逐帧处理过的视频（风格视频）中，通常会出现闪烁问题，FastBlend 则可以根据原视频（引导视频）中的运动特征，消除风格视频中的闪烁。

    在这段代码中，我们用 FastBlend 把风格视频中的闪烁效果删除。

    ```python
    from diffsynth import VideoData, save_video
    from diffsynth.extensions.FastBlend import FastBlendSmoother

    fastblend = FastBlendSmoother()
    guide_video = VideoData("guide_video.mp4", height=512, width=768).raw_data()
    style_video = VideoData("style_video.mp4", height=512, width=768).raw_data()
    output_video = fastblend(style_video, original_frames=guide_video)
    save_video(output_video, "output_video.mp4", fps=30)
    ```
