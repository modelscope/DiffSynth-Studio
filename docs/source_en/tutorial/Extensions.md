# Extension Features

This document introduces some technologies related to the Diffusion models implemented in DiffSynth, which have significant application potential in image and video processing.

- **[RIFE](https://github.com/hzwer/ECCV2022-RIFE)**: RIFE is a frame interpolation method based on real-time intermediate flow estimation. It uses a model with an IFNet structure that can quickly estimate intermediate flows end-to-end. RIFE does not rely on pre-trained optical flow models and supports frame interpolation at arbitrary time steps, processing through time-encoded inputs.

    In this code snippet, we use the RIFE model to double the frame rate of a video.

    ```python
    from diffsynth import VideoData, ModelManager, save_video
    from diffsynth.extensions.RIFE import RIFEInterpolater

    model_manager = ModelManager(model_id_list=["RIFE"])
    rife = RIFEInterpolater.from_model_manager(model_manager)
    video = VideoData("input_video.mp4", height=512, width=768).raw_data()
    video = rife.interpolate(video)
    save_video(video, "output_video.mp4", fps=60)
    ```

- **[ESRGAN](https://github.com/xinntao/ESRGAN)**: ESRGAN is an image super-resolution model that can achieve a fourfold increase in resolution. This method significantly enhances the realism of generated images by optimizing network architecture, adversarial loss, and perceptual loss.

    In this code snippet, we use the ESRGAN model to quadruple the resolution of an image.

    ```python
    from PIL import Image
    from diffsynth import ModelManager
    from diffsynth.extensions.ESRGAN import ESRGAN

    model_manager = ModelManager(model_id_list=["ESRGAN_x4"])
    esrgan = ESRGAN.from_model_manager(model_manager)
    image = Image.open("input_image.jpg")
    image = esrgan.upscale(image)
    image.save("output_image.jpg")
    ```

- **[FastBlend](https://arxiv.org/abs/2311.09265)**: FastBlend is a model-free video de-flickering algorithm. Flicker often occurs in style videos processed frame by frame using image generation models. FastBlend can eliminate flicker in style videos based on the motion features in the original video (guide video).

    In this code snippet, we use FastBlend to remove the flicker effect from a style video.

    ```python
    from diffsynth import VideoData, save_video
    from diffsynth.extensions.FastBlend import FastBlendSmoother

    fastblend = FastBlendSmoother()
    guide_video = VideoData("guide_video.mp4", height=512, width=768).raw_data()
    style_video = VideoData("style_video.mp4", height=512, width=768).raw_data()
    output_video = fastblend(style_video, original_frames=guide_video)
    save_video(output_video, "output_video.mp4", fps=30)
    ```
