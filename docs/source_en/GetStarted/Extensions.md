# Extensions

This document introduces some relevant techniques beyond the diffusion models implemented in DiffSynth, which have significant application potential in image and video processing.

- **[RIFE](https://github.com/hzwer/ECCV2022-RIFE)**: FIRE (Real-Time Intermediate Flow Estimation Algorithm) is a frame interpolation (VFI) method based on real-time intermediate flow estimation. It includes an end-to-end efficient intermediate flow estimation network called IFNet, as well as an optical flow supervision framework based on privileged distillation. RIFE supports inserting frames at any moment between two frames, achieving state-of-the-art performance across multiple datasets without relying on any pre-trained models.

- **[ESRGAN](https://github.com/xinntao/ESRGAN)**: ESRGAN (Enhanced Super Resolution Generative Adversarial Network) is an improved method based on SRGAN, aimed at enhancing the visual quality of single image super-resolution. This approach significantly improves the realism of generated images by optimizing three key components of SRGAN: network architecture, adversarial loss, and perceptual loss.

- **[FastBlend](https://arxiv.org/abs/2311.09265)**: FastBlend is a model-free toolkit designed for smoothing videos, integrated with Diffusion models to create a powerful video processing workflow. This tool effectively eliminates flickering in videos, performs interpolation on keyframe sequences, and can process complete videos based on a single image.

