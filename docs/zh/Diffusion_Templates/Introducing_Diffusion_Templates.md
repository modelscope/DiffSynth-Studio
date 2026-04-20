# Diffusion Templates

Diffusion Templates 是 DiffSynth-Studio 中的 Diffusion 模型可控生成插件框架，可以为基础模型提供额外的可控生成能力。

* 开源代码：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* 技术报告：coming soon
* 文档参考
    * Diffusion Templates 简介：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Introducing_Diffusion_Templates.html)
    * Diffusion Templates 架构详解：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Understanding_Diffusion_Templates.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Understanding_Diffusion_Templates.html)
    * Template 模型推理：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Template_Model_Inference.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Template_Model_Inference.html)
    * Template 模型训练：[English Version](https://diffsynth-studio-doc.readthedocs.io/en/latest/Diffusion_Templates/Template_Model_Training.html)、[中文版](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Diffusion_Templates/Template_Model_Training.html)
* 在线体验：[魔搭社区创空间](https://modelscope.cn/studios/DiffSynth-Studio/Diffusion-Templates)
* 模型：[合集](https://modelscope.cn/collections/DiffSynth-Studio/KleinBase4B-Templates)
    * 结构控制：[DiffSynth-Studio/Template-KleinBase4B-ControlNet](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ControlNet)
    * 亮度调节：[DiffSynth-Studio/Template-KleinBase4B-Brightness](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Brightness)
    * 色彩调节：[DiffSynth-Studio/Template-KleinBase4B-SoftRGB](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-SoftRGB)
    * 图像编辑：[DiffSynth-Studio/Template-KleinBase4B-Edit](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Edit)
    * 超分辨率：[DiffSynth-Studio/Template-KleinBase4B-Upscaler](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Upscaler)
    * 锐利激发：[DiffSynth-Studio/Template-KleinBase4B-Sharpness](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Sharpness)
    * 美学对齐：[DiffSynth-Studio/Template-KleinBase4B-Aesthetic](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Aesthetic)
    * 局部重绘：[DiffSynth-Studio/Template-KleinBase4B-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-Inpaint)
    * 内容参考：[DiffSynth-Studio/Template-KleinBase4B-ContentRef](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-ContentRef)
    * 魔性熊猫（彩蛋模型）：[DiffSynth-Studio/Template-KleinBase4B-PandaMeme](https://modelscope.cn/models/DiffSynth-Studio/Template-KleinBase4B-PandaMeme)
* 数据集：[合集](https://modelscope.cn/collections/DiffSynth-Studio/ImagePulseV2--shujuji)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Inpaint](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Inpaint)
    * [DiffSynth-Studio/ImagePulseV2-TextImage](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-TextImage)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Background](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Background)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Clothes](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Clothes)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Pose](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Pose)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Change](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Change)
    * [DiffSynth-Studio/ImagePulseV2-Edit-AddRemove](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-AddRemove)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Upscale](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Upscale)
    * [DiffSynth-Studio/ImagePulseV2-TextImage-Human](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-TextImage-Human)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Crop](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Crop)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Light](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Light)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Structure](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Structure)
    * [DiffSynth-Studio/ImagePulseV2-Edit-HumanFace](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-HumanFace)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Angle](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Angle)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Style](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Style)
    * [DiffSynth-Studio/ImagePulseV2-TextImage-MultiResolution](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-TextImage-MultiResolution)
    * [DiffSynth-Studio/ImagePulseV2-Edit-Merge](https://modelscope.cn/datasets/DiffSynth-Studio/ImagePulseV2-Edit-Merge)

## 模型效果一览

* 超分辨率 + 锐利激发：生成清晰度极高的图像

|低清晰度输入|高清晰度输出|
|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_lowres_100.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Upscaler_Sharpness.png)|

* 结构控制 + 美学对齐 + 锐利激发：全副武装的 ControlNet

|结构控制图|输出图|
|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_depth.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Controlnet_Aesthetic_Sharpness.png)|

* 结构控制 + 图像编辑 + 色彩调节：随心所欲的艺术风格创作

|结构控制图|编辑输入图|输出图|
|-|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_depth.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_reference.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Controlnet_Edit_SoftRGB.png)|

* 亮度控制 + 图像编辑 + 局部重绘：让图中的部分元素跨越次元

|参考图|重绘区域|输出图|
|-|-|-|
|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_reference.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_mask_1.jpg)|![](https://modelscope.cn/datasets/DiffSynth-Studio/examples_in_diffsynth/resolve/master/templates/image_Brightness_Edit_Inpaint.png)|

