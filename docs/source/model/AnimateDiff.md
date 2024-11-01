# AnimateDiff

## 相关链接

* 论文：
    * [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)
* 模型
    * AnimateDiff
        * [HuggingFace](https://huggingface.co/guoyww/animatediff)
        * [ModelScope](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/animatediff)

## 模型介绍

AnimateDiff 是一种文生图模型的扩展方法，可以将文生图模型扩展为动画生成器，而无需对文生图模型做任何微调。扩展的基本思路是从大型视频数据集中学习到运动先验知识并保存到运动模块中，使用时将运动模块插入文生图模型即可。以下为其生成的视频效果：

<div align="center">
<video width="256" height="256" controls>
  <source src="https://github.com/user-attachments/assets/d5c22c05-ddb3-4b05-982a-1e65dd19b1ef" type="video/mp4">
您的浏览器不支持Video标签。
</video>
</div>


AnnimateDiff 的训练主要分为三个阶段，分别对应了三个可训练的模块：Domain Adapter，Motion Module 和 MotionLoRA，如下图所示。

![](https://github.com/user-attachments/assets/a788caf8-9cc8-45bb-ba20-d80684d80e08)

第一阶段中主要训练 Domain Adapter。由于公开可用的视频训练数据集的质量远低于图像数据集的质量，直接从这种数据集上训练 Motion Module 可能就降低其视频生成质量。视频和图像数据集质量的差距被成为域差距。为了减小这一差距对 Motion Module 的影响，作者提出使用 Domain Adapter 来单独拟合这些域差距。Domain Adapter 具体通过LoRA来实现，即在文生图模型中的 Self/Cross-Attention 层中插入 LoRA 模块。以 Query Projection 为例，插入 LoRA 后的输出如下公式所示。其中，$\alpha$ 为 Domain Adapter 权重。在推理的时候，设置 $\alpha=0$ 以去除 Domain Adapter 的影响。

$$
Q=\mathcal{W}^Q z+\text { AdapterLayer }(z)=\mathcal{W}^Q z+\alpha \cdot A B^T z
$$

第二阶段主要训练 Motion Module，这一模块主要目的是学习视频的运动先验信息。如上图所示， Motion Module 主要结构为 Temporal Transformer，由输入输出映射层和若干个 Self-Attention 组成。将 Motion Module 插入文生图模型后，模型的输入维度为：$b\times c\times f \times h \times w$。在数据到达文生图模型的原始模块（上图白色）时，将帧数 $f$ 融合到 $b$ 维度上，即可完成正常计算。当数据到达 Motion Module 时，为了完成 Temporal Attention，又将 $h$ 和 $w$ 融合到 $b$ 维度上，数据维度变为： $\{b\cdot h\cdot w\} \times f \times c$。

尽管第二阶段训练的 Motion Module 学习了通用的运动先验知识，但仍然需要有效地将其适应到特定运动模式，比如相机缩放、平移等。因此，第三阶段主要针对个性化运动训练对应的 MotionLoRA。MotionLoRA 主要是通过在 Motion Module 的 Attention 中加入LoRA中实现的。实验证明，20 ~ 50 个参考视频、2000 个 step 就能学习到对应的运动能力。同时，多个 MotionLoRA 的运动效果是可以组合的。

## 代码样例

```python
from diffsynth import ModelManager, SDXLVideoPipeline, save_video, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/sd_xl_base_1.0.safetensors`: [link](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors)
# `models/AnimateDiff/mm_sdxl_v10_beta.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt)
download_models(["StableDiffusionXL_v1", "AnimateDiff_xl_beta"])

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/sd_xl_base_1.0.safetensors",
    "models/AnimateDiff/mm_sdxl_v10_beta.ckpt"
])
pipe = SDXLVideoPipeline.from_model_manager(model_manager)

prompt = "A panda standing on a surfboard in the ocean in sunset, 4k, high resolution.Realistic, Cinematic, high resolution"
negative_prompt = ""

torch.manual_seed(0)
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=8.5,
    height=1024, width=1024, num_frames=16,
    num_inference_steps=100,
)
save_video(video, "output_video.mp4", fps=16)
```