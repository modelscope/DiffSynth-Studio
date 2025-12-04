# 接入模型训练

在[接入模型](/docs/zh/Developer_Guide/Integrating_Your_Model.md)并[实现 Pipeline](/docs/zh/Developer_Guide/Building_a_Pipeline.md)后，接下来接入模型训练功能。

## 训推一致的 Pipeline 改造

为了保证训练和推理过程严格的一致性，我们会在训练过程中沿用大部分推理代码，但仍需作出少量改造。

首先，在推理过程中添加额外的逻辑，让图生图/视频生视频逻辑根据 `scheduler` 状态进行切换。以 Qwen-Image 为例：

```python
class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}
```

然后，在 `model_fn` 中启用 Gradient Checkpointing，这将以计算速度为代价，大幅度减少训练所需的显存。这并不是必需的，但我们强烈建议这么做。

以 Qwen-Image 为例，修改前：

```python
text, image = block(
    image=image,
    text=text,
    temb=conditioning,
    image_rotary_emb=image_rotary_emb,
    attention_mask=attention_mask,
)
```

修改后：

```python
from ..core import gradient_checkpoint_forward

text, image = gradient_checkpoint_forward(
    block,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    image=image,
    text=text,
    temb=conditioning,
    image_rotary_emb=image_rotary_emb,
    attention_mask=attention_mask,
)
```

## 编写训练脚本

`DiffSynth-Studio` 没有对训练框架做严格的封装，而是将脚本内容暴露给开发者，这种方式可以更方便地对训练脚本进行修改，实现额外的功能。开发者可参考现有的训练脚本，例如 `examples/qwen_image/model_training/train.py` 进行修改，从而适配新的模型训练。
