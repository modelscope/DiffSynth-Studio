# Integrating Model Training

After [integrating models](/docs/en/Developer_Guide/Integrating_Your_Model.md) and [implementing Pipeline](/docs/en/Developer_Guide/Building_a_Pipeline.md), the next step is to integrate model training functionality.

## Training-Inference Consistent Pipeline Modification

To ensure strict consistency between training and inference processes, we will use most of the inference code during training, but still need to make minor modifications.

First, add extra logic during inference to switch the image-to-image/video-to-video logic based on the `scheduler` state. Taking Qwen-Image as an example:

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

Then, enable Gradient Checkpointing in `model_fn`, which will significantly reduce the VRAM required for training at the cost of computational speed. This is not mandatory, but we strongly recommend doing so.

Taking Qwen-Image as an example, before modification:

```python
text, image = block(
    image=image,
    text=text,
    temb=conditioning,
    image_rotary_emb=image_rotary_emb,
    attention_mask=attention_mask,
)
```

After modification:

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

## Writing Training Scripts

`DiffSynth-Studio` does not strictly encapsulate the training framework, but exposes the script content to developers. This approach makes it more convenient to modify training scripts to implement additional functions. Developers can refer to existing training scripts, such as `examples/qwen_image/model_training/train.py`, for modification to adapt to new model training.