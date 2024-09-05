
# A Simple Example: Text-to-Image Synthesis with Flux

The following example shows how to use the FLUX.1 model for text-to-image tasks. The script provides a simple setup for generating images from text descriptions. It covers downloading the necessary models, configuring the pipeline, and generating images with and without classifier-free guidance.

For other models supported by DiffSynth, see [Models.md](Models.md).

## Setup

First, ensure you have the necessary models downloaded and configured:

```python
import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models

# Download the FLUX.1-dev model files
download_models(["FLUX.1-dev"])
```

For instructions on downloading models, see [Download_models.md](Download_models.md).

## Loading Models
Initialize the model manager with your device and data type:

```python
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
```

For instructions on loading models, see [ModelManager.md](ModelManager.md).

## Creating the Pipeline
Create an instance of the FluxImagePipeline from the loaded model manager:


```python
pipe = FluxImagePipeline.from_model_manager(model_manager)
```

For instructions on using the Pipeline, see [Pipeline.md](Pipeline.md).
## Text-to-Image Synthesis
Generate an image using a short prompt. Below are examples of generating images with and without classifier-free guidance.

### Basic Generation
```python
prompt = "A cute little turtle"
negative_prompt = ""

torch.manual_seed(6)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_1024.jpg")
```

### Generation with Classifier-Free Guidance
```python
torch.manual_seed(6)
image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    num_inference_steps=30, cfg_scale=2.0, embedded_guidance=3.5
)
image.save("image_1024_cfg.jpg")
```

### High-Resolution Fix
```python
torch.manual_seed(7)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5,
    input_image=image.resize((2048, 2048)), height=2048, width=2048, denoising_strength=0.6, tiled=True
)
image.save("image_2048_highres.jpg")
```

