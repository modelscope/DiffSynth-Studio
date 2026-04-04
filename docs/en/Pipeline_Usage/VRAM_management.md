# VRAM Management

VRAM management is a distinctive feature of `DiffSynth-Studio` that enables GPUs with low VRAM to run inference with large parameter models. This document uses Qwen-Image as an example to introduce how to use the VRAM management solution.

## Basic Inference

The following code does not enable any VRAM management, occupying 56G VRAM as a reference.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## CPU Offload

Since the model `Pipeline` consists of multiple components that are not called simultaneously, we can move some components to memory when they are not needed for computation, reducing VRAM usage. The following code implements this logic, occupying 40G VRAM.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## FP8 Quantization

Building upon CPU Offload, we further enable FP8 quantization to reduce VRAM requirements. The following code allows model parameters to be stored in VRAM with FP8 precision and temporarily converted to BF16 precision for computation during inference, occupying 21G VRAM. However, this quantization scheme has minor image quality degradation issues.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cuda",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

> Q: Why temporarily convert to BF16 precision during inference instead of computing with FP8 precision?
> 
> A: Native FP8 computation is only supported on Hopper architecture GPUs (such as H20) and has significant computational errors. We currently do not enable FP8 precision computation. The current FP8 quantization only reduces VRAM usage but does not improve computation speed.

## Dynamic VRAM Management

In CPU Offload, we control model components. In fact, we support Layer-level Offload, splitting a model into multiple Layers, keeping some resident in VRAM and storing others in memory for on-demand transfer to VRAM for computation. This feature requires model developers to provide detailed VRAM management solutions for each model. Related configurations are in `diffsynth/configs/vram_management_module_maps.py`.

By adding the `vram_limit` parameter to the `Pipeline`, the framework can automatically sense the remaining VRAM of the device and decide how to split the model between VRAM and memory. The smaller the `vram_limit`, the less VRAM occupied, but slower the speed.
* When `vram_limit=None`, the default state, the framework assumes unlimited VRAM and dynamic VRAM management is disabled
* When `vram_limit=10`, the framework will limit the model after VRAM usage exceeds 10G, moving the excess parts to memory storage
* When `vram_limit=0`, the framework will do its best to reduce VRAM usage, storing all model parameters in memory and transferring them to VRAM for computation only when necessary

When VRAM is insufficient to run model inference, the framework will attempt to exceed the `vram_limit` restriction to keep the model inference running. Therefore, the VRAM management framework cannot always guarantee that VRAM usage will be less than `vram_limit`. We recommend setting it to slightly less than the actual available VRAM. For example, when GPU VRAM is 16G, set it to `vram_limit=15.5`. In `PyTorch`, you can use `torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)` to get the GPU's VRAM.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## Disk Offload

In more extreme cases, when memory is also insufficient to store the entire model, the Disk Offload feature allows lazy loading of model parameters, meaning each Layer of the model only reads the corresponding parameters from disk when the forward function is called. When enabling this feature, we recommend using high-speed SSD drives.

Disk Offload is a very special VRAM management solution that only supports `.safetensors` format files, not `.bin`, `.pth`, `.ckpt`, or other binary files, and does not support [state dict converter](/docs/en/Developer_Guide/Integrating_Your_Model.md#step-2-model-file-format-conversion) with Tensor reshape.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=10,
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## More Usage Methods

Information in `vram_config` can be filled in manually, for example, Disk Offload without FP8 quantization:

```python
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
```

Specifically, the VRAM management module divides model Layers into the following four states:

* Offload: This model will not be called in the short term. This state is controlled by switching `Pipeline`
* Onload: This model will be called at any time soon. This state is controlled by switching `Pipeline`
* Preparing: Intermediate state between Onload and Computation. A temporary storage state when VRAM allows. This state is controlled by the VRAM management mechanism and enters this state if and only if [vram_limit is set to unlimited] or [vram_limit is set and there is spare VRAM]
* Computation: The model is being computed. This state is controlled by the VRAM management mechanism and is temporarily entered only during `forward`

If you are a model developer and want to control the VRAM management granularity of a specific model, please refer to [../Developer_Guide/Enabling_VRAM_management.md](/docs/en/Developer_Guide/Enabling_VRAM_management.md).

## Best Practices

* Sufficient VRAM -> Use [Basic Inference](#basic-inference)
* Insufficient VRAM
    * Sufficient memory -> Use [Dynamic VRAM Management](#dynamic-vram-management)
    * Insufficient memory -> Use [Disk Offload](#disk-offload)