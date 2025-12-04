# Building a Pipeline

After [integrating the required models for the Pipeline](/docs/en/Developer_Guide/Integrating_Your_Model.md), you also need to build a `Pipeline` for model inference. This document provides a standardized process for building a `Pipeline`. Developers can also refer to existing `Pipeline` implementations for construction.

The `Pipeline` implementation is located in `diffsynth/pipelines`. Each `Pipeline` contains the following essential key components:

* `__init__`
* `from_pretrained`
* `__call__`
* `units`
* `model_fn`

## `__init__`

In `__init__`, the `Pipeline` is initialized. Here is a simple implementation:

```python
import torch
from PIL import Image
from typing import Union
from tqdm import tqdm
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.new_models import XXX_Model, YYY_Model, ZZZ_Model

class NewDiffSynthPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler()
        self.text_encoder: XXX_Model = None
        self.dit: YYY_Model = None
        self.vae: ZZZ_Model = None
        self.in_iteration_models = ("dit",)
        self.units = [
            NewDiffSynthPipelineUnit_xxx(),
            ...
        ]
        self.model_fn = model_fn_new
```

This includes the following parts:

* `scheduler`: Scheduler, used to control the coefficients in the iterative formula during inference, controlling the noise content at each step.
* `text_encoder`, `dit`, `vae`: Models. Since [Latent Diffusion](https://arxiv.org/abs/2112.10752) was proposed, this three-stage model architecture has become the mainstream Diffusion model architecture. However, this is not immutable, and any number of models can be added to the `Pipeline`.
* `in_iteration_models`: Iteration models. This tuple marks which models will be called during iteration.
* `units`: Pre-processing units for model iteration. See [`units`](#units) for details.
* `model_fn`: The `forward` function of the denoising model during iteration. See [`model_fn`](#model_fn) for details.

> Q: Model loading does not occur in `__init__`, why initialize each model as `None` here?
> 
> A: By annotating the type of each model here, the code editor can provide code completion prompts based on each model, facilitating subsequent development.

## `from_pretrained`

`from_pretrained` is responsible for loading the required models to make the `Pipeline` callable. Here is a simple implementation:

```python
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = NewDiffSynthPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("xxx_text_encoder")
        pipe.dit = model_pool.fetch_model("yyy_dit")
        pipe.vae = model_pool.fetch_model("zzz_vae")
        # If necessary, load tokenizers here.
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
```

Developers need to implement the logic for fetching models. The corresponding model names are the `"model_name"` in the [model Config filled in during model integration](/docs/en/Developer_Guide/Integrating_Your_Model.md#step-3-writing-model-config).

Some models also need to load `tokenizer`. Extra `tokenizer_config` parameters can be added to `from_pretrained` as needed, and this part can be implemented after fetching the models.

## `__call__`

`__call__` implements the entire generation process of the Pipeline. Below is a common generation process template. Developers can modify it based on their needs.

```python
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        height: int = 1328,
        width: int = 1328,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 30,
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength
        )
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "denoising_strength": denoising_strength,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image
```

## `units`

`units` contains all the preprocessing processes, such as: width/height checking, prompt encoding, initial noise generation, etc. In the entire model preprocessing process, data is abstracted into three mutually exclusive parts, stored in corresponding dictionaries:

* `inputs_shared`: Shared inputs, parameters unrelated to [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) (CFG for short).
* `inputs_posi`: Positive side inputs for Classifier-Free Guidance, containing content related to positive prompts.
* `inputs_nega`: Negative side inputs for Classifier-Free Guidance, containing content related to negative prompts.

Pipeline Unit implementations include three types: direct mode, CFG separation mode, and takeover mode.

If some calculations are unrelated to CFG, direct mode can be used, for example, Qwen-Image's random noise initialization:

```python
class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}
```

If some calculations are related to CFG and need to separately process positive and negative prompts, but the input parameters on both sides are the same, CFG separation mode can be used, for example, Qwen-image's prompt encoding:

```python
class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        # Do something
        return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
```

If some calculations need global information, takeover mode is required, for example, Qwen-Image's entity partition control:

```python
class QwenImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("eligen_entity_prompts", "width", "height", "eligen_enable_on_negative", "cfg_scale"),
            output_params=("entity_prompt_emb", "entity_masks", "entity_prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: QwenImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        # Do something
        return inputs_shared, inputs_posi, inputs_nega
```

The following are the parameter configurations required for Pipeline Unit:

* `seperate_cfg`: Whether to enable CFG separation mode
* `take_over`: Whether to enable takeover mode
* `input_params`: Shared input parameters
* `output_params`: Output parameters
* `input_params_posi`: Positive side input parameters
* `input_params_nega`: Negative side input parameters
* `onload_model_names`: Names of model components to be called

When designing `unit`, please try to follow these principles:

* Default fallback: For optional function `unit` input parameters, the default is `None` rather than `False` or other values. Please provide fallback processing for this default value.
* Parameter triggering: Some Adapter models may not be loaded, such as ControlNet. The corresponding `unit` should control triggering based on whether the parameter input is `None` rather than whether the model is loaded. For example, when the user inputs `controlnet_image` but does not load the ControlNet model, the code should give an error rather than ignore these input parameters and continue execution.
* Simplicity first: Use direct mode as much as possible, only use takeover mode when the function cannot be implemented.
* VRAM efficiency: When calling models in `unit`, please use `pipe.load_models_to_device(self.onload_model_names)` to activate the corresponding models. Do not call other models outside `onload_model_names`. After `unit` calculation is completed, do not manually release VRAM with `pipe.load_models_to_device([])`.

> Q: Some parameters are not called during the inference process, such as `output_params`. Is it still necessary to configure them?
> 
> A: These parameters will not affect the inference process, but they will affect some experimental features. Therefore, we recommend configuring them properly. For example, "split training" - we can complete the preprocessing offline during training, but some model calculations that require gradient backpropagation cannot be split. These parameters are used to build computational graphs to infer which calculations can be split.

## `model_fn`

`model_fn` is the unified `forward` interface during iteration. For models where the open-source ecosystem is not yet formed, you can directly use the denoising model's `forward`, for example:

```python
def model_fn_new(dit=None, latents=None, timestep=None, prompt_emb=None, **kwargs):
    return dit(latents, prompt_emb, timestep)
```

For models with rich open-source ecosystems, `model_fn` usually contains complex and chaotic cross-model inference. Taking `diffsynth/pipelines/qwen_image.py` as an example, the additional calculations implemented in this function include: entity partition control, three types of ControlNet, Gradient Checkpointing, etc. Developers need to be extra careful when implementing this part to avoid conflicts between module functions.