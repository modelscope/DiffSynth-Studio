# 接入 Pipeline

在[将 Pipeline 所需的模型接入](/docs/zh/Developer_Guide/Integrating_Your_Model.md)之后，还需构建 `Pipeline` 用于模型推理，本文档提供 `Pipeline` 构建的标准化流程，开发者也可参考现有的 `Pipeline` 进行构建。

`Pipeline` 的实现位于 `diffsynth/pipelines`，每个 `Pipeline` 包含以下必要的关键组件：

* `__init__`
* `from_pretrained`
* `__call__`
* `units`
* `model_fn`

## `__init__`

在 `__init__` 中，`Pipeline` 进行初始化，以下是一个简易的实现：

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

其中包括以下几部分

* `scheduler`: 调度器，用于控制推理的迭代公式中的系数，控制每一步的噪声含量。
* `text_encoder`、`dit`、`vae`: 模型，自 [Latent Diffusion](https://arxiv.org/abs/2112.10752) 被提出以来，这种三段式模型架构已成为主流的 Diffusion 模型架构，但这并不是一成不变的，`Pipeline` 中可添加任意多个模型。
* `in_iteration_models`: 迭代中模型，这个元组标注了在迭代中会调用哪些模型。
* `units`: 模型迭代的前处理单元，详见[`units`](#units)。
* `model_fn`: 迭代中去噪模型的 `forward` 函数，详见[`model_fn`](#model_fn)。

> Q: 模型加载并不发生在 `__init__`，为什么这里仍要将每个模型初始化为 `None`？
> 
> A: 在这里标注每个模型的类型后，代码编辑器就可以根据每个模型提供代码补全提示，便于后续的开发。

## `from_pretrained`

`from_pretrained` 负责加载所需的模型，让 `Pipeline` 变成可调用的状态。以下是一个简易的实现：

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

开发者需要实现其中获取模型的逻辑，对应的模型名称即为[模型接入时填写的模型 Config](/docs/zh/Developer_Guide/Integrating_Your_Model.md#step-3-编写模型-config) 中的 `"model_name"`。

部分模型还需要加载 `tokenizer`，可根据需要在 `from_pretrained` 上添加额外的 `tokenizer_config` 参数并在获取模型后实现这部分。

## `__call__`

`__call__` 实现了整个 Pipeline 的生成过程，以下是常见的生成过程模板，开发者可根据需要在此基础上修改。

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

`units` 包含了所有的前处理过程，例如：宽高检查、提示词编码、初始噪声生成等。在整个模型前处理过程中，数据被抽象为了互斥的三部分，分别存储在对应的字典中：

* `inputs_shard`: 共享输入，与 [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)（简称 CFG）无关的参数。
* `inputs_posi`: Classifier-Free Guidance 的 Positive 侧输入，包含与正向提示词相关的内容。
* `inputs_nega`: Classifier-Free Guidance 的 Negative 侧输入，包含与负向提示词相关的内容。

Pipeline Unit 的实现包括三种：直接模式、CFG 分离模式、接管模式。

如果某些计算与 CFG 无关，可采用直接模式，例如 Qwen-Image 的随机噪声初始化：

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

如果某些计算与 CFG 有关，需分别处理正向和负向提示词，但两侧的输入参数是相同的，可采用 CFG 分离模式，例如 Qwen-image 的提示词编码：

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

如果某些计算需要全局的信息，则需要接管模式，例如 Qwen-Image 的实体分区控制：

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

以下是 Pipeline Unit 所需的参数配置：

* `seperate_cfg`: 是否启用 CFG 分离模式
* `take_over`: 是否启用接管模式
* `input_params`: 共享输入参数
* `output_params`: 输出参数
* `input_params_posi`: Positive 侧输入参数
* `input_params_nega`: Negative 侧输入参数
* `onload_model_names`: 需调用的模型组件名

在设计 `unit` 时请尽量按照以下原则进行：

* 缺省兜底：可选功能的 `unit` 输入参数默认为 `None`，而不是 `False` 或其他数值，请对此默认值进行兜底处理。
* 参数触发：部分 Adapter 模型可能是未被加载的，例如 ControlNet，对应的 `unit` 应当以参数输入是否为 `None` 来控制触发，而不是以模型是否被加载来控制触发。例如当用户输入了 `controlnet_image` 但没有加载 ControlNet 模型时，代码应当给出报错，而不是忽略这些输入参数继续执行。
* 简洁优先：尽可能使用直接模式，仅当功能无法实现时，使用接管模式。
* 显存高效：在 `unit` 中调用模型时，请使用 `pipe.load_models_to_device(self.onload_model_names)` 激活对应的模型，请不要调用 `onload_model_names` 之外的其他模型，`unit` 计算完成后，请不要使用 `pipe.load_models_to_device([])` 手动释放显存。

> Q: 部分参数并未在推理过程中调用，例如 `output_params`，是否仍有必要配置？
> 
> A: 这些参数不会影响推理过程，但会影响一些实验性功能，因此我们建议将其配置好。例如“拆分训练”，我们可以将训练中的前处理离线完成，但部分需要梯度回传的模型计算无法拆分，这些参数用于构建计算图从而推断哪些计算是可以拆分的。

## `model_fn`

`model_fn` 是迭代中的统一 `forward` 接口，对于开源模型生态尚未形成的模型，直接沿用去噪模型的 `forward` 即可，例如：

```python
def model_fn_new(dit=None, latents=None, timestep=None, prompt_emb=None, **kwargs):
    return dit(latents, prompt_emb, timestep)
```

对于开源生态丰富的模型，`model_fn` 通常包含复杂且混乱的跨模型推理，以 `diffsynth/pipelines/qwen_image.py` 为例，这个函数中实现的额外计算包括：实体分区控制、三种 ControlNet、Gradient Checkpointing 等，开发者在实现这一部分时要格外小心，避免模块功能之间的冲突。
