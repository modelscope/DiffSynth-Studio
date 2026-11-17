# DiffSynth Pipeline 模板与架构

> **权威定义**：本文档是 DiffSynth Pipeline 架构的唯一来源。所有 diffsynth-* skill 中涉及 Pipeline 设计的内容均以此为准。

## 核心原则

**Pipeline 是编排者，不是模型实现者。** Pipeline 负责调用独立模型完成完整流程，不包含任何模型的前向传播逻辑。

文件位置：`{diffsynth_root}/diffsynth/pipelines/{series}.py`

**一个模型系列对应一个 Pipeline 文件**，所有功能通过 `__call__` 的不同参数和 `units` 链来实现。

```python
"""
{Series Name} Pipeline for DiffSynth-Studio.
"""
import torch
from typing import Optional
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler  # 所有调度策略统一使用 FlowMatchScheduler
# Scheduler 接入规则详见 [scheduler-integration.md](./scheduler-integration.md)
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.{series}_dit import {DiTClass}
from ..models.{series}_vae import {VAEClass}
# ... 其他模型 import


class {SeriesName}Pipeline(BasePipeline):
    """Pipeline for {series description}."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,  # 根据模型需求调整
            width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("{Series}")  # template 名称从蓝图报告获取
        self.text_encoder: {TextEncoderClass} = None
        self.dit: {DiTClass} = None
        self.vae: {VAEClass} = None
        # ... 其他组件

        self.in_iteration_models = ("dit",)  # denoise loop 中需要的模型
        self.units = [
            # PipelineUnit 链，根据功能需求定义
            {SeriesName}Unit_ShapeChecker(),
            {SeriesName}Unit_PromptEmbedder(),
            {SeriesName}Unit_NoiseInitializer(),
            # ... 其他 unit，每个功能对应一个或多个 unit
        ]
        self.model_fn = model_fn_{series}
        self.compilable_models = ["dit"]  # 支持 torch.compile 的 denoiser 属性名，多 denoiser 时全部列出

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        """Load pipeline from pretrained checkpoints."""
        pipe = {SeriesName}Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("{series}_text_encoder")
        pipe.dit = model_pool.fetch_model("{series}_dit")
        pipe.vae = model_pool.fetch_model("{series}_vae")
        # ... fetch 其他组件
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Image
        input_image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Scheduler-specific parameters (if any, based on model series)
        # Example for LTX-2:
        # dynamic_shift_len: int = 4096,
        # terminal: float = 0.1,
        # 功能特定参数...
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        # 基础参数：所有模型都传
        scheduler_kwargs = {
            "num_inference_steps": num_inference_steps,
            "denoising_strength": denoising_strength,
        }
        # 模型特异参数：根据系列添加
        # scheduler_kwargs["dynamic_shift_len"] = dynamic_shift_len
        # scheduler_kwargs["terminal"] = terminal
        self.scheduler.set_timesteps(**scheduler_kwargs)
        
        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            # ... 其他功能参数
        }

        # Run pipeline units to prepare inputs
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class {SeriesName}Unit_ShapeChecker(PipelineUnit):
    """Check and adjust input/output shapes."""
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


# ... 其他 PipelineUnit，每个 Unit 处理一个方面


def model_fn_{series}(
    dit: {DiTClass},
    latents=None,
    timestep=None,
    prompt_embeds=None,
    **kwargs,
):
    """Model function called during denoising loop."""
    timestep = timestep / 1000.
    output = dit(latents, prompt_embeds, timestep)
    return output
```

## PipelineUnit 设计原则

- **每个 Unit 只做一件事**：shape 检查、prompt 编码、noise 初始化、ControlNet 输入处理等
- **通过 `input_params` 和 `output_params` 明确数据流**：让调用者清楚知道需要什么、产出什么
- **使用 `seperate_cfg=True` 处理 positive/negative prompt 分离的场景**
- **`onload_model_names` 指定该 Unit 需要加载到 GPU 的模型**，便于 VRAM 管理
- **Unit 的 `process` 方法应检查参数是否为 None**：如果需要的参数为 None，返回空 dict `{}` 跳过该 Unit 的处理

### 三大核心 Unit 概要

每个 Pipeline 至少包含以下 4 个核心 Unit，按固定顺序排列：

```
ShapeChecker → PromptEmbedder → NoiseInitializer → InputEmbedder → [功能专属 Unit...]
```

| Unit | 核心约束 |
|------|---------|
| **ShapeChecker** | 必须复用 `pipe.check_resize_height_width`，必须放在 Unit 链最前面 |
| **PromptEmbedder** | 必须用 `seperate_cfg=True`，从目标库的 `encode_prompt` 移植编码逻辑 |
| **NoiseInitializer** | 必须调用 `pipe.generate_noise(...)`，输出 key 固定为 `"noise"` |
| **InputEmbedder** | 推理时：返回 `{"latents": add_noise(...)}`（不返回 input_latents）；训练时：返回 `{"latents": noise, "input_latents": vae_encode(数据)}` |

详细设计规范（含代码模板和完整约束说明）见 [SKILL.md](../SKILL.md) 中的「四大核心 Unit 设计规范」章节。

### Embedder 类型区分

**`InputXxxEmbedder`（训练/推理共用）**：详见 [SKILL.md](../SKILL.md) 「四大核心 Unit 设计规范」中的 InputEmbedder 详细规范。

**`{Feature}Embedder`（功能专用）**：
- 图像编辑、局部重采样、分层编辑等功能，**必须编写独立的 Embedder Unit**
- 示例：`EditImageEmbedder`（编辑图像编码）、`InpaintEmbedder`（mask 处理）、`LayerInputImageEmbedder`（分层图像）、`ContextImageEmbedder`（上下文图像）
- 参考 Qwen-Image 的 `EditImageEmbedder`：接受 `edit_image` + `edit_image_auto_resize`，通过 VAE 编码得到 `edit_latents`
- LTX-2 的 `VideoRetakeEmbedder`、`AudioRetakeEmbedder`、`InContextVideoEmbedder` 同样遵循此模式

**不要将编辑功能的输入混入 `InputImageEmbedder`**。每个编辑功能应有自己的 Embedder，命名格式为 `{FeatureName}Embedder`。

- **参考实际 Pipeline**：`z_image.py`（689 行，8 个 Unit）、`qwen_image.py`（818 行，10 个 Unit）

## 不可违背的设计原则

| 原则 | 说明 |
|------|------|
| **Batch Size 恒为 1** | 整个框架中 batch size 永远为 1，不暴露 batch_size 参数，不编写多 batch 推理逻辑 |
| **正向/负向分离 Forward** | 永远不要将正向（positive）和负向（negative）的 forward 放到一个 batch 中运行。正向和负向分别通过 `inputs_posi` / `inputs_nega` 独立传递，各自单独调用模型 forward |
| **Denoising Loop 最小修改** | 模板中的 denoising loop（`cfg_guided_model_fn` + `scheduler.step` 循环）是框架的核心逻辑，除非有充分理由（如目标库推理流程明确要求不同的采样策略），否则不得修改其结构和执行顺序。新增功能应通过添加 Unit 或扩展 `model_fn` 参数来实现，而不是改动循环本身 |

## 推理示例脚本

文件位置：`{diffsynth_root}/examples/{series}/model_inference/{feature}.py`

通常 15-20 行：

```python
import os
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="...", origin_file_pattern="tokenizer/"),
)
prompt = "..."
output = pipe(prompt=prompt, seed=42)
output.save("output.jpg")
```

**注意**：模型路径通过 `ModelConfig` 指定，模型首次运行时会自动从 ModelScope 下载并创建软链接。

## 低显存推理示例

文件位置：`{diffsynth_root}/examples/{series}/model_inference_low_vram/{feature}.py`

与标准版本的区别：`from_pretrained` 中传入 `vram_limit` 参数：

```python
pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[...],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,  # 留 0.5G 余量
)
```

## 多功能接入规则

**每次只接入一个功能。** 按从简单到难的顺序，逐次调用 `diffsynth-pipeline` skill。

**Pipeline 文件按系列组织，不按功能拆分。** 一个模型系列对应一个 Pipeline 文件，所有功能通过 `__call__` 的不同参数和 `units` 链的不同组合来实现。

- **首次调用**：创建 Pipeline 文件（包含基础功能：ShapeChecker、PromptEmbedder、NoiseInitializer、Denoise loop、VAE decode）
- **后续调用**：
  1. 在 `__call__` 中添加该功能的新参数
  2. 在 `inputs_shared` 中添加该功能的参数
  3. 在 `units` 链中添加该功能对应的 Unit
  4. 在 `model_fn` 中添加该功能需要的参数（如有）
  5. 确保不改变已有功能的参数默认值和逻辑

## 各 Skill 中的 Pipeline 相关职责

| Skill | Pipeline 相关职责 |
|-------|------------------|
| `diffsynth-analyze-target` | 分析目标库的推理流程，规划功能列表和优先级，为 Pipeline 接入提供蓝图 |
| `diffsynth-model-code` | 确保模型组件（DiT、VAE、TextEncoder）已正确接入，为 Pipeline 提供可调用的模型 |
| `diffsynth-pipeline` | 创建/更新 Pipeline 文件、推理示例、训练模块，执行端到端一致性验证 |
| `diffsynth-docs` | 从 Pipeline `__call__` 签名提取参数说明，生成文档中的"模型推理"章节 |
| `diffsynth-testing` | 运行已接入的 Pipeline 推理脚本，验证输出质量 |

---

*本文档基于 `diffsynth/diffusion/base_pipeline.py` 和实际 Pipeline 实现（`z_image.py`、`flux2_image.py`）提取。修改时需同步更新所有依赖此文档的 skill。*
