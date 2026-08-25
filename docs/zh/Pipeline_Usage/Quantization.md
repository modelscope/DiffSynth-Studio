# 模型量化

量化通过降低模型权重的数值精度来减少显存占用，让大模型能在更小的显卡上运行。`DiffSynth-Studio` 提供统一的量化入口 `QuantizeConfig`，支持 bitsandbytes、torchao、comfy-kitchen 等多个量化后端，并支持在线量化、加载预量化权重、混合量化以及量化 + LoRA 训练。

本文以 `Z-Image` 为例介绍量化的使用。如果你希望把 `diffsynth.core.quant` 用到自己的代码库中，请参考 [`diffsynth.core.quant` API 文档](../API_Reference/core/quant.md)。

> **量化 与 显存管理 FP8 的区别**
>
> [显存管理](./VRAM_management.md)中的 FP8 通过 `offload_dtype` / `onload_dtype` 等参数控制权重在显存中的存储精度，作用于全部参数、不依赖第三方库，但只有简单的 FP8 转换。
>
> 本文的量化（`QuantizeConfig`）是针对 `nn.Linear` 的专门方案，支持 NF4、INT8、INT4、MXFP4、NVFP4 等更精细的格式，可保存/加载量化权重，支持激活量化与量化 + LoRA 训练。两者可以组合使用。

## 安装依赖

不同量化后端需要对应的第三方库：

| 后端 | 安装命令 | Project Page |
| --- | --- | --- |
| bitsandbytes | `pip install bitsandbytes` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| torchao | `pip install torchao>=0.16` | [torchao](https://github.com/pytorch/ao) |
| comfy-kitchen | `pip install comfy-kitchen` | [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) |

一次性安装全部：`pip install "diffsynth[quant]"`

## 快速开始

在任意 `ModelConfig` 上传入 `quantize` 即可对该模型启用在线量化。以下代码把 Z-Image 的 DiT 用 NF4 量化后加载：

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="bitsandbytes_nf4"),
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt=prompt, seed=42, num_inference_steps=50, cfg_scale=4)
image.save("image_z_image_nf4.jpg")
```

## 支持的量化方法

以下是内置的全部量化方法，`method` 即传入 `QuantizeConfig` 的名称。命名遵循 `W<权重位宽>A<激活位宽>` 约定：`w8a16` 表示只量化权重（weight-only），`w8a8` 表示权重与激活都量化。

| method | 后端 | 权重 / 激活 | 可序列化 | 支持 LoRA 训练 |
| --- | --- | --- | --- | --- |
| `bitsandbytes_nf4` | bitsandbytes | NF4 / 不量化 | ✅ | ✅ |
| `bitsandbytes_fp4` | bitsandbytes | FP4 / 不量化 | ✅ | ✅ |
| `torchao_int8_w8a16` | torchao | INT8 / 不量化 | ✅ | ✅ |
| `torchao_fp8_w8a16` | torchao | FP8 / 不量化 | ✅ | ✅ |
| `torchao_int4_w4a16` | torchao | INT4 / 不量化 | ✅ | ❌ |
| `torchao_nvfp4_w4a16` | torchao | NVFP4 / 不量化 | ✅ | ✅ |
| `torchao_int8_w8a8` | torchao | INT8 / INT8 动态 | ✅ | ❌ |
| `torchao_fp8_w8a8` | torchao | FP8 / FP8 动态 | ✅ | ❌ |
| `torchao_int4_w4a8` | torchao | INT4 / FP8 动态 | ✅ | ❌ |
| `torchao_mxfp8_w8a8` | torchao | MXFP8 / MXFP8 | ✅ | ❌ |
| `torchao_mxfp4_w4a4` | torchao | MXFP4 / MXFP4 | ✅ | ❌ |
| `torchao_nvfp4_w4a4` | torchao | NVFP4 / NVFP4 | ✅ | ❌ |
| `comfy_kitchen_int8_w8a8` | comfy_kitchen | INT8 / INT8 动态 | ✅ | ✅ |
| `comfy_kitchen_fp8_w8a8` | comfy_kitchen | FP8 E4M3 / FP8 | ✅ | ✅ |

几点说明：

- **激活量化**（`w8a8` / `w4a4`）在压缩权重之外还会量化激活值，在支持对应低精度矩阵乘的硬件上可以真正提速，而 weight-only 方案通常只省显存。
- **LoRA 训练**：只有表中"支持 LoRA 训练"为 ✅ 的方法可用于量化 + LoRA 训练。
- `comfy_kitchen_*` 方法读写的是 ComfyUI 的量化权重格式，可与 ComfyUI 生态互通。comfy-kitchen 需要 CUDA 13.0 及以上。
- MXFP8 / MXFP4 / NVFP4 等格式对计算硬件有要求，具体兼容性请查阅 [torchao](https://github.com/pytorch/ao) 文档。

你可以在代码中查询所有可用方法及其参数：

```python
from diffsynth.core.quant import describe_quant_method, QUANT_METHODS, backends

backends.load_all_backends()
print(sorted(QUANT_METHODS))          # 所有已注册的方法名

describe_quant_method("bitsandbytes_nf4")
```

输出如下，其中 `backend_config_kwargs (user-tunable)` 列出了该方法可调整的参数及默认值，这些参数决定量化的行为，你可以根据需要修改它们。对于 torchao 后端，部分参数会直接传递给 torchao 自己的 config（如 `Int8WeightOnlyConfig`）：

> 除非你清楚这些参数的含义，否则建议保留默认值。

```
method: bitsandbytes_nf4
backend: bitsandbytes
detail: 4bit, nf4, weight-only
backend config: diffsynth.core.quant.backends.bitsandbytes.BitsAndBytesNF4Config
backend_config_kwargs (user-tunable):
  compress_statistics = True
  blocksize           = None
  quant_storage       = torch.uint8
pinned by method (not overridable):
  quant_type = 'nf4'
```

## QuantizeConfig 详解

`QuantizeConfig` 描述了"用哪种方法、量化哪些层、量化后如何运行"：

- **`method`**：量化方法名，见上表，必填。
- **`mode`**：量化层的运行方式。
    - `"dynamic"`（默认）：保留量化 Linear，forward 时按需反量化，显存占用低。
    - `"dequant_once"`：量化完成后一次性还原成普通 fp `nn.Linear`（保留量化误差）。适合需要标准 `nn.Linear` 的场景，不再省显存。
- **`target_modules` / `exclude_modules`**：按层名过滤要量化的 `nn.Linear`，取值为列表。匹配规则是完整点分名称相等，或以 `"." + 名称` 结尾（例如 `"img_mod.1"` 可匹配 `transformer_blocks.0.img_mod.1`）。
- **`backend_config_kwargs`**：传给后端配置的参数字典，决定量化的行为（torchao 后端的部分参数会直接传给 torchao 自己的 config）。可先用 `describe_quant_method(method)` 查询可用参数。
- **`load_prequantized`**：设为 `True` 表示 checkpoint 中已经是量化权重，直接加载（见下文）。

示例：排除对量化敏感的层，并调整 NF4 的后端参数：

```python
from diffsynth.core.quant import QuantizeConfig

quantize = QuantizeConfig(
    method="bitsandbytes_nf4",
    mode="dynamic",
    exclude_modules=["time_embedder.proj_in", "time_embedder.proj_out", "proj_out"],
    backend_config_kwargs={"compress_statistics": False},
)
```

激活量化方法的用法完全一致，只是换个 `method`：

```python
quantize = QuantizeConfig(method="comfy_kitchen_int8_w8a8", backend_config_kwargs={"convrot_groupsize": 128})
```

## 加载预量化权重

除在线量化外，也支持直接加载已量化好的 checkpoint，省去每次加载时的量化开销。

对于官方发布的量化模型（如 `ideogram-ai/ideogram-4-nf4`），配置中已写好量化信息，像加载普通模型一样即可：

```python
ModelConfig(model_id="ideogram-ai/ideogram-4-nf4", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors")
```

对于自己保存的量化 checkpoint（见下一节），加载时显式传入 `quantize` 并设置 `load_prequantized=True`，其中 `method` 与 `exclude_modules` 必须与保存时保持一致：

```python
from diffsynth.core.quant import QuantizeConfig

ModelConfig(
    path="models/z-image-nf4/transformer.safetensors",
    quantize=QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True),
)
```

## 保存量化模型

想把一次在线量化的结果保存下来反复使用，可以用 `save_quantized_model`：

```python
from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.utils.quant.serialization import save_quantized_model

model_config = ModelConfig(
    model_id="Tongyi-MAI/Z-Image",
    origin_file_pattern="transformer/*.safetensors",
    quantize=QuantizeConfig(method="bitsandbytes_nf4"),
)
save_quantized_model(model_config, "models/z-image-nf4/transformer.safetensors")
```

它会下载并加载原始 fp 权重、执行量化，再把量化后的 state dict 存成 `.safetensors`。保存后即可用上一节的方式加载。

## 混合量化

不同层对量化的敏感程度不同。`MixedQuantizeConfig` 允许对不同层集合应用不同方法，例如对精度敏感的调制层用 INT8、其余层用 NF4：

```python
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig

mod_layers = ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out"]
quantize = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
])
```

各子配置匹配到的层集合必须互不重叠。所有子配置必须共享同一个 `mode`。`MixedQuantizeConfig` 对外接口与 `QuantizeConfig` 完全一致，可直接传给 `ModelConfig(quantize=...)`，也可以传给 `save_quantized_model`。

> 加载混合量化的预量化 checkpoint 时，`load_prequantized=True` 要设置在 `MixedQuantizeConfig` 上，而不是子配置上。

## 量化 + LoRA 训练

在大多数情况下，量化后的模型不支持训练，但支持冻结基础模型后的 LoRA 训练，从而在很小的显存里训练大模型。

可用于量化 + LoRA 训练的方法见[方法表](#支持的量化方法)的最后一列。有两种使用方式。

### 方式一：用预量化的底模训练

训练脚本通过 `--model_id_with_origin_paths` 指向预量化模型：

```bash
accelerate launch examples/.../train.py \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-nf4.safetensors,..." \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --output_path "./models/train/xxx-nf4"
```

### 方式二：用 `--quant_options` 在线量化

如果没有现成的预量化权重，可以用 `--quant_options` 在训练启动时对加载的模型做在线量化。取值以 `;` 分隔多个条目，每个条目的格式为 `<模型字符串>:<method>[/<exclude_modules>]`：

- `<模型字符串>`：要量化的模型，必须与 `--model_paths` / `--model_id_with_origin_paths` 中的写法完全一致。
- `<method>`：量化方法名，见[方法表](#支持的量化方法)。
- `<exclude_modules>`：可选，以 `,` 分隔的层名列表，这些层保持全精度。

以 Z-Image-Turbo 的量化 LoRA 训练为例（完整脚本见 `examples/z_image/model_training/special/quant_training/Z-Image-Turbo-bitsandbytes_nf4.sh`）：

```bash
accelerate launch examples/z_image/model_training/train.py \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --quant_options "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors:bitsandbytes_nf4;Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors:bitsandbytes_nf4" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --output_path "./models/train/Z-Image-Turbo_quant_lora"
```

上例中 DiT 与 text encoder 都启用了 NF4 量化。`text_encoder`、`vae` 等不参与训练的模块可以放心量化；被训练的 `dit` 只能在 LoRA 训练下量化，且必须选用支持 LoRA 训练的方法——如果指定了不可微的方法，训练会直接报错退出。

带 `exclude_modules` 的写法（保留对量化敏感的层）：

```bash
  --quant_options "MiniMaxAI/MiniMax-H3:FL2VA/transformer/model*.safetensors:bitsandbytes_nf4/time_embedder.proj_in,time_embedder.proj_out,video_patch_proj,audio_patch_proj"
```

> `--quant_options` 使用的是 `mode="dynamic"`，不支持配置 `backend_config_kwargs`、混合量化等更复杂的选项。有这类需求时请改用方式一：先用 `save_quantized_model` 保存量化权重，再用预量化底模训练。

### 通用说明

- 训练中量化底模保持冻结，只有 LoRA 分支更新，因此保存下来的是 fp 精度的 LoRA 权重。
- 推理时按"量化底模 + LoRA"加载：先像[加载预量化权重](#加载预量化权重)那样加载量化底模，再 `pipe.load_lora(pipe.dit, "epoch-x.safetensors")`。
- 大模型建议优先选方式一：在线量化需要先加载完整 fp 权重，启动慢且峰值内存高。

## 自定义量化后端

如果内置方法不满足需求，你可以实现自己的量化后端。完整的接入流程与可运行的示例（以玩具后端 INT9 为例）见[接入量化后端](../Developer_Guide/Integrating_Quantization_Backend.md)，完整的接口签名与契约见 [`diffsynth.core.quant` API 文档](../API_Reference/core/quant.md#扩展接口自定义后端)。

## 量化与显存管理组合

量化与[显存管理](./VRAM_management.md)解决的是不同层面的问题，可以同时启用：

- 量化降低**每一层的存储体积**，例如 NF4 约为 bf16 的 1/4。
- 显存管理决定**哪些层此刻留在显存里**，其余按需从内存/硬盘调入。

两者组合可以进一步压低推理所需的显存：先把权重压缩到 4bit/8bit，再用 `vram_limit` 把压缩后的模型拆分到显存与内存中。

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="bitsandbytes_nf4"), **vram_config,
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
```

两点注意：

- `vram_config` 中的 `offload_dtype` / `onload_dtype` 等参数作用于未量化的参数；已量化层的存储精度由量化方法决定，不受这些参数影响。
- **Disk Offload 与在线量化不兼容**。Disk Offload 按层从硬盘读取参数，要求量化后的参数已经保存在磁盘上，因此不能先走 Disk Offload 再做在线量化。若要结合 Disk Offload 使用量化，请先按[最佳实践](#最佳实践)的流程保存量化权重，再加载预量化 checkpoint。

## 最佳实践

以 MiniMax-H3 为例，展示从保存量化权重到加载推理的完整流程。

### 第一步：保存量化权重

MiniMax-H3 的 FL2VA 底模约 66G（bf16），直接用 NF4 在线量化会很慢，建议先量化并保存一次，之后反复加载。`save_quantized_model` 会返回保存文件的 hash：

```python
from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.utils.quant.serialization import save_quantized_model

quantize = QuantizeConfig(
    method="bitsandbytes_nf4",
    mode="dynamic",
    exclude_modules=[
        "time_embedder.proj_in", "time_embedder.proj_out",
        "video_patch_proj", "audio_patch_proj", "condition_proj",
        "final_layer.video_out", "final_layer.audio_out",
    ],
)
model_config = ModelConfig(
    model_id="MiniMaxAI/MiniMax-H3",
    origin_file_pattern="FL2VA/transformer/model*.safetensors",
    quantize=quantize,
)
model_hash = save_quantized_model(model_config, "models/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors")
print(model_hash)
```

`exclude_modules` 里是对量化敏感的层（时间步嵌入、输入输出投影），保留 bf16 以维持质量。

### 第二步：把 hash 写入模型配置

框架通过文件 hash 识别模型类型与量化配置。注册条目如下，`quant_config` 需与保存时的 `QuantizeConfig` 保持一致并加上 `load_prequantized: True`：

```python
config_entry = {
    # Example: ModelConfig(model_id="...", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors")
    "model_hash": model_hash,
    "model_name": "minimax_h3_dit",
    "model_class": "diffsynth.models.minimax_h3_dit.MiniMaxH3DiT",
    "quant_config": {"method": "bitsandbytes_nf4", "load_prequantized": True, "exclude_modules": ["time_embedder.proj_in", "time_embedder.proj_out", "video_patch_proj", "audio_patch_proj", "condition_proj", "final_layer.video_out", "final_layer.audio_out"]},
}
```

注册方式有两种：

**方式一：在 Python 代码中动态注册（推荐，即插即用）**。无需改动框架代码，在加载模型之前把条目加入 `MODEL_CONFIGS` 即可生效，仅作用于当前进程：

```python
from diffsynth.configs import MODEL_CONFIGS

MODEL_CONFIGS.append(config_entry)
```

**方式二：写入配置文件（永久生效）**。把上面的条目添加到 `diffsynth/configs/model_configs.py` 的 `MODEL_CONFIGS` 列表中，这样在本地无需再手动注册。如果你的量化权重已经公开发布，也欢迎把这个条目提交 PR 给我们，让其他用户可以直接加载。

### 第三步：加载推理

注册完成后，加载自己的量化权重就和加载普通模型一样，无需传入 `quantize` 参数：

```python
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
prompt = "A girl is very happy, she is speaking in english: 'I enjoy working with Diffsynth-Studio, it's a perfect framework.'"
video, audio = pipe(prompt=prompt, height=480, width=832, num_frames=124, num_inference_steps=50, seed=0)
write_video_audio(video=video, audio=audio, output_path="t2va.mp4", fps=24, audio_sample_rate=32000)
```

我们已将 MiniMax-H3 的 NF4 量化权重上传到 ModelScope（[DiffSynth-Studio/MiniMax-H3-NF4](https://modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)），可以直接使用而无需自行量化。如果你想把自己保存的量化权重上传到 ModelScope，可以用 modelscope SDK：

```python
from modelscope.hub.api import HubApi

api = HubApi()
api.login("your_access_token")
api.create_model("your-username/MiniMax-H3-NF4", visibility=1)
api.upload_folder(
    repo_id="your-username/MiniMax-H3-NF4",
    folder_path="models/MiniMax-H3-NF4",
    repo_type="model",
)
```
