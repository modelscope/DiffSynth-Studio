# HiUni

HiUni 是由 ERNIE Research 开源的统一多模态模型，同一个 checkpoint 即可完成图像理解、文生图与指令图像编辑。它是一个小模型，在视觉语言骨干网络上增加了一个轻量的像素空间扩散头，直接以像素生成图像，无需 VAE 与独立的文本编码器。

## 安装

在使用本项目进行模型推理前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以从 Hugging Face 加载 [ernie-research/HiUni](https://huggingface.co/ernie-research/HiUni) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载。

```python
from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
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

pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors", download_source="huggingface", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json", download_source="huggingface"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json", download_source="huggingface"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A cozy reading nook by a rain-streaked window, a ginger cat asleep on a knitted blanket, warm lamp light, film photography."
image = pipe(prompt, height=2048, width=2048, seed=42, num_inference_steps=25, cfg_scale=4.0, shift=4.0)
image.save("image_HiUni.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[ernie-research/HiUni](https://huggingface.co/ernie-research/HiUni)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference_low_vram/HiUni.py)|-|-|-|-|

## 模型推理

模型通过 `HiUniPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。权重托管在 Hugging Face 上，因此 `ModelConfig` 需设置 `download_source="huggingface"`。`tokenizer_config` 与 `processor_config` 从仓库根目录加载分词器与图像处理器，只会下载对应的 JSON 文件。

`HiUniPipeline` 推理的输入参数包括：

* `prompt`: 文本提示词。图像编辑时为编辑指令。
* `cfg_scale`: 文本引导系数，默认为 4.0。
* `height`: 输出图像高度，文生图默认为 1024。必须为 32 的倍数。
* `width`: 输出图像宽度，文生图默认为 1024。必须为 32 的倍数。
* `seed`: 随机种子，默认为 42。
* `num_inference_steps`: 推理步数，默认为 25。
* `shift`: 时间步偏移量，影响 sigma 计算，默认为 4.0。
* `edit_image`: 参考图像。传入后切换到图像编辑模式，默认为 None（文生图模式）。
* `image_cfg_scale`: 编辑时的图像引导系数，默认为 1.0。不等于 1.0 时会增加第三个无条件分支。
* `resize_to_edit_image`: 是否把编辑结果缩放回 `edit_image` 的尺寸，默认为 True。
* `min_pixels`、`max_pixels`: 编辑模式下参考图与生成图的像素预算，默认均为 262144（512x512）。

### 图像编辑

传入 `edit_image` 后切换到图像编辑模式。参考图经视觉编码器编码后放在用户轮的编辑指令之前，其像素也会在每一步输入扩散头。未指定 `height`/`width` 时，输出使用参考图预处理后的分辨率，在 `min_pixels`/`max_pixels` 预算内保持其宽高比。`resize_to_edit_image=True` 时，结果会缩放回参考图的原始尺寸。

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe("Change the background to a snowy mountain.", edit_image=edit_image, seed=42)
```

当 `image_cfg_scale != 1` 时，引导由三路预测组合：`full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full)`。其中 `posi` 同时看到参考图与指令，`nega` 只看到参考图，`full` 两者都看不到。默认 `image_cfg_scale=1.0` 时 `full` 分支相互抵消，不会计算。

### 图像理解与文本对话

`pipe.chat` 使用同一个 checkpoint 进行自回归文本生成。传入 `images`（单张图像或列表）进行图像理解，省略则为纯文本对话：

```python
answer = pipe.chat("Describe this image in detail.", images=edit_image)
answer = pipe.chat("Write a short poem about autumn.", think_mode="empty")
```

* `think_mode`: `"think"`（默认）让模型先写 `<think>` 推理块再回答，`"empty"` 预填一个空推理块，`"none"` 不预填。
* `max_new_tokens`: 最大生成 token 数，默认为 1024。
* `temperature`、`top_k`、`top_p`、`seed`: 采样参数。默认 `temperature=0.0` 为贪心解码。
* `min_pixels`、`max_pixels`: 输入图像的像素预算，默认为 262144 与 4194304。
* `return_raw`: 返回包含推理块的完整解码文本。默认只返回 `</think>` 之后的回答。

与训练一致，解码不使用 KV cache，每生成一个 token 都会重新计算整个序列。

### 架构说明

* **共享骨干网络**：理解、生成与编辑都经过同一个语言模型。扩散头读取骨干网络在噪声图像 token 上的隐状态，因此 DiffSynth 只注册了一个 `hiuni_dit` 组件。
* **无 VAE 的像素空间去噪**：`latents` 存放 16x16 RGB 栅格 patch，形状为 `(4 * H/32 * W/32, 768)`，取值范围 `[-1, 1]`。模型预测干净图像 `x0`，引导在 `x0` 上进行，再换算为 flow 速度用于调度器步进。
* **随分辨率缩放的噪声**：初始噪声乘以 `sqrt(4 * H/32 * W/32 / 256)`，与训练一致。
