# LingBot-Video

LingBot-Video 是一个基于 flow-matching 的文生视频生成模型。本文档介绍 DiffSynth-Studio 对 **Dense-1.3B** 文生视频权重的推理与 LoRA SFT 训练支持。

该接入基于标准的 DiffSynth Pipeline 组件栈构建：

- **DiT** — `LingBotVideoDiT`（`diffsynth/models/lingbot_video_dit.py`），视频去噪器。Dense-1.3B 版本使用普通 FFN；该架构同时支持 MoE FFN。
- **文本编码器** — `LingBotVideoTextEncoder`（Qwen3-VL）。提示词会被包裹进提示增强的对话模板中编码，随后裁剪掉模板前缀 token。
- **VAE** — 复用 DiffSynth 的 `QwenImageVAE`（与 LingBot-Video 的 VAE 逐字节一致），空间 8× / 时间 4× 压缩。
- **调度器** — DiffSynth 的 `FlowMatchScheduler`（Wan 模板）：推理时使用一阶 flow-matching Euler；训练时使用完整分辨率的 1000 步 flow-matching 调度。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

LingBot-Video 还额外依赖 `transformers >= 5.x`（用于 Qwen3-VL）以及 `imageio` / `imageio-ffmpeg`（用于视频读写）。更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) 模型并进行文生视频推理。首次运行时会自动下载所需文件。

> **⚠️ 推理前请先把提示词改写为结构化 caption。** LingBot-Video 使用**结构化 JSON caption** 训练，而非自由文本。下面代码里的普通句子能跑通，但属于分布外输入，生成结果会明显偏"糊"、质量偏低。这是模型的预期行为，并非 bug —— 正式推理前，请先把想法转成模型期望的结构化 caption。详见下文[提示词改写](#提示词改写对质量很重要)；可直接运行的 [`lingbot-video-dense-1.3b.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py) 示例默认使用随仓库发布的结构化 caption，并在文件末尾给出可选的改写流程。

```python
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
)
video = pipe(
    # 普通句子仅用于最简单的跑通验证，属于分布外输入。
    # 正式使用请改传结构化 caption（见"提示词改写"章节）。
    prompt="A playful puppy runs across a lush green meadow, its golden fur shining in the bright sunlight. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0, seed=0,
)
save_video(video, "video.mp4", fps=15, quality=10)
```

**低显存：** 在每个 `ModelConfig` 上设置 `offload_dtype` / `offload_device` 才能开启逐层 offload；单独传 `vram_limit` 没有效果（它只在 offload 已开启时限制常驻显存上限）。详见下表中的低显存示例。

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b.py)|

## 模型推理

模型通过 `LingBotVideoPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`LingBotVideoPipeline` 推理的输入参数包括：

* `prompt`: 描述视频内容的提示词。支持结构化 caption（`dict`）或普通字符串；详见[提示词改写](#提示词改写对质量很重要)。
* `negative_prompt`: 负向提示词，描述不希望出现在视频中的内容，默认值为 `""`。官方 T2V 负向提示词内置在 `pipe.default_negative_prompt` 中，可通过 `negative_prompt=pipe.default_negative_prompt` 传入。
* `input_video`: 输入视频（帧列表或 `VideoData`），用于视频生视频，需与 `denoising_strength` 配合使用。
* `denoising_strength`: 降噪强度，取值范围 0~1，默认为 1.0。值越小，越保留输入视频的结构。仅在提供 `input_video` 时生效。
* `height`: 视频高度，默认为 480，需能被 16 整除。
* `width`: 视频宽度，默认为 480，需能被 16 整除。
* `num_frames`: 视频帧数，默认为 81，需满足 `4k+1`（VAE 在时间维上做 4× 压缩）。
* `cfg_scale`: 分类器自由引导系数，默认为 6.0。Dense-1.3B 模型推荐使用 3.0。
* `num_inference_steps`: 推理步数，默认为 40。
* `sigma_shift`: flow-matching 时间步偏移，默认为 3.0。
* `seed`: 随机种子，默认为 `None`，即完全随机。
* `rand_device`: 生成初始噪声的设备，默认为 `"cpu"`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm`，可设为 `lambda x: x` 关闭。

显存不足时，请参考[显存管理](../Pipeline_Usage/VRAM_management.md)启用显存管理功能。

## 提示词改写（对质量很重要）

LingBot-Video 使用**结构化 JSON caption** 训练，而非自由文本。喂入一句普通句子属于分布外（out-of-distribution）输入，会明显降低质量；喂入模型期望的结构化 caption 则能恢复质量。Pipeline 接受 `dict` 或普通字符串形式的 caption，并将其归一化为 DiT 训练时使用的紧凑 JSON 格式——`dict` 会自动序列化，普通字符串会原样透传，因此已有脚本无需改动。

若要把一个**简短想法**转成该结构化 caption，可使用随示例发布的两阶段改写器（`examples/lingbot_video/model_training/scripts/prompt_rewriter.py`）：阶段一将想法*扩写*为自然语言 caption，阶段二将其*映射*为结构化 JSON。

改写器是**独立的 VLM + 阶段二 LoRA 适配器**（不是 DiT），**不会自动下载**——运行改写前，需要先自行下载这两个权重：

| 角色 | 模型 ID | 大小 |
|-|-|-|
| 改写器 base VLM（阶段一 + 二） | [`Qwen/Qwen3.6-27B`](https://modelscope.cn/models/Qwen/Qwen3.6-27B) | ~55 GB |
| 改写器阶段二 LoRA 适配器 | [`Robbyant/lingbot-video-rewriter-lora`](https://modelscope.cn/models/Robbyant/lingbot-video-rewriter-lora) | ~0.5 GB |

```shell
# 1. 下载改写器 base VLM 及其阶段二 LoRA 适配器。
modelscope download --model Qwen/Qwen3.6-27B --local_dir ./models/Qwen/Qwen3.6-27B
modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir ./models/Robbyant/lingbot-video-rewriter-lora
```

```python
# 2. 让改写器指向已下载的权重，再改写并推理。
import os
os.environ["REWRITER_BASE_MODEL"] = "./models/Qwen/Qwen3.6-27B"
os.environ["REWRITER_ADAPTER"] = "./models/Robbyant/lingbot-video-rewriter-lora"

# 在仓库根目录下运行，此包式 import 才能解析。
from examples.lingbot_video.model_training.scripts.prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

除了 env var，也可以给 `rewrite_prompt` 传 `base=` / `adapter=`；或者完全不下载本地 VLM，改为传入一个暴露 `generate(text, image, use_lora)` 方法的自定义对象作为 `backend=`，来驱动托管的 / OpenAI 兼容的推理端点。详见 `examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py` 文件末尾的可选改写小节。

如果没有改写器模型，官方 LingBot-Video t2v 结构化 caption 已随样例数据集发布（`DiffSynth-Studio/diffsynth_example_dataset` 中的 `t2v_example_*.json`，推理示例脚本会自动下载）。用 `json.load` 读入后作为 `dict` 传给 pipeline，也可以复制一个作为编写自己 caption 的模板。

## 模型训练

LingBot-Video 通过 [`examples/lingbot_video/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/train.py) 进行训练，使用 flow-matching SFT 目标对 DiT 做 LoRA 微调。脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径（含 `video` 列与 `prompt` 列的 CSV / JSONL）。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要按文件加载的字段名称，通常是视频文件路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，用逗号分隔。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在保存模型的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。留空则每个 epoch 保存一次。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上，例如 `dit`。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: 用于续训 / 继续训练的 LoRA 检查点路径。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 视频宽高配置
        * `--height`: 视频高度，需能被 16 整除。
        * `--width`: 视频宽度，需能被 16 整除。
        * `--num_frames`: 视频帧数，需满足 `4k+1`。
* LingBot-Video 专有参数
    * `--processor_path`: 文本编码器使用的 Qwen3-VL processor 路径。

启动脚本会先下载 DiffSynth-Studio 通用的示例视频 SFT 数据集，然后在其上训练：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset
```

### 仅注意力 LoRA（默认范围）

推荐的启动脚本仅在文本+视频联合自注意力上添加 LoRA：

```
--lora_base_model "dit"
--lora_target_modules "to_q,to_k,to_v,to_out"
--lora_rank 32
--remove_prefix_in_ckpt "pipe.dit."
```

MoE / FFN 专家（`gate_proj`、`up_proj`、`down_proj`）与 router 保持冻结。若要同时微调 FFN，可将这些模块名加入 `--lora_target_modules`。

为获得最佳效果，`prompt` 列应存放**结构化 JSON caption**（与推理时一致的分布内格式——见[提示词改写](#提示词改写对质量很重要)）。Pipeline 会在内部对每条 prompt 做归一化。若数据集存放的是原始文本，请在训练前用 `examples/lingbot_video/model_training/scripts/rewrite_captions.py` 离线改写一次。

我们编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。

## 注意事项

- 文本编码器与已有的 `krea2_text_encoder` 共享同一 checkpoint 指纹（Qwen3-VL 架构相同），因此模型加载器在加载 LingBot-Video 时会同时实例化两者。这只是冗余的加载耗时——Pipeline 会按名称取用正确的编码器，另一个会被释放。
- 5D 视频的 VAE 编码/解码及其 latent 归一化都在 Pipeline 内实现（`LingBotVideoPipeline.encode_video` / `decode_video`），因此 `QwenImageVAE` 与它在图像场景下的用法保持逐字节一致；Pipeline 不会再额外应用 `latents_mean` / `latents_std`。
