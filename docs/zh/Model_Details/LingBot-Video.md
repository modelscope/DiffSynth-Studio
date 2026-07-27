# LingBot-Video

LingBot-Video 是一个基于 flow-matching 的文生视频生成模型。本文档介绍 DiffSynth-Studio 对 **Dense-1.3B** 与 **MoE-30B-A3B** 文生视频权重的推理支持，以及对 Dense-1.3B 的 LoRA SFT 训练支持。

该接入基于标准的 DiffSynth Pipeline 组件栈构建：

- **DiT** — `LingBotVideoDiT`（`diffsynth/models/lingbot_video_dit.py`），视频去噪器。同一个类同时覆盖两个版本：Dense-1.3B 使用普通 FFN（`num_experts=0`），MoE-30B-A3B 使用稀疏 MoE FFN。
- **文本编码器** — `LingBotVideoTextEncoder`（Qwen3-VL）。提示词会被包裹进提示增强的对话模板中编码，随后裁剪掉模板前缀 token。
- **VAE** — 复用 DiffSynth 的 `QwenImageVAE`（与 LingBot-Video 的 VAE 逐字节一致），空间 8× / 时间 4× 压缩。
- **调度器** — `LingBotVideoUniPCScheduler`：推理时使用 UniPC 多步调度；训练时回退到完整分辨率的 flow-matching 调度。

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
    prompt="A playful puppy runs across a lush green meadow, its golden fur shining in the bright sunlight. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0, seed=0,
)
save_video(video, "video.mp4", fps=15, quality=10)
```

**低显存：** 在每个 `ModelConfig` 上设置 `offload_dtype` / `offload_device` 即可开启逐层 offload，可再配合 `vram_limit=<GB>`，详见下表中的低显存示例。

## MoE-30B-A3B

[Robbyant/lingbot-video-moe-30b-a3b](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b) 是更大的版本：总参数量 30B，每个 token 激活约 3B。每个 MoE 层包含 128 个路由专家和 1 个共享专家，并使用 group-limited top-k（4 组，取 top-2 组）将每个 token 路由到 8 个专家。

它复用同一条 pipeline，只需更换模型 ID 与分片通配符（权重被切分为 13 个 shard）：

```python
pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-moe-30b-a3b", origin_file_pattern="processor/"),
)
```

MoE 版本的注意事项：

- 专家矩阵乘在 CUDA 上使用 `torch._grouped_mm`，在其他设备上回退到等价的逐专家循环。
- 专家权重以分组的 `nn.Parameter` 而非 `nn.Linear` 存储，因此显存管理包装的是专家容器本身。由于专家占据了模型的绝大部分参数，低显存路径能让常驻显存接近约 3B 的激活规模，而不是完整的 30B。
- 官方权重包中还提供了第二阶段的 `refiner/` DiT（架构相同，用于在 base 结果之上做超分精修），目前尚未接入 pipeline。

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b.py)|
|[Robbyant/lingbot-video-moe-30b-a3b](https://modelscope.cn/models/Robbyant/lingbot-video-moe-30b-a3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-moe-30b-a3b.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-moe-30b-a3b.py)|-|-|-|-|

## 模型推理

模型通过 `LingBotVideoPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`LingBotVideoPipeline` 推理的输入参数包括：

* `prompt`: 描述视频内容的提示词。支持结构化 caption（`dict` / `list`）、`prompt.json` 路径，或普通字符串；详见[提示词改写](#提示词改写对质量很重要)。
* `negative_prompt`: 负向提示词，描述不希望出现在视频中的内容。Pipeline 内置了默认（T2V）负向提示词，因此可以不设置。
* `input_video`: 输入视频（帧列表或 `VideoData`），用于视频生视频，需与 `denoising_strength` 配合使用。
* `denoising_strength`: 降噪强度，取值范围 0~1，默认为 1.0。值越小，越保留输入视频的结构。仅在提供 `input_video` 时生效。
* `height`: 视频高度，默认为 480，需能被 16 整除。
* `width`: 视频宽度，默认为 480，需能被 16 整除。
* `num_frames`: 视频帧数，默认为 81，需满足 `4k+1`（VAE 在时间维上做 4× 压缩）。
* `cfg_scale`: 分类器自由引导系数，默认为 6.0。LingBot-Video 推荐使用 3.0。
* `num_inference_steps`: 推理步数，默认为 40。
* `sigma_shift`: flow-matching 时间步偏移，默认为 3.0。
* `seed`: 随机种子，默认为 `None`，即完全随机。
* `rand_device`: 生成初始噪声的设备，默认为 `"cpu"`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm`，可设为 `lambda x: x` 关闭。

显存不足时，请参考[显存管理](../Pipeline_Usage/VRAM_management.md)启用显存管理功能。

## 提示词改写（对质量很重要）

LingBot-Video 使用**结构化 JSON caption** 训练，而非自由文本。喂入一句普通句子属于分布外（out-of-distribution）输入，会明显降低质量；喂入模型期望的结构化 caption 则能恢复质量。Pipeline 接受 `dict`、`prompt.json` 路径或普通字符串形式的 caption，并将其（通过 `normalize_caption`）归一化为 DiT 训练时使用的紧凑 JSON 格式——普通字符串会原样透传，因此已有脚本无需改动。

若要把一个**简短想法**转成该结构化 caption，可使用内置的两阶段改写器（`diffsynth/pipelines/lingbot_video_prompt_rewriter.py`）：阶段一将想法*扩写*为自然语言 caption，阶段二将其*映射*为结构化 JSON。

```python
from diffsynth.pipelines.lingbot_video_prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

改写器是**独立的 VLM + 阶段二 LoRA 适配器**（不是 DiT）。可通过 `REWRITER_BASE_MODEL` / `REWRITER_ADAPTER`（或 `base=` / `adapter=`）指向权重，也可以通过传入一个暴露 `generate(text, image, use_lora)` 方法的自定义对象作为 `backend=`，来驱动托管的 / OpenAI 兼容的推理端点。详见 `examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_rewrite.py`。

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

为获得最佳效果，`prompt` 列应存放**结构化 JSON caption**（与推理时一致的分布内格式——见[提示词改写](#提示词改写对质量很重要)）。`train.py` 会对每条 prompt 调用 `normalize_caption`。若数据集存放的是原始文本，请在训练前用 `examples/lingbot_video/model_training/rewrite_captions.py` 离线改写一次。

我们编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。

## 注意事项

- 文本编码器与已有的 `krea2_text_encoder` 共享同一 checkpoint 指纹（Qwen3-VL 架构相同），因此模型加载器在加载 LingBot-Video 时会同时实例化两者。这只是冗余的加载耗时——Pipeline 会按名称取用正确的编码器，另一个会被释放。
- Latent 归一化在 VAE 的 5D 视频代码路径内部处理；Pipeline 不会再额外应用 `latents_mean` / `latents_std`。
