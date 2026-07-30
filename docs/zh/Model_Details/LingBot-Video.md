# LingBot-Video

LingBot-Video 是由 LingBot 团队研发的 flow-matching 视频生成模型，单个模型即可完成文生视频、图生视频和文生图三种任务。

特别感谢 [NancyFyong](https://github.com/NancyFyong) 在模型接入中做出的杰出贡献！

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 6G 显存即可运行。

```python
import torch
import json
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_t2v/*",
)
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_t2v/t2v_example_1.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video.mp4", fps=15, quality=10)
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Robbyant/lingbot-video-dense-1.3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_t2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_t2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_t2v.py)|
|[Robbyant/lingbot-video-dense-1.3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_ti2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_ti2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_ti2v.py)|
|[Robbyant/lingbot-video-dense-1.3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2i.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2i.py)|-|-|-|-|

## 模型推理

模型通过 `LingBotVideoPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`LingBotVideoPipeline` 推理的输入参数包括：

* `prompt`: 描述视频内容的提示词，接受结构化 JSON caption（`dict`）或纯字符串。LingBot-Video 在结构化 caption 上训练，Pipeline 会自动对 `dict` 进行归一化。示例数据集中提供了发布版的结构化 caption（见下方[提示词改写](#提示词改写)）。
* `negative_prompt`: 描述不应出现内容的负向提示词。`pipe.default_negative_prompt` 提供了官方 T2V/V2V/TI2V 负向提示词；`pipe.default_negative_prompt_image` 是移除时序项后的 T2I 版本。
* `input_image`: 图生视频（TI2V）的首帧 PIL 图像。该帧被 VAE 编码成 clean latent，在每个采样步之后重新写入第一个时间槽，使模型只生成后续帧。T2V / V2V / T2I 时留 `None`。
* `input_video`: 视频到视频生成的输入视频（帧列表或 `VideoData`），与 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围 `[0, 1]`，默认 `1.0`。较小值保留更多输入视频结构。仅当 `input_video` 提供时生效。
* `height`: 视频 / 图像高度，默认 `480`，必须是 16 的倍数。
* `width`: 视频 / 图像宽度，默认 `480`，必须是 16 的倍数。
* `num_frames`: 帧数，默认 `81`，须满足 `4k+1`（VAE 时间上 4× 压缩）。文生图使用 `num_frames=1`。
* `cfg_scale`: 无分类器指导强度，默认 `3.0`。
* `num_inference_steps`: 推理步数，默认 `40`。
* `sigma_shift`: Flow-matching 时间步 shift，默认 `3.0`。
* `seed`: 随机种子，默认 `None`（完全随机）。
* `rand_device`: 生成初始噪声的设备，默认 `"cpu"`。
* `progress_bar_cmd`: 进度条，默认 `tqdm`，可设为 `lambda x: x` 关闭。

显存不足时请参考[显存管理](../Pipeline_Usage/VRAM_management.md)启用显存管理功能。我们在示例代码中提供了每个任务的推荐低显存配置，见上方"模型总览"中的表格。

### 提示词改写

LingBot-Video 训练时使用的是**结构化 JSON caption**，直接喂平铺句子属于分布外输入，会明显降低生成质量。Pipeline 接受 `dict` 形式的 caption（与训练一致的格式）或纯字符串，`dict` 会被内部归一化。

发布版的结构化 caption 已通过 `DiffSynth-Studio/diffsynth_example_dataset` 示例数据集提供（`t2v_example_*.json`、`ti2v_example.json`、`t2i_example.json`，推理示例脚本会自动下载）。用 `json.load` 读入后作为 `dict` 传入 Pipeline，或作为编写自定义 caption 的模板。

如需将一段简短描述改写为结构化 caption，可使用 `examples/lingbot_video/model_training/scripts/prompt_rewriter.py` 中的两阶段改写器：阶段 1 将想法扩展为自然语言描述，阶段 2 将其映射为结构化 JSON。改写器是**独立的 VLM + 阶段二 LoRA 适配器**，需要另外下载：

| 角色 | 模型 ID | 大小 |
|-|-|-|
| 改写器 base VLM（阶段 1 + 2） | [`Qwen/Qwen3.6-27B`](https://modelscope.cn/models/Qwen/Qwen3.6-27B) | ~55 GB |
| 改写器阶段二 LoRA 适配器 | [`Robbyant/lingbot-video-rewriter-lora`](https://modelscope.cn/models/Robbyant/lingbot-video-rewriter-lora) | ~0.5 GB |

```python
import os
os.environ["REWRITER_BASE_MODEL"] = "./models/Qwen/Qwen3.6-27B"
os.environ["REWRITER_ADAPTER"] = "./models/Robbyant/lingbot-video-rewriter-lora"

# 在仓库根目录下运行，此包式 import 才能解析。
from examples.lingbot_video.model_training.scripts.prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

除环境变量外，也可以直接向 `rewrite_prompt` 传 `base=` / `adapter=`；或者提供一个实现了 `generate(text, image, use_lora)` 方法的自定义对象作为 `backend=`，从而对接托管服务或 OpenAI-compatible 端点。

## 模型训练

LingBot-Video 系列模型统一通过 [`examples/lingbot_video/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 视频的高度，必须能被 16 整除。
        * `--width`: 视频的宽度，必须能被 16 整除。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数，须满足 `4k+1`。
* LingBot-Video 专有参数
    * `--processor_path`: Qwen3-VL processor 目录（或 `model_id:origin_file_pattern` 形式）路径，用于对 prompt 进行 tokenize。
    * `--first_frame_as_condition`: 启用图生视频（TI2V）的 LoRA / 全量训练。每段视频以自己的第一帧作为条件：该帧被 VAE 编码为 clean latent 固定到第一个时间槽（同时作为视觉输入送入 Qwen3-VL 文本编码器），并从 flow-matching 损失中排除。
    * `--max_timestep_boundary`: 训练时时间步的上边界，取值 `[0, 1]` 表示相对训练调度的比例。
    * `--min_timestep_boundary`: 训练时时间步的下边界，取值 `[0, 1]` 表示相对训练调度的比例。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

我们构建了一个样例数据集供您测试，可通过以下命令下载：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "lingbot_video/lingbot-video-dense-1.3b_t2v/*" --local_dir ./data/diffsynth_example_dataset
```

训练时 `prompt` 字段应存放**结构化 JSON caption**（与推理时使用的分布内格式一致）。如果数据集里存的是原始散文，可先使用 [`examples/lingbot_video/model_training/scripts/rewrite_captions.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/scripts/rewrite_captions.py) 离线改写一次。

我们为每个任务编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
