# Fibo 2.0

Fibo 2.0 是由 Bria AI 开发的 80 亿参数统一图像生成与编辑模型。它在 FLUX.2 VAE 的隐空间上运行，使用 Qwen3-VL-4B 编码提示词和参考图像，主干为 44 层的单流 DiT。文生图和图像编辑共享同一套权重。

Fibo 2.0 Dev 是完整、非蒸馏的基础模型，使用 classifier-free guidance，适合微调。Fibo 2.0 Edit Turbo 是蒸馏后的编辑专用模型，支持 4 步、无 guidance 推理。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev) 模型并进行文生图推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载。

```python
import json
import torch
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig

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
pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
caption = {
    "short_description": "一只折纸鲸鱼在纸云中游动。",
    "objects": [{"description": "一只由光泽纸张折叠而成的蓝色折纸鲸鱼", "location": "画面中央"}],
    "background_setting": "由层叠白色纸云构成的明亮天空",
    "style_medium": "纸艺摄影",
}
prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
image = pipe(prompt=prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=5.0)
image.save("image.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference/Fibo-2.0-dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference_low_vram/Fibo-2.0-dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py)|
|[briaai/Fibo-2.0-edit-turbo](https://www.modelscope.cn/models/briaai/Fibo-2.0-edit-turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference/Fibo-2.0-edit-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference_low_vram/Fibo-2.0-edit-turbo.py)|-|-|-|-|

表格中的训练列指向文生图脚本，Fibo 2.0 Dev 还支持图像编辑训练，见[模型训练](#模型训练)。

## 模型推理

模型通过 `FiboPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`from_pretrained` 还接受：

* `processor_config`: Qwen3-VL processor 的 `ModelConfig`。
* `font_config`: 绘制 `text_render` 文字图像所用字体的 `ModelConfig`。可以指向字体目录、单个字体文件或字体文件列表。

`FiboPipeline` 推理的输入参数包括：

* `prompt`: 结构化 JSON 描述、普通描述，或编辑指令。编辑时也可以是包含 `edit_instruction` 字段的结果描述 JSON。
* `negative_prompt`: 当 `cfg_scale > 1` 时使用的负向提示词，默认为 Fibo 2.0 训练时的空描述。
* `edit_image`: 待编辑图像或参考图像列表，文生图时留空。
* `mask`: 单图编辑时的掩码，白色区域会被重绘。
* `text_images`: 描述中 `text_render` 条目对应的文字图像；不传时自动绘制。
* `fonts`: 绘制内置字体缺失字符时使用的字体路径。
* `render_text_images`: 是否自动绘制文字图像，默认 `True`。
* `height`、`width`: 输出尺寸，需为 16 的倍数。编辑时留空则按第一张图像的宽高比选择训练尺寸。
* `cfg_scale`: Classifier-free guidance 参数，默认 5.0。
* `num_inference_steps`: 推理步数，默认 30。
* `denoising_strength`: 去噪强度，范围为 0~1，默认 1。数值越低，起始噪声水平越低。
* `seed`、`rand_device`: 随机种子与生成噪声的设备。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文"模型总览"中的表格。

### 图像编辑

`edit_image` 支持单张图像，也支持最多 5 张图像的多参考编辑。下面的示例使用蒸馏后的 `Fibo-2.0-edit-turbo` 模型：

```python
from PIL import Image
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig

pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="fonts/"),
)
source = Image.open("photo.jpg").convert("RGB")
image = pipe(prompt="把夹克换成红色皮夹克", edit_image=source, seed=42, num_inference_steps=4, cfg_scale=1.0)
image.save("edited.jpg")
```

`Fibo-2.0-dev` 使用 50 步、`cfg_scale=5.0`；`Fibo-2.0-edit-turbo` 使用 4 步、`cfg_scale=1.0`。

### 掩码编辑

单图编辑时，可以同时传入 `edit_image` 和 `mask`（白色区域为需要重绘的区域）：

```python
mask = Image.open("mask.png").convert("L")  # 白色重绘，黑色保留
image = pipe(
    prompt="把行李箱换成复古棕色皮箱",
    edit_image=source, mask=mask, seed=42, num_inference_steps=4, cfg_scale=1.0,
)
```

### 文字渲染

Fibo 会读取结构化描述中的 `text_render` 字段，把每条文字绘制成白底黑字的小图，再与提示词一起送入 Qwen3-VL：

```python
caption = {
    "short_description": "一位少女举着写有 \"ModelScope\" 的白色牌子。",
    "objects": [{"description": "一块写有粗体黑色文字的白色牌子", "location": "画面中央"}],
    "style_medium": "摄影",
    "text_render": [{"text": "ModelScope"}],
}
image = pipe(
    prompt=json.dumps(caption, separators=(",", ":"), ensure_ascii=False),
    height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=5.0,
)
```

`font_config` 用于加载模型自带的字体。如果要渲染这些字体不包含的字符，可通过 `fonts` 额外传入字体路径：

```python
image = pipe(prompt=prompt, fonts=["/path/to/a/cjk-font.ttf"], ...)
```

所有字体都不包含的字符会使用 Pillow 内置字体绘制。文字渲染的准确率取决于模型本身，非拉丁文字的偏差通常更明显。

## 模型训练

Fibo 2.0 Dev 通过 [`examples/fibo/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"briaai/Fibo-2.0-dev:transformer/*.safetensors"`。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，例如训练图像编辑模型时的 `edit_image` 和 `mask`，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
        * `--quant_options`: 对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
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
        * `--height`: 图像的高度。留空启用动态分辨率。
        * `--width`: 图像的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
* Fibo 2.0 专有参数
    * `--processor_path`: Qwen3-VL processor 的路径或 `model_id:origin_file_pattern`，留空则自动从远程下载。
    * `--font_path`: 文字渲染字体的路径或 `model_id:origin_file_pattern`，留空则自动从远程下载。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

我们构建了样例数据集，包括文生图数据集和图像编辑数据集，通过以下命令可以下载：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "fibo/*" --local_dir ./data/diffsynth_example_dataset
```

* `data/diffsynth_example_dataset/fibo/Fibo-2.0-dev`：文生图，`metadata.csv` 包含 `image` 和 `prompt`。
* `data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit`：图像编辑，`metadata.json` 包含 `image`、`prompt`、`edit_image`，以及可选的 `mask`。

Fibo 2.0 Dev 同时支持文生图训练和图像编辑训练。训练复用推理 Pipeline：VAE 和文本编码器冻结，DiT 使用 flow matching 训练。全量训练使用 `--trainable_models "dit"`，LoRA 训练使用 `--lora_base_model "dit"`。图像编辑训练时，`edit_image` 和 `mask` 通过 `UnifiedDataset` 加载，并由 `--extra_inputs` 注入 Pipeline。

| 模式 | 全量训练 | LoRA 训练 | 全量训练后验证 | LoRA 训练后验证 |
|-|-|-|-|-|
| 文生图 | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py) |
| 图像编辑 | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-Edit.sh) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-Edit.sh) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-Edit.py) | [脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-Edit.py) |

全量训练脚本使用 `--config_file examples/fibo/model_training/full/accelerate_config.yaml`；LoRA 脚本默认单卡运行。`mask` 是可选字段，缺少 `mask` 的样本会作为普通编辑样本训练。Fibo 2.0 Edit Turbo 不用于微调，请使用 Fibo 2.0 Dev 训练。

我们为每个模型编写了推荐的训练脚本，请参考上文中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。

## 许可协议

Fibo 2.0 以 CC BY-NC 4.0 许可证发布，仅限非商业用途。商业用途需与 Bria AI 另行签署协议。
