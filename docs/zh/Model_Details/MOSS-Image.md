# MOSS-Image

MOSS-Image 是 OpenMOSS 开源的 5B 图像生成模型系列，采用单流 DiT 主干网络搭配 Qwen3.5 文本编码器，支持自然语言提示词、标签序列与 JSON 布局三种输入形式，可精确控制文字位置与物体排布，在中英日文文字渲染、动漫与写实风格生成上均有稳定表现。Base 为基础模型，适合社区在其上进行 LoRA 微调；SFT 在 Base 的基础上使用精选数据微调，文字分组与版面组织更清晰。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [openmoss/MOSS-Image-Base](https://www.modelscope.cn/models/openmoss/MOSS-Image-Base) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载。

```python
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
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
vae_vram_config = {**vram_config, "offload_dtype": torch.float32, "onload_dtype": torch.float32, "preparing_dtype": torch.float32, "computation_dtype": torch.float32}
pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vae_vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
image.save("image_MOSS-Image-Base.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[openmoss/MOSS-Image-Base](https://www.modelscope.cn/models/openmoss/MOSS-Image-Base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/full/MOSS-Image-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_full/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-Base.py)|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/full/MOSS-Image-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_full/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-SFT.py)|
|[DiffSynth-Studio/MOSS-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MOSS-Image-EntroPack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-entropack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-entropack.py)|

## 模型推理

模型通过 `MOSSImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`MOSSImagePipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述画面中出现的内容。除自然语言与标签序列外，也可以传入 JSON 字符串形式的布局描述，用于精确控制文字与物体的位置。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 3.0。
* `max_sequence_length`: 文本编码的最大 token 数，默认值为 4096。
* `uncond_skip_layers`: 负向分支中跳过计算的 DiT 层索引，默认值为 `(22,)`。跳过后可减少一半的负向分支计算量；置为 `()` 则关闭跳层。
* `height`: 图像高度，需保证高度为 16 的倍数，默认值为 1024。
* `width`: 图像宽度，需保证宽度为 16 的倍数，默认值为 1024。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。
* `num_inference_steps`: 推理步数，默认值为 50。
* `sigma_shift`: Flow Matching 时间表的 shift 系数，默认值为 5.0。
* `progress_bar_cmd`: 包装推理时间步的进度条函数，默认使用 `tqdm`。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文"模型总览"中的表格。

## 模型训练

MOSS-Image 系列模型统一通过 [`examples/moss_image/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"openmoss/MOSS-Image-Base:transformer/diffusion_pytorch_model*.safetensors"`。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
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
* MOSS-Image 专有参数
    * `--tokenizer_path`: tokenizer 的路径，留空则自动从远程下载。

我们构建了一个样例图像数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
