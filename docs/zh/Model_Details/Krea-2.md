# Krea-2

Krea-2 是由 Krea 团队开发的图像生成模型。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 24G 显存即可运行。

```python
from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
import torch

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
pipe = Krea2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/Krea-2-Raw", origin_file_pattern="raw.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 1,
)
prompt = "A cat standing on a stone."
image = pipe(prompt, seed=0, num_inference_steps=52, cfg_scale=4.5)
image.save("image.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference_low_vram/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/full/Krea-2-Raw.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_full/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/lora/Krea-2-Raw.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_lora/Krea-2-Raw.py)|
|[krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/full/Krea-2-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_full/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/lora/Krea-2-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py)|

## 模型推理

模型通过 `Krea2Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`Krea2Pipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述要生成的图像内容，默认值为 `""`。
* `negative_prompt`: 负向提示词，描述图像中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 3.5。
* `height`: 图像高度，需保证为 16 的倍数，默认值为 1024。
* `width`: 图像宽度，需保证为 16 的倍数，默认值为 1024。
* `seed`: 随机种子，默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。
* `num_inference_steps`: 推理次数，默认值为 52。
* `mu`: 时间步动态位移参数，默认为 `None`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文"模型总览"中的表格。

## 模型训练

Krea-2 系列模型统一通过 [`examples/krea2/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/train.py) 进行训练，脚本的参数包括：

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
        * `--height`: 图像的高度。留空启用动态分辨率。
        * `--width`: 图像的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
* Krea-2 专有参数
    * `--tokenizer_path`: tokenizer 的路径，留空则自动从远程下载。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。
    * `--align_to_opensource_format`: 是否将 LoRA 格式对齐为开源格式，适用于与其他框架兼容的 LoRA 模型。

我们构建了样例数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "krea2/*" --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
