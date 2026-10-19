# FLUX.3

FLUX.3 Image Klein 是 Black Forest Labs 开源的 5.6B 图像生成模型，在同一模型中支持文生图、最多 10 张参考图像的编辑和边界框布局控制。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 模型访问权限

`black-forest-labs/flux-3-image-klein` 和 `black-forest-labs/flux-3-image-klein-base` 是 ModelScope 上的受限模型，需要申请访问后才能下载。下载前请：

1. 在模型页面申请访问（[Klein](https://modelscope.cn/models/black-forest-labs/flux-3-image-klein) 或 [Klein Base](https://modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)）。
2. 使用 ModelScope token 登录：`modelscope login --token ms-xxx`。token 可在[访问令牌](https://modelscope.cn/my/settings/token)页面获取。

请确保申请访问的账号与登录使用的账号一致。

## 快速开始

运行以下代码可以快速加载 [black-forest-labs/flux-3-image-klein](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 4G 显存即可运行。

```python
from PIL import Image
import torch

from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / 1024**3 - 0.5,
)

prompt = "Masterpiece, best quality. Cinematic photograph of a beautiful young woman with long black hair wearing a navy sailor-style school uniform with a green necktie, standing in a sunlit school corridor with large windows, warm afternoon light and soft bokeh in the background."
image_1 = pipe(prompt=prompt, seed=0, num_inference_steps=16, cfg_scale=1.0)
image_1.save("image1.png")

prompt = (
    "Masterpiece, best quality. Cinematic photograph of <girl_1>, a cheerful young woman with a short auburn bob "
    "wearing a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing on a leafy campus path at golden hour, "
    "a brown leather satchel <bag_1> resting on the ground at frame-left and a vintage bicycle <bicycle_1> parked at frame-right, soft bokeh in the background. "
    '[{"id":"girl_1","bbox":[60,330,980,690],"desc":"Cheerful young woman with a short auburn bob, a white short-sleeved blouse and a deep red pleated skirt with red Mary Jane shoes, standing facing the camera."},'
    '{"id":"bag_1","bbox":[740,100,940,360],"desc":"Brown leather school satchel with brass buckles and a shoulder strap, resting on the leaf-covered ground."},'
    '{"id":"bicycle_1","bbox":[380,720,880,980],"desc":"Vintage bicycle with a dark green frame, cream tires and a woven front basket, leaning on its kickstand."}]'
)
image_2 = pipe(prompt=prompt, seed=1, num_inference_steps=16, cfg_scale=1.0)
image_2.save("image2.png")

prompt = "Masterpiece, best quality. Cinematic photograph of the two young women from the two reference images standing side by side at the same scale in a sunlit school courtyard, the long-black-haired woman in the navy sailor school uniform and the short-auburn-haired woman in the white blouse and deep red pleated skirt, warm afternoon light, keeping both women's hairstyles and outfits unchanged."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt=prompt, edit_image=edit_image, seed=2, num_inference_steps=16, cfg_scale=1.0)
image_3.save("image3.png")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[black-forest-labs/flux-3-image-klein](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/full/flux-3-image-klein.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_full/flux-3-image-klein.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein.py)|
|[black-forest-labs/flux-3-image-klein-base](https://www.modelscope.cn/models/black-forest-labs/flux-3-image-klein-base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/full/flux-3-image-klein-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_full/flux-3-image-klein-base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein-base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein-base.py)|
|[DiffSynth-Studio/flux-3-image-klein-entropack](https://www.modelscope.cn/models/DiffSynth-Studio/flux-3-image-klein-entropack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference/flux-3-image-klein-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_inference_low_vram/flux-3-image-klein-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/lora/flux-3-image-klein-entropack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/validate_lora/flux-3-image-klein-entropack.py)|

## 模型推理

模型通过 `Flux3ImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`Flux3ImagePipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述目标图像、编辑要求或布局；默认值为 `""`。
* `negative_prompt`: 负向提示词，Base 在启用 CFG 时使用；默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 强度，默认值为 1.0。Distilled 必须使用 1.0；Base 示例使用 3.5。
* `input_image`: 图生图输入图像；留空时从纯噪声开始生成。
* `denoising_strength`: 图生图去噪强度，范围为 0 到 1，默认值为 1.0。
* `edit_image`: 编辑参考图像，支持单张 `PIL.Image` 或图像列表；最多支持 10 张参考图像。
* `height`: 输出图像高度，默认值为 1024，并对齐到 16 的倍数。
* `width`: 输出图像宽度，默认值为 1024，并对齐到 16 的倍数。
* `seed`: 随机种子，默认值为 `None`。
* `rand_device`: 生成初始高斯噪声的设备，默认值为 `"cpu"`。
* `num_inference_steps`: 推理步数，默认值为 16。Distilled 示例使用 16 步，Base 示例使用 50 步。
* `alpha`: FLUX.3 flow-matching 时间表的 rational shift 系数，默认值为 6.93。
* `progress_bar_cmd`: 包装推理时间步的进度条函数，默认使用 `tqdm`。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，推荐配置见前文“模型总览”中的低显存脚本。

## 模型训练

FLUX.3 系列模型统一通过 [`examples/flux3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux3/model_training/train.py) 进行训练，脚本的参数包括：

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
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
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
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* FLUX.3 专有参数
    * `--processor_path`: processor 的本地路径或模型 ID；留空时使用 Base 模型的 `text_encoder/`。
    * `--initialize_model_on_cpu`: 在 CPU 上初始化模型，以降低分布式训练启动阶段的显存峰值。

我们构建了一个样例图像数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
