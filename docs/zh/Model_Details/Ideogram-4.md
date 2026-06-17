# Ideogram 4

Ideogram 4 是由 Ideogram 开源的图像生成模型。DiffSynth-Studio 支持 FP8 量化版本和 BF16 重打包版本的推理、低显存推理，以及全量训练和 LoRA 训练。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8) 模型并进行推理。最低 24G 显存即可运行。

```python
from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline
from diffsynth.core import ModelConfig
import torch


pipe = Ideogram4Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        # unconditional_transformer is optional. You can delete this line to reduce VRAM required.
        ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="unconditional_transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="tokenizer/"),
)
prompt = r"""
{
  "high_level_description": "A medium-shot photograph of Formula 1 driver Max Verstappen wearing his Red Bull Racing racing suit and cap, smiling as he holds his racing helmet and talks to a man in a white shirt and black vest at a race track.",
  "style_description": {
    "aesthetics": "saturated primary colors, rule of thirds, joyful and triumphant",
    "lighting": "overcast daylight, diffused, soft subtle shadows",
    "photo": "shallow depth of field, sharp focus, eye-level, telephoto",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "The background is an out-of-focus racing paddock or track environment. Several blurred figures are visible, including one in an orange shirt. A purple and white structure with a red 'F1' logo stands on the left. The scene is outdoors with daylight, though the sky is not visible.",
    "elements": [
      {"type": "obj", "bbox": [55, 642, 1000, 937], "desc": "An older man standing in profile, facing left toward Max Verstappen. He has grey hair and fair skin. He is wearing a white long-sleeved button-down shirt with a navy blue quilted vest over it. He has a slight smile."},
      {"type": "obj", "bbox": [34, 137, 1000, 617], "desc": "Max Verstappen, a fair-skinned male Formula 1 driver, positioned in the center. He is facing forward with a joyful expression and a slight smile. He wears a navy blue Red Bull Racing team uniform with numerous sponsor logos and a matching baseball cap with the number '1'. He is holding a white and red racing helmet in his hands. He has a silver watch on his left wrist."},
      {"type": "obj", "bbox": [422, 212, 792, 452], "desc": "Max Verstappen's racing helmet, held in front of his chest. It features a white, red, and yellow design with the Red Bull logo and the 'Player 0.0' branding. The visor is clear and open."},
      {"type": "text", "bbox": [657, 0, 755, 142], "text": "F1", "desc": "Large, stylized red logo on a black and purple background in the lower left."},
      {"type": "text", "bbox": [768, 0, 818, 147], "text": "Formula 1\nWorld Championship™", "desc": "Small white sans-serif text below the F1 logo on the left side."},
      {"type": "text", "bbox": [78, 447, 117, 510], "text": "ORACLE\nRed Bull\nRacing", "desc": "Very small white and orange logo on the front of the navy blue cap."},
      {"type": "text", "bbox": [78, 417, 120, 440], "text": "1", "desc": "Bold red numeral '1' on the front left side of the navy blue cap."},
      {"type": "text", "bbox": [332, 442, 363, 483], "text": "Red Bull", "desc": "Small yellow and red text logo on the collar of the uniform."},
      {"type": "text", "bbox": [373, 490, 423, 532], "text": "RAUCH", "desc": "Small yellow and blue logo on the right chest of the uniform."},
      {"type": "text", "bbox": [422, 473, 500, 532], "text": "BYBIT\nHONDA", "desc": "Medium-sized white sans-serif text on the right chest of the uniform."},
      {"type": "text", "bbox": [410, 203, 442, 257], "text": "RAUCH", "desc": "Small yellow logo on the left upper arm of the uniform."},
      {"type": "text", "bbox": [530, 448, 627, 510], "text": "Red Bull", "desc": "Medium red text logo on the right side of the torso, part of the Red Bull graphic."},
      {"type": "text", "bbox": [680, 417, 768, 523], "text": "Red Bull", "desc": "Large red text logo across the lower torso of the uniform."},
      {"type": "text", "bbox": [797, 475, 815, 518], "text": "MAX", "desc": "Small white text next to a Dutch flag on the belt area of the uniform."},
      {"type": "text", "bbox": [558, 317, 715, 355], "text": "Player 0.0", "desc": "Black sans-serif text on a white band on the racing helmet."},
      {"type": "text", "bbox": [560, 800, 582, 835], "text": "IA.COM", "desc": "Small blue sans-serif text on the right sleeve of the white shirt."},
      {"type": "text", "bbox": [968, 8, 997, 332], "text": "© Anadolu Agency via Getty Images", "desc": "Small white watermark text in the bottom left corner."}
    ]
  }
}
"""
image = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=48, cfg_scale=7.0, seed=42)
image.save("image_ideogram-4-fp8.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_inference/ideogram-4-fp8.py)|-|-|-|-|-|
|[DiffSynth-Studio/ideogram-4-bf16-repackage](https://www.modelscope.cn/models/DiffSynth-Studio/ideogram-4-bf16-repackage)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_inference/ideogram-4-bf16-repackage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_inference_low_vram/ideogram-4-bf16-repackage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_training/full/Ideogram-4-bf16-repackage.sh)|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_training/lora/Ideogram-4-bf16-repackage.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_training/validate_lora/Ideogram-4-bf16-repackage.py)|

## 模型推理

模型通过 `Ideogram4Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`Ideogram4Pipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述画面中出现的内容。Ideogram 4 支持结构化的 JSON 格式提示词，包含高层描述、风格描述和构图解构等信息。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 7.0。
* `input_image`: 输入图像，用于图生图，该参数与 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围是 0～1，默认值为 1。当数值接近 0 时，生成图像与输入图像相似；当数值接近 1 时，生成图像与输入图像相差更大。在不输入 `input_image` 参数时，请不要将其设置为非 1 的数值。
* `height`: 图像高度，需保证高度为 16 的倍数，默认值为 1024。
* `width`: 图像宽度，需保证宽度为 16 的倍数，默认值为 1024。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。
* `num_inference_steps`: 推理次数，默认值为 50。

## 模型训练

ideogram4 系列模型统一通过 `examples/ideogram4/model_training/train.py` 进行训练，脚本的参数包括：

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
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* Ideogram-4 专有参数
    * `--tokenizer_path`: Tokenizer 路径。默认从 `ideogram-ai/ideogram-4-fp8` 下载。

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
