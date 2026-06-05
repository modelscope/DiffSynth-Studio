# Ideogram 4

Ideogram 4 是由 Ideogram 开源的图像生成模型，由于目前仅开源了量化版本，DiffSynth-Studio 对其提供有限支持，目前仅支持推理。

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
