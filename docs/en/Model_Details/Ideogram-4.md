# Ideogram 4

Ideogram 4 is an image generation model open-sourced by Ideogram. Since only the quantized version has been released so far, DiffSynth-Studio provides limited support for it, currently supporting inference only.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Install Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8) model for inference. A minimum of 24GB VRAM is required to run.

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

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[ideogram-ai/ideogram-4-fp8](https://www.modelscope.cn/models/ideogram-ai/ideogram-4-fp8)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ideogram4/model_inference/ideogram-4-fp8.py)|-|-|-|-|-|

## Model Inference

The model is loaded via `Ideogram4Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `Ideogram4Pipeline` inference include:

* `prompt`: Prompt describing the content appearing in the image. Ideogram 4 supports structured JSON format prompts, including high-level description, style description, and compositional deconstruction.
* `negative_prompt`: Negative prompt describing content that should not appear in the image, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 7.0.
* `input_image`: Input image for image-to-image generation, used in conjunction with `denoising_strength`.
* `denoising_strength`: Denoising strength, range is 0~1, default value is 1. When the value approaches 0, the generated image is similar to the input image; when the value approaches 1, the generated image differs more from the input image. When `input_image` parameter is not provided, do not set this to a non-1 value.
* `height`: Image height, must be a multiple of 16, default value is 1024.
* `width`: Image width, must be a multiple of 16, default value is 1024.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Computing device for generating random Gaussian noise matrix, default is `"cpu"`.
* `num_inference_steps`: Number of inference steps, default value is 50.
