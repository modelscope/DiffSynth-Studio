# 翻译、润色——提示词的魔法

在生成图像时，我们需要编写提示词，用来描述图像的内容。提示词会直接影响生成的效果，但提示词的编写也是一门学问，好的提示词可以生成具有高度美感的图像，我们提供了一系列模型来帮助用户处理提示词。

## 翻译

目前大多数文生图模型都是只支持英文提示词的，对于非英文母语的用户，使用起来有些困难，我们可以使用开源的翻译模型把提示词翻译成英文。在下面这个例子中，我们以“一个女孩”为提示词，使用模型 opus-mt-zh-en（可在 [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) 或 [ModelScope](https://modelscope.cn/models/moxying/opus-mt-zh-en) 下载）进行翻译。

```python
from diffsynth import ModelManager, SDXLImagePipeline, Translator
import torch

model_manager = ModelManager(
    torch_dtype=torch.float16, device="cuda",
    model_id_list=["BluePencilXL_v200", "opus-mt-zh-en"]
)
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[Translator])

torch.manual_seed(0)
prompt = "一个女孩"
image = pipe(
    prompt=prompt, negative_prompt="",
    height=1024, width=1024, num_inference_steps=30
)
image.save("image_1.jpg")
```

## 润色

详细的提示词可以生成细节更丰富的图像，我们可以使用提示词润色模型 BeautifulPrompt（可在 [HuggingFace](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd) 或 [ModelScope](https://modelscope.cn/models/AI-ModelScope/pai-bloom-1b1-text2prompt-sd) 下载）对简单的提示词进行润色，这个模型能够让整体画面风格更加华丽。

这个模块可以和翻译模块同时启用，但请注意顺序，先翻译，后润色。

```python
from diffsynth import ModelManager, SDXLImagePipeline, Translator, BeautifulPrompt
import torch

model_manager = ModelManager(
    torch_dtype=torch.float16, device="cuda",
    model_id_list=["BluePencilXL_v200", "opus-mt-zh-en", "BeautifulPrompt"]
)
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[Translator, BeautifulPrompt])

torch.manual_seed(0)
prompt = "一个女孩"
image = pipe(
    prompt=prompt, negative_prompt="",
    height=1024, width=1024, num_inference_steps=30
)
image.save("image_2.jpg")
```

我们还内置了一个通义千问模型，这个模型可以一步到位地完成提示词的翻译和润色工作。

```python
from diffsynth import ModelManager, SDXLImagePipeline, QwenPrompt
import torch

model_manager = ModelManager(
    torch_dtype=torch.float16, device="cuda",
    model_id_list=["BluePencilXL_v200", "QwenPrompt"]
)
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[QwenPrompt])

torch.manual_seed(0)
prompt = "一个女孩"
image = pipe(
    prompt=prompt, negative_prompt="",
    height=1024, width=1024, num_inference_steps=30
)
image.save("image_3.jpg")
```
