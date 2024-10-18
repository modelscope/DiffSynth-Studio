# Translation and Polishing — The Magic of Prompt Words

When generating images, we need to write prompt words to describe the content of the image. Prompt words directly affect the outcome of the generation, but crafting them is also an art. Good prompt words can produce images with a high degree of aesthetic appeal. We offer a range of models to help users handle prompt words effectively.

## Translation

Most text-to-image models currently only support English prompt words, which can be challenging for users who are not native English speakers. To address this, we can use open-source translation models to translate the prompt words into English. In the following example, we take "一个女孩" (a girl) as the prompt word and use the model opus-mt-zh-en for translation(which can be downloaded from [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) or [ModelScope](https://modelscope.cn/models/moxying/opus-mt-zh-en)).
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

![image_1](https://github.com/user-attachments/assets/c8070a6b-3d2f-4faf-a806-c403b91f1a94)

## Polishing

Detailed prompt words can generate images with richer details. We can use a prompt polishing model like BeautifulPrompt(which can be downloaded from [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en) or [ModelScope](https://modelscope.cn/models/moxying/opus-mt-zh-en)) to embellish simple prompt words. This model can make the overall picture style more gorgeous.

This module can be activated simultaneously with the translation module, but please pay attention to the order: translate first, then polish.

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

![image_2](https://github.com/user-attachments/assets/94f64a7d-b14a-41e2-a013-c9a74635a84d)

We have also integrated a Tongyi Qwen model that can seamlessly complete the translation and polishing of prompt words in one step.

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

![image_3](https://github.com/user-attachments/assets/fc1a201d-aef1-4e6a-81d6-2e2249ffa230)
