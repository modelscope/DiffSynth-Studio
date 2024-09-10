# Prompt Processing

DiffSynth includes prompt processing functionality, which is divided into:

- **Prompt Refiners (`prompt_refiner_classes`)**: Includes prompt refinement, prompt translation from Chinese to English, and both refinement and translation of prompts. Available parameters are as follows:

    - **English Prompt Refinement**: 'BeautifulPrompt', using the model [pai-bloom-1b1-text2prompt-sd](https://modelscope.cn/models/AI-ModelScope/pai-bloom-1b1-text2prompt-sd).

    - **Prompt Translation from Chinese to English**: 'Translator', using the model [opus-mt-zh-e](https://modelscope.cn/models/moxying/opus-mt-zh-en).

    - **Prompt Translation and Refinement**: 'QwenPrompt', using the model [Qwen2-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct).

- **Prompt Extenders (`prompt_extender_classes`)**: Based on Omost's prompt partition control expansion. Available parameter is:

    - **Prompt Partition Expansion**: 'OmostPromter'.

## Usage Instructions

### Prompt Refiners

When loading the model pipeline, you can specify the desired prompt refiner functionality using the `prompt_refiner_classes` parameter. For example code, refer to [sd_prompt_refining.py](examples/image_synthesis/sd_prompt_refining.py).

Available `prompt_refiner_classes` parameters include: Translator, BeautifulPrompt, QwenPrompt.

```python
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[Translator, BeautifulPrompt])
```

### Prompt Extenders

When loading the model pipeline, you can specify the desired prompt extender using the prompt_extender_classes parameter. For example code, refer to [omost_flux_text_to_image.py](examples/image_synthesis/omost_flux_text_to_image.py).

```python
pipe = FluxImagePipeline.from_model_manager(model_manager, prompt_extender_classes=[OmostPromter])
```
