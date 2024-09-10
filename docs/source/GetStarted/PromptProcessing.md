# Prompt Processing



# 提示词（Prompt）处理

DiffSynth 内置了提示词处理功能，分为：

- **提示词润色器（`prompt_refiner_classes`）**：包括提示词润色、提示词中译英、提示词同时润色与中译英，可选参数如下：

    - **英文提示词润色**：'BeautifulPrompt'，使用到的是[pai-bloom-1b1-text2prompt-sd](https://modelscope.cn/models/AI-ModelScope/pai-bloom-1b1-text2prompt-sd)。

    - **提示词中译英**：'Translator'，使用到的是[opus-mt-zh-e](https://modelscope.cn/models/moxying/opus-mt-zh-en)。

    - **提示词中译英并润色**：'QwenPrompt'，使用到的是[Qwen2-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct)。

- **提示词扩展器（`prompt_extender_classes`）**：基于Omost的提示词分区控制扩写，可选参数如下：

    - **提示词分区扩写**：'OmostPromter'。


## 使用说明

### 提示词润色器

在加载模型 Pipeline 时，可以通过参数 `prompt_refiner_classes` 指定所需的提示词润色器功能。有关示例代码，请参考 [sd_prompt_refining.py](examples/image_synthesis/sd_prompt_refining.py)。

可选的 `prompt_refiner_classes` 参数包括：Translator、BeautifulPrompt、QwenPrompt。

```python
pipe = SDXLImagePipeline.from_model_manager(model_manager, prompt_refiner_classes=[Translator, BeautifulPrompt])
```

### 提示词扩展器

在加载模型 Pipeline 时，可以通过参数 `prompt_extender_classes` 指定所需的提示词扩展器。有关示例代码，请参考 [omost_flux_text_to_image.py](examples/image_synthesis/omost_flux_text_to_image.py)。

```python
pipe = FluxImagePipeline.from_model_manager(model_manager, prompt_extender_classes=[OmostPromter])
```

