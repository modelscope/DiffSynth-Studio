# EliGen: Entity-Level Controlled Image Generation

## Introduction

We propose EliGen, a novel approach that leverages fine-grained entity-level information to enable precise and controllable text-to-image generation. EliGen excels in tasks such as entity-level controlled image generation and image inpainting, while its applicability is not limited to these areas. Additionally, it can be seamlessly integrated with existing community models, such as the IP-Adpater.

* Paper: Comming soon
* Github: https://github.com/modelscope/DiffSynth-Studio
* Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen)
* Training dataset: Coming soon

## Methodology

![regional-attention](https://github.com/user-attachments/assets/9a147201-15ab-421f-a6c5-701075754478)

We introduce a regional attention mechanism within the DiT framework to effectively process the conditions of each entity. This mechanism enables the local prompt associated with each entity to semantically influence specific regions through regional attention. To further enhance the layout control capabilities of EliGen, we meticulously curate an entity-annotated dataset and fine-tune the model using the LoRA framework. 

1. **Regional Attention**: Regional attention is shown in above figure, which can be easily applied to other text-to-image models. Its core principle involves transforming the positional information of each entity into an attention mask, ensuring that the mechanism only affects the designated regions.
   
2. **Dataset with Entity Annotation**: To curate a dedicated entity control dataset, we start by randomly selecting captions from DiffusionDB and generating the corresponding source image using Flux. Next, we employ Qwen2-VL 72B, recognized for its advanced grounding capabilities among MLLMs, to randomly identify entities within the image. These entities are annotated with local prompts and bounding boxes for precise localization, forming the foundation of our dataset for further training.

3. **Training**: We apply LoRA and deepspeed to finetune regional attention with curated dataset, enabling our EliGen performing effective entity-level control.

## Usage
1. **Entity-Level Controlled Image Generation**
See [./entity_control.py](./entity_control.py) for usage.
2. **Image Inpainting**
   To apply EliGen to image inpainting task, we propose a inpainting fusion pipeline to preserve the non-painting areas while enabling precise, entity-level modifications over inpaining regions.
   See [./entity_inpaint.py](./entity_inpaint.py) for usage.
3. **Styled Entity Control**
   EliGen can be seamlessly integrated with existing community models. We have provided an example of how to integrate it with the IP-Adpater. See [./entity_control_ipadapter.py](./entity_control_ipadapter.py) for usage.
4. **Play with EliGen using UI**
   Download the checkpoint of EliGen from [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen) to `models/lora/entity_control` and run the following command to try interactive UI: 
   ```bash
   python apps/gradio/entity_level_control.py
   ```
## Examples
### Entity-Level Controlled Image Generation

1. The effect of generating images with continuously changing entity positions.

https://github.com/user-attachments/assets/4fc76df1-b26a-46e8-a950-865cdf02a38d

2. The image generation effect of complex Entity combinations, demonstrating the strong generalization of EliGen.

|![image_1_base](https://github.com/user-attachments/assets/b8564b28-19b5-424f-bf3c-6476f2923ff9)|![image_1_base](https://github.com/user-attachments/assets/20793715-42d3-46f7-8d62-0cb4cacef38d)|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/70ef12fe-d300-4b52-9d11-eabc9b5464a8)|![image_1_enhance](https://github.com/user-attachments/assets/7645ce0e-4aa7-4b1e-b7a7-bccfd9796461)|
|![image_2_base](https://github.com/user-attachments/assets/2f1e44e1-8f1f-4c6e-ab7a-1b6861a33a69)|![image_2_enhance](https://github.com/user-attachments/assets/faf78498-57ba-41bd-b516-570c86984515)|
|![image_3_base](https://github.com/user-attachments/assets/206d1cef-2e96-4469-aed5-cdeb06ab9e99)|![image_3_enhance](https://github.com/user-attachments/assets/75d784d6-d5a1-474f-a5d5-ef8074135f35)|
### Image Inpainting
|Inpainting Input|Inpainting Output|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/5f74c710-bf30-4db1-ae40-a1e1995ccef6)|![image_1_enhance](https://github.com/user-attachments/assets/1cd71177-e956-46d3-86ce-06f774c96efd)|
|![image_2_base](https://github.com/user-attachments/assets/5ef499f3-3d8a-49cc-8ceb-86af7f5cb9f8)|![image_2_enhance](https://github.com/user-attachments/assets/fb967035-7b28-466c-a753-c00135559121)|
### Styled Entity Control
|Style Reference|Entity Control Variance 1|Entity Control Variance 2|Entity Control Variance 3|
|-|-|-|-|
|![image_1_base](https://github.com/user-attachments/assets/5e2dd3ab-37d3-4f58-8e02-ee2f9b238604)|![image_1_enhance](https://github.com/user-attachments/assets/0f6711a2-572a-41b3-938a-95deff6d732d)|![image_1_enhance](https://github.com/user-attachments/assets/ce2e66e5-1fdf-44e8-bca7-555d805a50b1)|![image_1_enhance](https://github.com/user-attachments/assets/ad2da233-2f7c-4065-ab57-b2d84dc2c0e2)|
|![image_2_base](https://github.com/user-attachments/assets/77cf7ceb-48e3-442d-8ffc-5fa4a10fe81a)|![image_2_enhance](https://github.com/user-attachments/assets/59a4f3c2-e59d-40c7-886c-0768f14fcc89)|![image_2_enhance](https://github.com/user-attachments/assets/a9187fb0-489a-49c9-a52f-56b1bd96faf7)|![image_2_enhance](https://github.com/user-attachments/assets/a62caee4-3863-4b56-96ff-e0785c6d93bb)|