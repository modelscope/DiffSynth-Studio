# EliGen: Entity-Level Controlled Image Generation

## Introduction

We propose EliGen, a novel approach that leverages fine-grained entity-level information to enable precise and controllable text-to-image generation. EliGen excels in tasks such as entity-level controlled image generation and image inpainting, while its applicability is not limited to these areas. Additionally, it can be seamlessly integrated with existing community models, such as the IP-Adpater and In-Cotext LoRA.

* Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
* Github: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen)
* Training dataset: Coming soon

## Methodology

![regional-attention](https://github.com/user-attachments/assets/9a147201-15ab-421f-a6c5-701075754478)

We introduce a regional attention mechanism within the DiT framework to effectively process the conditions of each entity. This mechanism enables the local prompt associated with each entity to semantically influence specific regions through regional attention. To further enhance the layout control capabilities of EliGen, we meticulously contribute an entity-annotated dataset and fine-tune the model using the LoRA framework. 

1. **Regional Attention**: Regional attention is shown in above figure, which can be easily applied to other text-to-image models. Its core principle involves transforming the positional information of each entity into an attention mask, ensuring that the mechanism only affects the designated regions.
   
2. **Dataset with Entity Annotation**: To construct a dedicated entity control dataset, we start by randomly selecting captions from DiffusionDB and generating the corresponding source image using Flux. Next, we employ Qwen2-VL 72B, recognized for its advanced grounding capabilities among MLLMs, to randomly identify entities within the image. These entities are annotated with local prompts and bounding boxes for precise localization, forming the foundation of our dataset for further training.

3. **Training**: We utilize LoRA (Low-Rank Adaptation) and DeepSpeed to fine-tune regional attention mechanisms using a curated dataset, enabling our EliGen model to achieve effective entity-level control.

## Usage
1. **Entity-Level Controlled Image Generation**
   EliGen achieves effective entity-level control results. See [./entity_control.py](./entity_control.py) for usage.
2. **Image Inpainting**
   To apply EliGen to image inpainting task, we propose a inpainting fusion pipeline to preserve the non-painting areas while enabling precise, entity-level modifications over inpaining regions.
   See [./entity_inpaint.py](./entity_inpaint.py) for usage.
3. **Styled Entity Control**
   EliGen can be seamlessly integrated with existing community models. We have provided an example of how to integrate it with the IP-Adpater. See [./entity_control_ipadapter.py](./entity_control_ipadapter.py) for usage.
4. **Entity Transfer**
   We have provided an example of how to integrate EliGen with In-Cotext LoRA, which achieves interesting entity transfer results. See [./entity_transfer.py](./entity_transfer.py) for usage.
5. **Play with EliGen using UI**
   Download the checkpoint of EliGen from [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen) to `models/lora/entity_control` and run the following command to try interactive UI: 
   ```bash
   python apps/gradio/entity_level_control.py
   ```
## Examples
### Entity-Level Controlled Image Generation

1. The effect of generating images with continuously changing entity positions.

https://github.com/user-attachments/assets/4fc76df1-b26a-46e8-a950-865cdf02a38d

2. The image generation effect of complex Entity combinations, demonstrating the strong generalization of EliGen. See [./entity_control.py](./entity_control.py) `example_1-6` for generation prompts.

|![image_1_base](https://github.com/user-attachments/assets/4b9fb79f-cb3c-45a5-8d22-14e52865387c)|![image_1_base](https://github.com/user-attachments/assets/2e60e51b-f8d5-4b25-ae21-f64531e20b1b)|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/5bdfebc2-8c1e-4619-87f3-579883a3671e)|![image_1_enhance](https://github.com/user-attachments/assets/d38be37d-68ed-4123-9cb2-429069dbd870)|
|![image_2_base](https://github.com/user-attachments/assets/e4b37440-fde0-4d7c-9658-98995d335097)|![image_2_enhance](https://github.com/user-attachments/assets/aa4ccae9-1074-4200-b890-5687f3409a70)|

3. Demonstration of the robustness of EliGen. The following examples are generated using the same prompt but different seeds. Refer to [./entity_control.py](./entity_control.py) `example_7` for the prompts.

|![image_1_base](https://github.com/user-attachments/assets/fb39ca42-074b-4d7c-85c8-55e4dec9a851)|![image_1_base](https://github.com/user-attachments/assets/34d7f17a-06b6-492a-8522-44aa1c75f233)|
|-|-|
|![image_1_base](https://github.com/user-attachments/assets/cecfee76-7e44-496b-8d02-ffa29f5142a3)|![image_1_enhance](https://github.com/user-attachments/assets/a51d3bba-52e6-483f-9c75-f7e87120b30c)|

### Image Inpainting
Demonstration of the inpainting mode of EliGen, see [./entity_inpaint.py](./entity_inpaint.py) for generation prompts.
|Inpainting Input|Inpainting Output|
|-|-|
|![image_2_base](https://github.com/user-attachments/assets/5ef499f3-3d8a-49cc-8ceb-86af7f5cb9f8)|![image_2_enhance](https://github.com/user-attachments/assets/88fc3bde-0984-4b3c-8ca9-d63de660855b)|
|![image_1_base](https://github.com/user-attachments/assets/5f74c710-bf30-4db1-ae40-a1e1995ccef6)|![image_1_enhance](https://github.com/user-attachments/assets/1cd71177-e956-46d3-86ce-06f774c96efd)|
### Styled Entity Control
Demonstration of the styled entity control results with EliGen and IP-Adapter, see [./entity_control_ipadapter.py](./entity_control_ipadapter.py) for generation prompts.
|Style Reference|Entity Control Variance 1|Entity Control Variance 2|Entity Control Variance 3|
|-|-|-|-|
|![image_1_base](https://github.com/user-attachments/assets/5e2dd3ab-37d3-4f58-8e02-ee2f9b238604)|![image_1_enhance](https://github.com/user-attachments/assets/0f6711a2-572a-41b3-938a-95deff6d732d)|![image_1_enhance](https://github.com/user-attachments/assets/ce2e66e5-1fdf-44e8-bca7-555d805a50b1)|![image_1_enhance](https://github.com/user-attachments/assets/ad2da233-2f7c-4065-ab57-b2d84dc2c0e2)|

### Entity Transfer
Demonstration of the entity transfer results with EliGen and In-Context LoRA, see [./entity_transfer.py](./entity_transfer.py) for generation prompts.

|Entity to Transfer|Transfer Target Image|Transfer Example 1|Transfer Example 2|
|-|-|-|-|
|![image_1_base](https://github.com/user-attachments/assets/bb3d4a46-8d82-4d3c-bce8-8c01a9973b8d)|![image_1_enhance](https://github.com/user-attachments/assets/44c0f422-525e-42ca-991b-f407f8faafc3)|![image_1_enhance](https://github.com/user-attachments/assets/a042ff5b-2748-4d91-8321-cec8f9eb73e4)|![image_1_enhance](https://github.com/user-attachments/assets/98f2d1b1-16e1-4c8f-b521-5cd68b567293)|