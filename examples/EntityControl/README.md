# EliGen: Entity-Level Controlled Image Generation

## Introduction

We propose EliGen, a novel approach that leverages fine-grained entity-level information to enable precise and controllable text-to-image generation. EliGen excels in tasks such as entity-level controlled image generation and image inpainting, while its applicability is not limited to these areas. Additionally, it can be seamlessly integrated with existing community models, such as the IP-Adpater and In-Cotext LoRA.

* Paper: [EliGen: Entity-Level Controlled Image Generation with Regional Attention](https://arxiv.org/abs/2501.01097)
* Github: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* Model: [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen)
* Online Demo: [ModelScope EliGen Studio](https://www.modelscope.cn/studios/DiffSynth-Studio/EliGen)
* Training dataset: Coming soon

## Methodology

![regional-attention](https://github.com/user-attachments/assets/bef5ae2b-cc03-404e-b9c8-0c037ac66190)

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
   Run the following command to try interactive UI: 
   ```bash
   python apps/gradio/entity_level_control.py
   ```
## Examples
### Entity-Level Controlled Image Generation

1. The effect of generating images with continuously changing entity positions.

https://github.com/user-attachments/assets/54a048c8-b663-4262-8c40-43c87c266d4b

2. The image generation effect of complex Entity combinations, demonstrating the strong generalization of EliGen. See [./entity_control.py](./entity_control.py) `example_1-6` for generation prompts.

|Entity Conditions|Generated Image|
|-|-|
|![eligen_example_1_mask_0](https://github.com/user-attachments/assets/68cbedc0-32aa-4a8e-99d2-306dbb4620de)|![eligen_example_1_0](https://github.com/user-attachments/assets/c678c4b1-aa19-41df-b612-adc01b8b2009)|
|![eligen_example_2_mask_0](https://github.com/user-attachments/assets/1c6d9445-5022-4d91-ad2e-dc05321883d1)|![eligen_example_2_0](https://github.com/user-attachments/assets/86739945-cb07-4a49-b3b3-3bb65c90d14f)|
|![eligen_example_3_mask_27](https://github.com/user-attachments/assets/5ca4440d-d1db-45dd-b03c-0affefbd9ac3)|![eligen_example_3_27](https://github.com/user-attachments/assets/9160c22a-89ac-4d52-be1d-17ba2d8a67eb)|
|![eligen_example_4_mask_21](https://github.com/user-attachments/assets/26dfde2b-cc9a-4cb3-806a-7f7436d971a7)|![eligen_example_4_21](https://github.com/user-attachments/assets/1fff7346-6a8c-4eb6-986f-4ea848c6b363)|
|![eligen_example_5_mask_0](https://github.com/user-attachments/assets/8ca94e5f-f896-451d-a700-bcdc23689adb)|![eligen_example_5_0](https://github.com/user-attachments/assets/881a9395-6cc2-43e9-89b4-30b8f5437e6d)|
|![eligen_example_6_mask_8](https://github.com/user-attachments/assets/26c95abf-f2b1-44db-92c1-75d02c714c74)|![eligen_example_6_8](https://github.com/user-attachments/assets/8883abde-3fad-4a8b-ade0-ca5b977a290f)|

1. Demonstration of the robustness of EliGen. The following examples are generated using the same prompt but different seeds. Refer to [./entity_control.py](./entity_control.py) `example_7` for the prompts.

|Entity Conditions|Generated Image|
|-|-|
|![eligen_example_7_mask_5](https://github.com/user-attachments/assets/85630237-9d8b-41ea-9bd5-506652c61776)|![eligen_example_7_5](https://github.com/user-attachments/assets/d34b54d2-c59c-4c39-8ab4-c22f155283f1)|
|![eligen_example_7_mask_5](https://github.com/user-attachments/assets/85630237-9d8b-41ea-9bd5-506652c61776)|![eligen_example_7_6](https://github.com/user-attachments/assets/4050a3bf-a089-4f4f-81e0-e3b391cf7ceb)|
![eligen_example_7_mask_5](https://github.com/user-attachments/assets/85630237-9d8b-41ea-9bd5-506652c61776)|![eligen_example_7_7](https://github.com/user-attachments/assets/682feb5e-a27a-4ae4-a800-018b4e0e504c)|
|![eligen_example_7_mask_5](https://github.com/user-attachments/assets/85630237-9d8b-41ea-9bd5-506652c61776)|![eligen_example_7_8](https://github.com/user-attachments/assets/50266950-24b3-426a-ae74-c3ebadb853d9)|

### Image Inpainting
Demonstration of the inpainting mode of EliGen, see [./entity_inpaint.py](./entity_inpaint.py) for generation prompts.
|Inpainting Input|Inpainting Output|
|-|-|
|![inpaint_i1](https://github.com/user-attachments/assets/5ef499f3-3d8a-49cc-8ceb-86af7f5cb9f8)|![inpaint_o1](https://github.com/user-attachments/assets/88fc3bde-0984-4b3c-8ca9-d63de660855b)|
|![inpaint_i2](https://github.com/user-attachments/assets/5f74c710-bf30-4db1-ae40-a1e1995ccef6)|![inpaint_o2](https://github.com/user-attachments/assets/7c3b4857-b774-47ea-b163-34d49e7c976d)|
### Styled Entity Control
Demonstration of the styled entity control results with EliGen and IP-Adapter, see [./entity_control_ipadapter.py](./entity_control_ipadapter.py) for generation prompts.
|Style Reference|Entity Control Variance 1|Entity Control Variance 2|Entity Control Variance 3|
|-|-|-|-|
|![image_1_base](https://github.com/user-attachments/assets/5e2dd3ab-37d3-4f58-8e02-ee2f9b238604)|![result1](https://github.com/user-attachments/assets/0f6711a2-572a-41b3-938a-95deff6d732d)|![result2](https://github.com/user-attachments/assets/ce2e66e5-1fdf-44e8-bca7-555d805a50b1)|![result3](https://github.com/user-attachments/assets/ad2da233-2f7c-4065-ab57-b2d84dc2c0e2)|

### Entity Transfer
Demonstration of the entity transfer results with EliGen and In-Context LoRA, see [./entity_transfer.py](./entity_transfer.py) for generation prompts.

|Entity to Transfer|Transfer Target Image|Transfer Example 1|Transfer Example 2|
|-|-|-|-|
|![source](https://github.com/user-attachments/assets/0d40ef22-0a09-420d-bd5a-bfb93120b60d)|![targe](https://github.com/user-attachments/assets/f6c58ef2-54c1-4d86-8429-dad2eb0e0685)|![result1](https://github.com/user-attachments/assets/05eed2e3-097d-40af-8aae-1e0c75051f32)|![result2](https://github.com/user-attachments/assets/54314d16-244b-411e-8a91-96c500efa5f5)|