Certainly, here is the continuation of the translation:

---

# Training Framework

We have implemented a training framework for text-to-image diffusion models, allowing users to effortlessly train LoRA models with our framework. Our provided scripts come with the following features:

* **Comprehensive Functionality**: Our training framework supports multi-GPU and multi-node configurations, is optimized for acceleration with DeepSpeed, and includes gradient checkpointing to accommodate models with higher memory requirements.
* **Succinct Code**: We have avoided large, complex code blocks. The general module is implemented in `diffsynth/trainers/text_to_image.py`, while model-specific training scripts contain only the minimal code necessary for the model architecture, facilitating ease of use for academic researchers.
* **Modular Design**: Built on the versatile PyTorch Lightning framework, our training framework is decoupled in functionality, enabling developers to easily incorporate additional training techniques by modifying our scripts to suit their specific needs.

Examples of images fine-tuned with LoRA. Prompts are "A little dog jumping around with colorful flowers around and mountains in the background" (for Chinese models) or "a dog is jumping, flowers around the dog, the background is mountains and clouds" (for English models).

||FLUX.1-dev|Kolors|Stable Diffusion 3|Hunyuan-DiT|
|-|-|-|-|-|
|Without LoRA|![image_without_lora](https://github.com/user-attachments/assets/df62cef6-d54f-4e3d-a602-5dd290079d49)|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318)|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e)|![image_without_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e)|
|With LoRA|![image_with_lora](https://github.com/user-attachments/assets/4fd39890-0291-4d19-8a88-d70d0ae18533)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf)|![image_with_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282)|

## Install Additional Packages

```bash
pip install peft lightning
```

## Prepare the Dataset

We provide an [example dataset](https://modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/files). You need to organize your training dataset in the following structure:

```
data/dog/
└── train
    ├── 00.jpg
    ├── 01.jpg
    ├── 02.jpg
    ├── 03.jpg
    ├── 04.jpg
    └── metadata.csv
```

`metadata.csv`:

```
file_name,text
00.jpg,a dog
01.jpg,a dog
02.jpg,a dog
03.jpg,a dog
04.jpg,a dog
```

Please note that if the model is a Chinese model (e.g., Hunyuan-DiT and Kolors), we recommend using Chinese text in the dataset. For example:

```
file_name,text
00.jpg,a dog
01.jpg,a dog
02.jpg,a dog
03.jpg,a dog
04.jpg,a dog
```

## Train LoRA Model

General parameter options:

```
  --lora_target_modules LORA_TARGET_MODULES
                        Layers where the LoRA modules are located.
  --dataset_path DATASET_PATH
                        Path to the dataset.
  --output_path OUTPUT_PATH
                        Path where the model will be saved.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps per epoch.
  --height HEIGHT        The height of the image.
  --width WIDTH          The width of the image.
  --center_crop         Whether to center crop the input image to the specified resolution. If not set, the image will be randomly cropped. The image will be resized to the specified resolution before cropping.
  --random_flip         Whether to randomly horizontally flip the image.
  --batch_size BATCH_SIZE
                        Batch size for the training data loader (per device).
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        The number of subprocesses used for data loading. A value of 0 means the data will be loaded in the main process.
  --precision {32,16,16-mixed}
                        The precision for training.
  --learning_rate LEARNING_RATE
                        The learning rate.
  --lora_rank LORA_RANK
                        The dimension of the LoRA update matrix.
  --lora_alpha LORA_ALPHA
                        The weight of the LoRA update matrix.
  --use_gradient_checkpointing
                        Whether to use gradient checkpointing.
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        The number of batches for gradient accumulation.
  --training_strategy {auto,deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3}
                        The training strategy.
  --max_epochs MAX_EPOCHS
                        The number of training epochs.
  --modelscope_model_id MODELSCOPE_MODEL_ID
                        The model ID on ModelScope (https://www.modelscope.cn/). If the model ID is provided, the model will be automatically uploaded to ModelScope.
```
