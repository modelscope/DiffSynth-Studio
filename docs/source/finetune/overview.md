# 训练框架

我们实现了一个用于文本到图像扩散模型的训练框架，使用户能够轻松地使用我们的框架训练 LoRA 模型。我们提供的脚本具有以下特点：

* **功能全面**：我们的训练框架支持多GPU和多机器配置，便于使用 DeepSpeed 加速，并包括梯度检查点优化，适用于内存需求较大的模型。
* **代码简洁**：我们避免了大块复杂的代码。通用模块实现于 `diffsynth/trainers/text_to_image.py` 中，而模型特定的训练脚本仅包含与模型架构相关的最少代码，便于学术研究人员使用。
* **模块化设计**：基于通用的 Pytorch-Lightning 框架，我们的训练框架在功能上是解耦的，允许开发者通过修改我们的脚本轻松引入额外的训练技术，以满足他们的需求。

LoRA 微调的图像示例。提示词为 "一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉"（针对中文模型）或 "a dog is jumping, flowers around the dog, the background is mountains and clouds"（针对英文模型）。

||<div style="width:150px">FLUX.1-dev</div>|<div style="width:150px">Kolors</div>|<div style="width:150px">Stable Diffusion 3</div>|<div style="width:150px">Hunyuan-DiT</div>|
|-|:-:|:-:|:-:|:-:|
|Without LoRA|<img src="https://github.com/user-attachments/assets/df62cef6-d54f-4e3d-a602-5dd290079d49" width="150"  alt="image_without_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318" width="150"  alt="image_without_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e" width="150"  alt="image_without_lora">|<img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e" width="150"  alt="image_without_lora">|
|With LoRA|<img src="https://github.com/user-attachments/assets/4fd39890-0291-4d19-8a88-d70d0ae18533" width="150"  alt="image_with_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0" width="150"  alt="image_with_lora">|<img src="https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf" width="150"  alt="image_with_lora">|<img src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282" width="150"  alt="image_with_lora">|


## 安装额外包

```
pip install peft lightning
```

## 准备数据集

我们提供了一个[示例数据集](https://modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/files)。你需要将训练数据集按照如下形式组织：

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

请注意，如果模型是中文模型（例如，Hunyuan-DiT 和 Kolors），我们建议在数据集中使用中文文本。例如：

```
file_name,text
00.jpg,一只小狗
01.jpg,一只小狗
02.jpg,一只小狗
03.jpg,一只小狗
04.jpg,一只小狗
```

## 训练 LoRA 模型

通用参数选项：

```
  --lora_target_modules LORA_TARGET_MODULES
                        LoRA 模块所在的层。
  --dataset_path DATASET_PATH
                        数据集的路径。
  --output_path OUTPUT_PATH
                        模型保存路径。
  --steps_per_epoch STEPS_PER_EPOCH
                        每个周期的步数。
  --height HEIGHT       图像高度。
  --width WIDTH         图像宽度。
  --center_crop         是否将输入图像中心裁剪到指定分辨率。如果未设置，图像将被随机裁剪。图像会在裁剪前先调整到指定分辨率。
  --random_flip         是否随机水平翻转图像。
  --batch_size BATCH_SIZE
                        训练数据加载器的批量大小（每设备）。
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        数据加载使用的子进程数量。0 表示数据将在主进程中加载。
  --precision {32,16,16-mixed}
                        训练精度。
  --learning_rate LEARNING_RATE
                        学习率。
  --lora_rank LORA_RANK
                        LoRA 更新矩阵的维度。
  --lora_alpha LORA_ALPHA
                        LoRA 更新矩阵的权重。
  --use_gradient_checkpointing
                        是否使用梯度检查点。
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        梯度累积的批次数量。
  --training_strategy {auto,deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3}
                        训练策略。
  --max_epochs MAX_EPOCHS
                        训练轮数。
  --modelscope_model_id MODELSCOPE_MODEL_ID
                        ModelScope 上的模型 ID (https://www.modelscope.cn/)。如果提供模型 ID，模型将自动上传到 ModelScope。
  --modelscope_access_token MODELSCOPE_ACCESS_TOKEN
                        在 ModelScope (https://www.modelscope.cn/) 上获取访问密钥。您需要此密钥将模型上传到 ModelScope。
```
