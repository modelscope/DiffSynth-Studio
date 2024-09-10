# 微调

我们实现了一个用于文本到图像扩散模型的训练框架，使用户能够轻松地使用我们的框架训练 LoRA 模型。我们提供的脚本具有以下特点：

* **全面功能与用户友好性**：我们的训练框架支持多GPU和多机器配置，便于使用 DeepSpeed 加速，并包括梯度检查点优化，适用于内存需求较大的模型。
* **代码简洁与研究者可及性**：我们避免了大块复杂的代码。通用模块实现于 `diffsynth/trainers/text_to_image.py` 中，而模型特定的训练脚本仅包含与模型架构相关的最少代码，便于研究人员使用。
* **模块化设计与开发者灵活性**：基于通用的 Pytorch-Lightning 框架，我们的训练框架在功能上是解耦的，允许开发者通过修改我们的脚本轻松引入额外的训练技术，以满足他们的需求。

LoRA 微调的图像示例。提示词为 "一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉"（针对中文模型）或 "a dog is jumping, flowers around the dog, the background is mountains and clouds"（针对英文模型）。

||Kolors|Stable Diffusion 3|Hunyuan-DiT|
|-|-|-|-|
|Without LoRA|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318)|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e)|![image_without_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e)|
|With LoRA|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf)|![image_with_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282)|

## 下载需要的包

```bash
pip install peft lightning
```

## 准备你的数据

我们提供了一个 [示例数据集](https://modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/files)。你需要将训练数据集按照如下形式组织：


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

参数选项：

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
                        训练周期数。
  --modelscope_model_id MODELSCOPE_MODEL_ID
                        ModelScope 上的模型 ID (https://www.modelscope.cn/)。如果提供模型 ID，模型将自动上传到 ModelScope。

```

### Kolors

以下文件将用于构建 Kolors。你可以从 [HuggingFace](https://huggingface.co/Kwai-Kolors/Kolors) 或 [ModelScope](https://modelscope.cn/models/Kwai-Kolors/Kolors) 下载 Kolors。由于精度溢出问题，我们需要下载额外的 VAE 模型（从 [HuggingFace](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) 或 [ModelScope](https://modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix)）。你可以使用以下代码下载这些文件：


```python
from diffsynth import download_models

download_models(["Kolors", "SDXL-vae-fp16-fix"])
```

```
models
├── kolors
│   └── Kolors
│       ├── text_encoder
│       │   ├── config.json
│       │   ├── pytorch_model-00001-of-00007.bin
│       │   ├── pytorch_model-00002-of-00007.bin
│       │   ├── pytorch_model-00003-of-00007.bin
│       │   ├── pytorch_model-00004-of-00007.bin
│       │   ├── pytorch_model-00005-of-00007.bin
│       │   ├── pytorch_model-00006-of-00007.bin
│       │   ├── pytorch_model-00007-of-00007.bin
│       │   └── pytorch_model.bin.index.json
│       ├── unet
│       │   └── diffusion_pytorch_model.safetensors
│       └── vae
│           └── diffusion_pytorch_model.safetensors
└── sdxl-vae-fp16-fix
    └── diffusion_pytorch_model.safetensors
```

使用下面的命令启动训练任务：

```
CUDA_VISIBLE_DEVICES="0" python examples/train/kolors/train_kolors_lora.py \
  --pretrained_unet_path models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors \
  --pretrained_text_encoder_path models/kolors/Kolors/text_encoder \
  --pretrained_fp16_vae_path models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16-mixed" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/kolors/train_kolors_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。



```python
from diffsynth import ModelManager, SD3ImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SD3ImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, extra tails",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```


### Stable Diffusion 3

训练脚本只需要一个文件。你可以使用 [`sd3_medium_incl_clips.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors)（没有 T5 Encoder）或 [`sd3_medium_incl_clips_t5xxlfp16.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors)（有 T5 Encoder）。请使用以下代码下载这些文件：


```python
from diffsynth import download_models

download_models(["StableDiffusion3", "StableDiffusion3_without_T5"])
```

```
models/stable_diffusion_3/
├── Put Stable Diffusion 3 checkpoints here.txt
├── sd3_medium_incl_clips.safetensors
└── sd3_medium_incl_clips_t5xxlfp16.safetensors
```

使用下面的命令启动训练任务：

```
CUDA_VISIBLE_DEVICES="0" python examples/train/stable_diffusion_3/train_sd3_lora.py \
  --pretrained_path models/stable_diffusion_3/sd3_medium_incl_clips.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16-mixed" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/stable_diffusion_3/train_sd3_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。

```python
from diffsynth import ModelManager, SD3ImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SD3ImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, extra tails",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```

### Hunyuan-DiT

构建 Hunyuan DiT 需要四个文件。你可以从 [HuggingFace](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 或 [ModelScope](https://www.modelscope.cn/models/modelscope/HunyuanDiT/summary) 下载这些文件。你可以使用以下代码下载这些文件：


```python
from diffsynth import download_models

download_models(["HunyuanDiT"])
```

```
models/HunyuanDiT/
├── Put Hunyuan DiT checkpoints here.txt
└── t2i
    ├── clip_text_encoder
    │   └── pytorch_model.bin
    ├── model
    │   └── pytorch_model_ema.pt
    ├── mt5
    │   └── pytorch_model.bin
    └── sdxl-vae-fp16-fix
        └── diffusion_pytorch_model.bin
```

Launch the training task using the following command:

```
CUDA_VISIBLE_DEVICES="0" python examples/train/hunyuan_dit/train_hunyuan_dit_lora.py \
  --pretrained_path models/HunyuanDiT/t2i \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16-mixed" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/hunyuan_dit/train_hunyuan_dit_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。


```python
from diffsynth import ModelManager, HunyuanDiTImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
                                 "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
                                 "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
                                 "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉", 
    negative_prompt="",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```

### Stable Diffusion

训练脚本只需要一个文件。我们支持 [CivitAI](https://civitai.com/) 中的主流检查点。默认情况下，我们使用基础的 Stable Diffusion v1.5。你可以从 [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors) 或 [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/resolve/master/v1-5-pruned-emaonly.safetensors) 下载。你可以使用以下代码下载这个文件：

```python
from diffsynth import download_models

download_models(["StableDiffusion_v15"])
```

```
models/stable_diffusion
├── Put Stable Diffusion checkpoints here.txt
└── v1-5-pruned-emaonly.safetensors
```

Launch the training task using the following command:

```
CUDA_VISIBLE_DEVICES="0" python examples/train/stable_diffusion/train_sd_lora.py \
  --pretrained_path models/stable_diffusion/v1-5-pruned-emaonly.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 512 \
  --width 512 \
  --center_crop \
  --precision "16-mixed" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/stable_diffusion/train_sd_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。



```python
from diffsynth import ModelManager, SDImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion/v1-5-pruned-emaonly.safetensors"])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SDImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, extra tails",
    cfg_scale=7.5,
    num_inference_steps=100, width=512, height=512,
)
image.save("image_with_lora.jpg")
```

### Stable Diffusion XL

训练脚本只需要一个文件。我们支持 [CivitAI](https://civitai.com/) 中的主流检查点。默认情况下，我们使用基础的 Stable Diffusion XL。你可以从 [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) 或 [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/resolve/master/sd_xl_base_1.0.safetensors) 下载。也可以使用以下代码下载这个文件：

```python
from diffsynth import download_models

download_models(["StableDiffusionXL_v1"])
```

```
models/stable_diffusion_xl
├── Put Stable Diffusion XL checkpoints here.txt
└── sd_xl_base_1.0.safetensors
```

We observed that Stable Diffusion XL is not float16-safe, thus we recommand users to use float32.

```
CUDA_VISIBLE_DEVICES="0" python examples/train/stable_diffusion_xl/train_sdxl_lora.py \
  --pretrained_path models/stable_diffusion_xl/sd_xl_base_1.0.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "32" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

有关参数的更多信息，请使用 `python examples/train/stable_diffusion_xl/train_sdxl_lora.py -h` 查看详细信息。

训练完成后，使用 `model_manager.load_lora` 加载 LoRA 以进行推理。

```python
from diffsynth import ModelManager, SDXLImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_xl/sd_xl_base_1.0.safetensors"])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, extra tails",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```
