# Hunyuan DiT

Hunyuan DiT is an image generation model based on DiT. We provide training and inference support for Hunyuan DiT.

## Download models

Four files will be used for constructing Hunyuan DiT. You can download them from [huggingface](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) or [modelscope](https://www.modelscope.cn/models/modelscope/HunyuanDiT/summary).

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

## Inference

### Text-to-image with highres-fix

The original resolution of Hunyuan DiT is 1024x1024. If you want to use larger resolutions, please use highres-fix.

Hunyuan DiT is also supported in our UI.

```python
from diffsynth import ModelManager, HunyuanDiTImagePipeline
import torch


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
    "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
    "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
    "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
])
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

# Enjoy!
torch.manual_seed(0)
image = pipe(
    prompt="少女手捧鲜花，坐在公园的长椅上，夕阳的余晖洒在少女的脸庞，整个画面充满诗意的美感",
    negative_prompt="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，",
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_1024.png")

# Highres fix
image = pipe(
    prompt="少女手捧鲜花，坐在公园的长椅上，夕阳的余晖洒在少女的脸庞，整个画面充满诗意的美感",
    negative_prompt="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，",
    input_image=image.resize((2048, 2048)),
    num_inference_steps=50, height=2048, width=2048,
    cfg_scale=3.0, denoising_strength=0.5, tiled=True,
)
image.save("image_2048.png")
```

Prompt: 少女手捧鲜花，坐在公园的长椅上，夕阳的余晖洒在少女的脸庞，整个画面充满诗意的美感

|1024x1024|2048x2048 (highres-fix)|
|-|-|
|![image_1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/2b6528cf-a229-46e9-b7dd-4a9475b07308)|![image_2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/11d264ec-966b-45c9-9804-74b60428b866)|

### In-context reference (experimental)

This feature is similar to the "reference-only" mode in ControlNets. By extending the self-attention layer, the content in the reference image can be retained in the new image. Any number of reference images are supported, and the influence from each reference image can be controled by independent `reference_strengths` parameters.

```python
from diffsynth import ModelManager, HunyuanDiTImagePipeline
import torch


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
    "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
    "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
    "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
])
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

# Generate an image as reference
torch.manual_seed(0)
reference_image = pipe(
    prompt="梵高，星空，油画，明亮",
    negative_prompt="",
    num_inference_steps=50, height=1024, width=1024,
)
reference_image.save("image_reference.png")

# Generate a new image with reference
image = pipe(
    prompt="层峦叠嶂的山脉，郁郁葱葱的森林，皎洁明亮的月光，夜色下的自然美景",
    negative_prompt="",
    reference_images=[reference_image], reference_strengths=[0.4],
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_with_reference.png")
```

Prompt: 层峦叠嶂的山脉，郁郁葱葱的森林，皎洁明亮的月光，夜色下的自然美景

|Reference image|Generated new image|
|-|-|
|![image_reference](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/99b0189d-6175-4842-b480-3c0d2f9f7e17)|![image_with_reference](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/8e41dddb-f302-4a2d-9e52-5487d1f47ae6)|

## Train

### Install training dependency

```
pip install peft lightning pandas torchvision
```

### Prepare your dataset

We provide an example dataset [here](https://modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/files). You need to manage the training images as follows:

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
00.jpg,一只小狗
01.jpg,一只小狗
02.jpg,一只小狗
03.jpg,一只小狗
04.jpg,一只小狗
```

### Train a LoRA model

We provide a training script `train_hunyuan_dit_lora.py`. Before you run this training script, please copy it to the root directory of this project.

If GPU memory >= 24GB, we recommmand to use the following settings.

```
CUDA_VISIBLE_DEVICES="0" python train_hunyuan_dit_lora.py \
  --pretrained_path models/HunyuanDiT/t2i \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --center_crop
```

If 12GB <= GPU memory <= 24GB, we recommand to enable gradient checkpointing.

```
CUDA_VISIBLE_DEVICES="0" python train_hunyuan_dit_lora.py \
  --pretrained_path models/HunyuanDiT/t2i \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --center_crop \
  --use_gradient_checkpointing
```

Optional arguments:
```
  -h, --help            show this help message and exit
  --pretrained_path PRETRAINED_PATH
                        Path to pretrained model. For example, `./HunyuanDiT/t2i`.
  --dataset_path DATASET_PATH
                        The path of the Dataset.
  --output_path OUTPUT_PATH
                        Path to save the model.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps per epoch.
  --height HEIGHT       Image height.
  --width WIDTH         Image width.
  --center_crop         Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.
  --random_flip         Whether to randomly flip images horizontally
  --batch_size BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
  --precision {32,16,16-mixed}
                        Training precision
  --learning_rate LEARNING_RATE
                        Learning rate.
  --lora_rank LORA_RANK
                        The dimension of the LoRA update matrices.
  --lora_alpha LORA_ALPHA
                        The weight of the LoRA update matrices.
  --use_gradient_checkpointing
                        Whether to use gradient checkpointing.
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        The number of batches in gradient accumulation.
  --training_strategy {auto,deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3}
                        Training strategy
  --max_epochs MAX_EPOCHS
                        Number of epochs.
```

### Inference with your own LoRA model

After training, you can use your own LoRA model to generate new images. Here are some examples.

```python
from diffsynth import ModelManager, HunyuanDiTImagePipeline
from peft import LoraConfig, inject_adapter_in_model
import torch


def load_lora(dit, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    dit = inject_adapter_in_model(lora_config, dit)
    state_dict = torch.load(lora_path, map_location="cpu")
    dit.load_state_dict(state_dict, strict=False)
    return dit


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
    "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
    "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
    "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
])
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

# Generate an image with lora
pipe.dit = load_lora(
    pipe.dit, lora_rank=4, lora_alpha=4.0,
    lora_path="path/to/your/lora/model/lightning_logs/version_x/checkpoints/epoch=x-step=xxx.ckpt"
)
torch.manual_seed(0)
image = pipe(
    prompt="一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉",
    negative_prompt="",
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_with_lora.png")
```

Prompt: 一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉

|Without LoRA|With LoRA|
|-|-|
|![image_without_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e)|![image_with_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282)|
