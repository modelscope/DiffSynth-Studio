# Kolors

Kolors is a Chinese diffusion model, which is based on ChatGLM and Stable Diffusion XL. We provide training scripts here.

## Download models

The following files will be used for constructing Kolors. You can download them from [huggingface](https://huggingface.co/Kwai-Kolors/Kolors) or [modelscope](https://modelscope.cn/models/Kwai-Kolors/Kolors).

```
models/kolors/Kolors
├── text_encoder
│   ├── config.json
│   ├── pytorch_model-00001-of-00007.bin
│   ├── pytorch_model-00002-of-00007.bin
│   ├── pytorch_model-00003-of-00007.bin
│   ├── pytorch_model-00004-of-00007.bin
│   ├── pytorch_model-00005-of-00007.bin
│   ├── pytorch_model-00006-of-00007.bin
│   ├── pytorch_model-00007-of-00007.bin
│   └── pytorch_model.bin.index.json
├── unet
│   └── diffusion_pytorch_model.safetensors
└── vae
    └── diffusion_pytorch_model.safetensors
```

You can use the following code to download these files:

```python
from diffsynth import download_models

download_models(["Kolors"])
```

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

We provide a training script `train_kolors_lora.py`. Before you run this training script, please copy it to the root directory of this project.

The following settings are recommended. **We found the UNet model suffers from precision overflow issues, thus the training script doesn't support float16. 40GB VRAM is required. We are working on overcoming this pitfall.**

```
CUDA_VISIBLE_DEVICES="0" python examples/train/kolors/train_kolors_lora.py \
  --pretrained_path models/kolors/Kolors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 10 \
  --center_crop \
  --use_gradient_checkpointing \
  --precision 32
```

Optional arguments:
```
  -h, --help            show this help message and exit
  --pretrained_path PRETRAINED_PATH
                        Path to pretrained model. For example, `models/kolors/Kolors`.
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
from diffsynth import ModelManager, KolorsImagePipeline
from peft import LoraConfig, inject_adapter_in_model
import torch


def load_lora(model, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out"],
    )
    model = inject_adapter_in_model(lora_config, model)
    state_dict = torch.load(lora_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                             ])
pipe = KolorsImagePipeline.from_model_manager(model_manager)

# Generate an image with lora
pipe.unet = load_lora(
    pipe.unet,
    lora_rank=4, lora_alpha=4.0, # The two parameters should be consistent with those in your training script.
    lora_path="path/to/your/lora/model/lightning_logs/version_x/checkpoints/epoch=x-step=xxx.ckpt"
)
torch.manual_seed(0)
image = pipe(
    prompt="一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉",
    negative_prompt="",
    cfg_scale=4,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_with_lora.jpg")
```

Prompt: 一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉

|Without LoRA|With LoRA|
|-|-|
|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0)|
