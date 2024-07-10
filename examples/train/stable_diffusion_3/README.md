# Stable Diffusion 3

Stable Diffusion 3 is a powerful text-to-image model. We provide training scripts here.

## Download models

Only one file is required in the training script. You can use [`sd3_medium_incl_clips.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors) (without T5 encoder) or [`sd3_medium_incl_clips_t5xxlfp16.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors) (with T5 encoder).

```
models/stable_diffusion_3/
├── Put Stable Diffusion 3 checkpoints here.txt
├── sd3_medium_incl_clips.safetensors
└── sd3_medium_incl_clips_t5xxlfp16.safetensors
```

You can use the following code to download these files:

```python
from diffsynth import download_models

download_models(["StableDiffusion3", "StableDiffusion3_without_T5"])
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
00.jpg,a dog
01.jpg,a dog
02.jpg,a dog
03.jpg,a dog
04.jpg,a dog
```

### Train a LoRA model

We provide a training script `train_sd3_lora.py`. Before you run this training script, please copy it to the root directory of this project.

We recommand to enable gradient checkpointing. 10GB VRAM is enough if you train LoRA without the T5 encoder (use `sd3_medium_incl_clips.safetensors`), and 19GB VRAM is required if you enable the T5 encoder (use `sd3_medium_incl_clips_t5xxlfp16.safetensors`).

```
CUDA_VISIBLE_DEVICES="0" python train_sd3_lora.py \
  --pretrained_path models/stable_diffusion_3/sd3_medium_incl_clips.safetensors \
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
                        Path to pretrained model. For example, `models/stable_diffusion_3/sd3_medium_incl_clips.safetensors` or `models/stable_diffusion_3/sd3_medium_incl_clips_t5xxlfp16.safetensors`.
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
from diffsynth import ModelManager, SD3ImagePipeline
import torch
from peft import LoraConfig, inject_adapter_in_model


def load_lora(dit, lora_rank, lora_alpha, lora_path):
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["a_to_qkv", "b_to_qkv"],
    )
    dit = inject_adapter_in_model(lora_config, dit)
    state_dict = torch.load(lora_path, map_location="cpu")
    dit.load_state_dict(state_dict, strict=False)
    return dit


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)


# Generate an image with lora
pipe.dit = load_lora(
    pipe.dit,
    lora_rank=4, lora_alpha=4.0, # The two parameters should be consistent with those in your training script.
    lora_path="path/to/your/lora/model/lightning_logs/version_x/checkpoints/epoch=x-step=xxx.ckpt"
)
torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds", 
    negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi, extra tails",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```

Prompt: 

|Without LoRA|With LoRA|
|-|-|
|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf)|
