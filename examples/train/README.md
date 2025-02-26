# DiffSynth Training Framework

We have implemented a training framework for text-to-image Diffusion models, enabling users to easily train LoRA models using our framework. Our provided scripts come with the following advantages:

* **Comprehensive Functionality & User-Friendliness**: Our training framework supports multi-GPU and multi-machine setups, facilitates the use of DeepSpeed for acceleration, and includes gradient checkpointing optimizations for models with excessive memory demands.
* **Code Conciseness & Researcher Accessibility**: We avoid large blocks of complicated code. General-purpose modules are implemented in `diffsynth/trainers/text_to_image.py`, while model-specific training scripts contain only minimal code pertinent to the model architecture, making it researcher-friendly.
* **Modular Design & Developer Flexibility**: Built on the universal Pytorch-Lightning framework, our training framework is decoupled in terms of functionality, allowing developers to easily introduce additional training techniques by modifying our scripts to suit their needs.

Image Examples of fine-tuned LoRA. The prompt is "一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉" (for Chinese models) or "a dog is jumping, flowers around the dog, the background is mountains and clouds" (for English models).

||FLUX.1-dev|Kolors|Stable Diffusion 3|Hunyuan-DiT|
|-|-|-|-|-|
|Without LoRA|![image_without_lora](https://github.com/user-attachments/assets/df62cef6-d54f-4e3d-a602-5dd290079d49)|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/9d79ed7a-e8cf-4d98-800a-f182809db318)|![image_without_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/ddb834a5-6366-412b-93dc-6d957230d66e)|![image_without_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1aa21de5-a992-4b66-b14f-caa44e08876e)|
|With LoRA|![image_with_lora](https://github.com/user-attachments/assets/4fd39890-0291-4d19-8a88-d70d0ae18533)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/02f62323-6ee5-4788-97a1-549732dbe4f0)|![image_with_lora](https://github.com/modelscope/DiffSynth-Studio/assets/35051019/8e7b2888-d874-4da4-a75b-11b6b214b9bf)|![image_with_lora](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/83a0a41a-691f-4610-8e7b-d8e17c50a282)|

## Install additional packages

```
pip install peft lightning pandas
```

## Prepare your dataset

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

Note that if the model is Chinese model (for example, Hunyuan-DiT and Kolors), we recommend to use Chinese texts in the dataset. For example

```
file_name,text
00.jpg,一只小狗
01.jpg,一只小狗
02.jpg,一只小狗
03.jpg,一只小狗
04.jpg,一只小狗
```

## Train a LoRA model

General options:

```
  --lora_target_modules LORA_TARGET_MODULES
                        Layers with LoRA modules.
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
  --modelscope_model_id MODELSCOPE_MODEL_ID
                        Model ID on ModelScope (https://www.modelscope.cn/). The model will be uploaded to ModelScope automatically if you provide a Model ID.
  --modelscope_access_token MODELSCOPE_ACCESS_TOKEN
                        Access key on ModelScope (https://www.modelscope.cn/). Required if you want to upload the model to ModelScope.
```

### FLUX

The following files will be used for constructing FLUX. You can download them from [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-dev) or [modelscope](https://www.modelscope.cn/models/ai-modelscope/flux.1-dev). You can use the following code to download these files:

```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

```
models/FLUX/
└── FLUX.1-dev
    ├── ae.safetensors
    ├── flux1-dev.safetensors
    ├── text_encoder
    │   └── model.safetensors
    └── text_encoder_2
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        └── model.safetensors.index.json
```

Launch the training task using the following command (39G VRAM required):

```
CUDA_VISIBLE_DEVICES="0" python examples/train/flux/train_flux_lora.py \
  --pretrained_text_encoder_path models/FLUX/FLUX.1-dev/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path models/FLUX/FLUX.1-dev/text_encoder_2 \
  --pretrained_dit_path models/FLUX/FLUX.1-dev/flux1-dev.safetensors \
  --pretrained_vae_path models/FLUX/FLUX.1-dev/ae.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 100 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --use_gradient_checkpointing \
  --align_to_opensource_format
```

By adding parameter `--quantize "float8_e4m3fn"`, you can save approximate 10G VRAM.

**`--align_to_opensource_format` means that this script will export the LoRA weights in the opensource format. This format can be loaded in both DiffSynth-Studio and other codebases.**

For more information about the parameters, please use `python examples/train/flux/train_flux_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

```python
from diffsynth import ModelManager, FluxImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda",
                             file_path_list=[
                                 "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
                                 "models/FLUX/FLUX.1-dev/text_encoder_2",
                                 "models/FLUX/FLUX.1-dev/ae.safetensors",
                                 "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = FluxImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds",
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_with_lora.jpg")
```

### Kolors

The following files will be used for constructing Kolors. You can download Kolors from [huggingface](https://huggingface.co/Kwai-Kolors/Kolors) or [modelscope](https://modelscope.cn/models/Kwai-Kolors/Kolors). Due to precision overflow issues, we need to download an additional VAE model (from [huggingface](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) or [modelscope](https://modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix)). You can use the following code to download these files:

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

Launch the training task using the following command:

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

For more information about the parameters, please use `python examples/train/kolors/train_kolors_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

```python
from diffsynth import ModelManager, SDXLImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SDXLImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="一只小狗蹦蹦跳跳，周围是姹紫嫣红的鲜花，远处是山脉", 
    negative_prompt="",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```

### Stable Diffusion 3.5 Series


You need to download the text encoders and DiT model files. Please use the following code to download these files:

```python
from diffsynth import download_models

download_models(["StableDiffusion3.5-large"])
```

```
models/stable_diffusion_3
├── Put Stable Diffusion 3 checkpoints here.txt
├── sd3.5_large.safetensors
└── text_encoders
    ├── clip_g.safetensors
    ├── clip_l.safetensors
    └── t5xxl_fp16.safetensors
```

Launch the training task using the following command:

```
CUDA_VISIBLE_DEVICES="0" python examples/train/stable_diffusion_3/train_sd3_lora.py \
  --pretrained_path models/stable_diffusion_3/text_encoders/clip_g.safetensors,models/stable_diffusion_3/text_encoders/clip_l.safetensors,models/stable_diffusion_3/text_encoders/t5xxl_fp16.safetensors,models/stable_diffusion_3/sd3.5_large.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

For more information about the parameters, please use `python examples/train/stable_diffusion_3/train_sd3_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

```python
from diffsynth import ModelManager, SD3ImagePipeline
import torch

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/stable_diffusion_3/text_encoders/clip_g.safetensors",
                                 "models/stable_diffusion_3/text_encoders/clip_l.safetensors",
                                 "models/stable_diffusion_3/text_encoders/t5xxl_fp16.safetensors",
                                 "models/stable_diffusion_3/sd3.5_large.safetensors"
                             ])
model_manager.load_lora("models/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt", lora_alpha=1.0)
pipe = SD3ImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="a dog is jumping, flowers around the dog, the background is mountains and clouds",
    num_inference_steps=30, cfg_scale=7
)
image.save("image_with_lora.jpg")
```

### Stable Diffusion 3

Only one file is required in the training script. You can use [`sd3_medium_incl_clips.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips.safetensors) (without T5 encoder) or [`sd3_medium_incl_clips_t5xxlfp16.safetensors`](https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors) (with T5 encoder). Please use the following code to download these files:

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

Launch the training task using the following command:

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
  --precision "16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing
```

For more information about the parameters, please use `python examples/train/stable_diffusion_3/train_sd3_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

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

Four files will be used for constructing Hunyuan DiT. You can download them from [huggingface](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) or [modelscope](https://www.modelscope.cn/models/modelscope/HunyuanDiT/summary). You can use the following code to download these files:

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

For more information about the parameters, please use `python examples/train/hunyuan_dit/train_hunyuan_dit_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

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

Only one file is required in the training script. We support the mainstream checkpoints in [CivitAI](https://civitai.com/). By default, we use the base Stable Diffusion v1.5. You can download it from [huggingface](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors) or [modelscope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/resolve/master/v1-5-pruned-emaonly.safetensors). You can use the following code to download this file:

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

For more information about the parameters, please use `python examples/train/stable_diffusion/train_sd_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

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

Only one file is required in the training script. We support the mainstream checkpoints in [CivitAI](https://civitai.com/). By default, we use the base Stable Diffusion XL. You can download it from [huggingface](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) or [modelscope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/resolve/master/sd_xl_base_1.0.safetensors). You can use the following code to download this file:

```python
from diffsynth import download_models

download_models(["StableDiffusionXL_v1"])
```

```
models/stable_diffusion_xl
├── Put Stable Diffusion XL checkpoints here.txt
└── sd_xl_base_1.0.safetensors
```

We observed that Stable Diffusion XL is not float16-safe, thus we recommend users to use float32.

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

For more information about the parameters, please use `python examples/train/stable_diffusion_xl/train_sdxl_lora.py -h` to see the details.

After training, use `model_manager.load_lora` to load the LoRA for inference.

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
