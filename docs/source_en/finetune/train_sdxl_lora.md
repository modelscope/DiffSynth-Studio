# Training Stable Diffusion XL LoRA


The training script only requires one file. We support mainstream checkpoints on [CivitAI](https://civitai.com/). By default, we use the basic Stable Diffusion XL. You can download it from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) 或 [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/resolve/master/sd_xl_base_1.0.safetensors). You can also use the following code to download this file:
```python
from diffsynth import download_models

download_models(["StableDiffusionXL_v1"])
```

```
models/stable_diffusion_xl
├── Put Stable Diffusion XL checkpoints here.txt
└── sd_xl_base_1.0.safetensors
```

We have observed that Stable Diffusion XL may experience numerical precision overflows when using float16 precision, so we recommend that users train with float32 precision. To start the training task, use the following command:
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

For more information about the parameters, please use `python examples/train/stable_diffusion_xl/train_sdxl_lora.py -h` to view detailed information.

After training is complete, use `model_manager.load_lora` to load LoRA for inference.


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
