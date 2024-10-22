# Training Stable Diffusion LoRA

The training script only requires one file. We support mainstream checkpoints on [CivitAI](https://civitai.com/). By default, we use the basic Stable Diffusion v1.5. You can download it from [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors) or [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/resolve/master/v1-5-pruned-emaonly.safetensors). You can use the following code to download this file:

```python
from diffsynth import download_models

download_models(["StableDiffusion_v15"])
```

```
models/stable_diffusion
├── Put Stable Diffusion checkpoints here.txt
└── v1-5-pruned-emaonly.safetensors
```

Start the training task with the following command:

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

For more information about the parameters, please use `python examples/train/stable_diffusion/train_sd_lora.py -h` to view detailed information.

After training is complete, use `model_manager.load_lora` to load LoRA for inference.


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
