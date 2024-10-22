# 训练 Stable Diffusion 3 LoRA

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
