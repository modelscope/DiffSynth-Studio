# 训练 Stable Diffusion XL LoRA

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

我们观察到 Stable Diffusion XL 在 float16 精度下会出现数值精度溢出，因此我们建议用户使用 float32 精度训练，使用以下命令启动训练任务：

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
