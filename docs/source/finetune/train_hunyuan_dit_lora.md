# 训练 Hunyuan-DiT LoRA

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

使用以下命令启动训练任务：

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
