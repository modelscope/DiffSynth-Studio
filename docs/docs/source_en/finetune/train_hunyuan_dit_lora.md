# Training Hunyuan-DiT LoRA

Building the Hunyuan DiT model requires four files. You can download these files from [HuggingFace](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) or [ModelScope](https://www.modelscope.cn/models/modelscope/HunyuanDiT/summary). You can use the following code to download these files:

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

Use the following command to start the training task:

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

For more information about the parameters, please use `python examples/train/hunyuan_dit/train_hunyuan_dit_lora.py -h` to view detailed information.

After the training is complete, use `model_manager.load_lora` to load the LoRA for inference.


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
    prompt="A little puppy hops and jumps playfully, surrounded by a profusion of colorful flowers, with a mountain range visible in the distance.
", 
    negative_prompt="",
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_with_lora.jpg")
```
