# Training Kolors LoRA
The following files will be used to build Kolors. You can download Kolors from [HuggingFace](https://huggingface.co/Kwai-Kolors/Kolors) or [ModelScope](https://modelscope.cn/models/Kwai-Kolors/Kolors). Due to precision overflow issues, we need to download an additional VAE model （from [HuggingFace](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) or [ModelScope](https://modelscope.cn/models/AI-ModelScope/sdxl-vae-fp16-fix). You can use the following code to download these files:

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

Use the following command to start the training task:

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

For more information on the parameters, please use `python examples/train/kolors/train_kolors_lora.py -h` to view detailed information.

After the training is complete, use `model_manager.load_lora` to load the LoRA for inference.




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
